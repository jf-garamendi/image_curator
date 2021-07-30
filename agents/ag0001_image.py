import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent
from os.path import join, exists
import matplotlib.pyplot as plt

# import your classes here

from torch.utils.tensorboard import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics
from utils.dirs import create_dirs
from torchvision import transforms, datasets
from torch.nn import *
from torch.optim import *
from torch.utils.data import DataLoader, SubsetRandomSampler
cudnn.benchmark = True
from utils.misc import imshow

from graphs.models import *
from graphs.losses import *
plt.ion()

class Ag001_Image(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and config.general.device == "cuda") else "cpu")

        # set the manual seed for torch
        self.seed = self.config.general.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # initialize counter
        self.total_epochs = config.training.total_epochs
        self.current_epoch = 0
        self.best_acc = 0

        self.batch_size = config.dataset.batch_size

        ######################
        # Names and Folders
        self.exp_name = config.exp_name
        self.checkpoint_filename = join(self.checkpoint_dir, config.checkpoint.checkpoint_filename)

        # define models
        self.model = self.build_model(config.model)

        # define data_loader
        self.data_loaders = self.build_dataLoaders(config.dataset)

        # optimizers
        self.optimizer, self.lr_scheduler = self.build_optimizer(config.optimizer)

        # define loss
        self.losses_fn, self.losses_weight = self.build_losses(config.losses)

        # Tensor Board writers
        self.TB_writer = SummaryWriter(join(self.tensorboard_for_all_exp, self.exp_name))

        #################################
        # If exists checkpoints, load them
        self.load_checkpoint(self.checkpoint_filename)


    ########################################################################
    # SET FUNCTIONS
    def build_model(self, model_config):
        model = globals()[model_config.model]
        model = model(model_config.model_name).to(self.device)

        return model
    def build_dataLoaders(self, data_config):
        # Define Transforms
        image_transforms = transforms.Compose([
                #todo: mirar en el dataset el histograma de tamaños
                transforms.Resize((data_config.W, data_config.H)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        dataset = datasets.ImageFolder(root = data_config.root_dir ,
                                      transform = image_transforms
                                     )

        self.idx2class = {v: k for k, v in dataset.class_to_idx.items()}
        self.class_names = dataset.classes

        print("Found Following Classes : ", dataset.classes)

        # split into train and validation
        dataset_size = len(dataset)
        dataset_indices = list(range(dataset_size))

        # shuffle the list of indices
        np.random.shuffle(dataset_indices)

        # Split the indices based on train-val percentage
        val_split_index = int(np.floor((1-data_config.perc_val) * dataset_size))

        # Slice the lists to obtain 2 lists of indices, one for train and other for val.
        # From 0 to index goes for training, the last perc_val goes to val
        train_idx = dataset_indices[:val_split_index]
        val_idx = dataset_indices[val_split_index:]

        # create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=data_config.batch_size, sampler=train_sampler,
                                  num_workers=data_config.num_workers)
        val_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=data_config.batch_size, sampler=val_sampler,
                                num_workers=data_config.num_workers)

        dataloaders = {'train': train_loader,
                       'val': val_loader}

        return dataloaders

    def build_optimizer(self, optim_config):
        if not hasattr(optim_config, 'optim_name'):
            optim_config.optim_name = 'SGD'

        optimizer = globals()[optim_config.optim_name]

        optimizer =  optimizer(self.model.parameters(), **optim_config.optim_param)

        # Decay LR by a factor of 0.1 every 7 epochs
        if not hasattr(optim_config, 'gamma_decay'):
            optim_config.gamma_decay = 1
            optim_config.decay_after_epochs = 1000
            self.logger.info('No decay for Learning rate \n')

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_config.decay_after_epochs, gamma=optim_config.gamma_decay, verbose=True)

        return optimizer, exp_lr_scheduler

    def build_losses(self, loss_config):

        losses_fn = []
        losses_weight = []
        for loss, weight in zip(loss_config.fn, loss_config.weights):
            losses_fn.append(globals()[loss](loss_config))
            losses_weight.append(weight)

        return losses_fn, losses_weight

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        # if exists, load checkpoint
        if exists(file_name):
            checkpoint = torch.load(file_name)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_acc']

            self.logger.info('** Checkpoint ' + file_name + ' loaded \n')


    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """

        chk = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_acc': self.best_acc
        }

        #just for safety
        if exists(file_name):
            shutil.copyfile(file_name, file_name[:-4]+'_previous.tar')

        #save
        torch.save(chk, file_name)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch+1, self.total_epochs+1):
            self.current_epoch = epoch
            print('Epoch {}/{}'.format(epoch, self.total_epochs ))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                with torch.set_grad_enabled(phase=='train'):
                    loss, acc = self.run_one_epoch(data_loader=self.data_loaders[phase], phase=phase)


                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))

                self.TB_writer.add_scalar('Loss: ' + phase, loss, epoch)
                self.TB_writer.add_scalar('Accuracy: ' + phase, acc, epoch)


                if phase=='val' and acc > self.best_acc:
                    self.best_acc = acc

                    self.save_checkpoint(self.checkpoint_filename)


            self.TB_writer.add_scalar('LR: ', self.optimizer.param_groups[0]['lr'], epoch)
            self.lr_scheduler.step()





    def run_one_epoch(self, data_loader, phase=''):
        """
        One epoch of training
        :return:
        """
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(data_loader, leave=False, desc=phase):
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass.
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = 0#torch.tensor(0.0).to(self.device)
            for loss_fn, loss_weight in zip(self.losses_fn, self.losses_weight):
                loss += torch.tensor(loss_weight).to(self.device) *  loss_fn(outputs, labels)

            if phase=='train':
                loss.backward()
                self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


        epoch_loss = running_loss /(len(data_loader)*self.batch_size)
        epoch_acc = running_corrects.item()/(len(data_loader)*self.batch_size)
        return epoch_loss, epoch_acc

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Finalize the model after " + str(self.current_epoch) + " epochs of training, with best accuracy of " + str(self.best_acc))
        print(
            "Finalize the model after " + str(self.current_epoch) + " epochs of training, with best accuracy of " + str(
                self.best_acc))

        if exists(self.checkpoint_filename):
            self.visualize_model(num_images=6)

            plt.ioff()
            plt.show()


    def visualize_model(self, num_images=6):
        model = self.model

        #load the best checkpoint
        self.load_checkpoint(self.checkpoint_filename)

        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.data_loaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}, GT: {}'.format(self.class_names[preds[j]], self.class_names[labels[i]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
