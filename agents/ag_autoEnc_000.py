import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.ag004_image import Ag004_Image
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
plt.ion()

class Ag_AutoEnc_000(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and config.general.device == "cuda") else "cpu")

        # set the manual seed for torch
        self.seed = self.config.general.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # initialize counter
        self.total_autoEnc_epochs = config.training.total_autoEnc_epochs
        self.current_autoEnc_epoch = 0
        self.best_autoEnc_loss = 0

        self.batch_size = config.dataset.batch_size

        self.training_type = config.model.type
        self.initial_lr = config.optimizer.optim_param.lr

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
        model = model().to(self.device)

        return model

    def build_dataLoaders(self, data_config):
        # Define Transforms
        image_transforms = transforms.Compose([
                #todo: mirar en el dataset el histograma de tamaños
                transforms.Resize((data_config.W, data_config.H)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        dataset = datasets.ImageFolder(root = join(data_config.root_dir, data_config.root_for_autoEnc),
                                      transform = image_transforms
                                     )

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
        optimizer = globals()[optim_config.optim_name]

        optimizer =  optimizer(self.model.parameters(), **optim_config.optim_param)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="max",
                                                               factor = optim_config.factor_decay,
                                                               min_lr= optim_config.min_lr,
                                                               patience= optim_config.patience,
                                                               verbose = True)

        return optimizer, scheduler

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
            self.current_autoEnc_epoch = checkpoint['autoEnc_epoch']
            self.best_class_acc = checkpoint['best_class_acc']
            self.best_autoEnc_loss= checkpoint['best_autoEnc_loss']

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
            'autoEnc_epoch': self.current_AutoEnc_epoch,
            'best_acc': self.best_acc
        }

        #just for safety
        if exists(file_name):
            shutil.copyfile(file_name, file_name[:-4]+'_previous.tar')

        #save
        torch.save(chk, file_name)

    def train(self):
        """
        Main training loop
        :return:
        """

        self.train_autoEnc()
        self.train_classifier()

    def train_autoEnc(self):
        epochs_without_improving = 0
        already_reset = False
        for epoch in range(self.autoEnc_current_epoch+1, self.autoEnc_total_epochs+1):
            self.encDec_current_epoch = epoch
            print('EncDec Epoch {}/{}'.format(epoch, self.total_epochs ))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train_autoEnc()  # Set model to training mode
                else:
                    self.model.eval_encDec()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                with torch.set_grad_enabled(phase=='train'):
                    loss= self.run_one_autoEnc_epoch(data_loader=self.data_loaders[phase], phase=phase)


                print('{} Loss: {:.4f} '.format(phase, loss))

                self.TB_writer.add_scalar('encDec Loss: ' + phase, loss, epoch)


                if phase=='val':
                    self.lr_scheduler.step(loss)

                    if loss < self.best_autoEnc_loss:
                        self.best_autoEnc_loss = loss
                        self.save_checkpoint(self.checkpoint_filename)


            self.TB_writer.add_scalar('LR: ', self.optimizer.param_groups[0]['lr'], epoch)

    def run_one_autoEnc_epoch(self, data_loader, phase=''):
        """
                One epoch of training
                :return:
                """
        running_loss = 0.0

        for inputs in tqdm(data_loader, leave=False, desc=phase):
            inputs = inputs.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass.
            outputs = self.model(inputs)

            loss = 0  # torch.tensor(0.0).to(self.device)
            for loss_fn, loss_weight in zip(self.losses_fn, self.losses_weight):
                loss += torch.tensor(loss_weight).to(self.device) * loss_fn(outputs, inputs)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / (len(data_loader) * self.batch_size)

        return epoch_loss