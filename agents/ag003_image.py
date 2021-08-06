import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.ag001_image import Ag001_Image
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
from torch.utils.data import DataLoader, SubsetRandomSampler
cudnn.benchmark = True
from utils.misc import imshow

from graphs.models import *
plt.ion()

class Ag003_Image(Ag001_Image):

    def __init__(self, config):
        super().__init__(config)

        self.training_type = config.model.type
        self.initial_lr = config.optimizer.optim_param.lr

    def train(self):
        """
        Main training loop
        :return:
        """
        epochs_without_improving = 0
        train_complete_model = False

        if (self.training_type == 'scratch'):
            train_complete_model = True

            for param in self.model.parameters():
                param.requires_grad = True
                self.logger.info('Training from scratch \n')




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
                    epochs_without_improving = 0

                    self.best_acc = acc
                    self.save_checkpoint(self.checkpoint_filename)
                else:
                    epochs_without_improving += 1

                    if (not train_complete_model) and (self.training_type == 'fine_tuning') and (epochs_without_improving > 5):
                        train_complete_model = True
                        self.optimizer.param_groups[0]['lr'] = self.initial_lr
                        for param in self.model.parameters():
                            param.requires_grad = True

                        self.logger.info('Unfreeze backbone \n')


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

            loss = 0  # torch.tensor(0.0).to(self.device)
            for loss_fn, loss_weight in zip(self.losses_fn, self.losses_weight):
                loss += torch.tensor(loss_weight).to(self.device) * loss_fn(outputs, labels)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / (len(data_loader) * self.batch_size)
        epoch_acc = running_corrects.item() / (len(data_loader) * self.batch_size)
        return epoch_loss, epoch_acc
