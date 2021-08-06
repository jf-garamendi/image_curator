import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.ag003_image import Ag003_Image
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

class Ag004_Image(Ag003_Image):

    def __init__(self, config):
        super().__init__(config)


    def build_optimizer(self, optim_config):
        if not hasattr(optim_config, 'optim_name'):
            optim_config.optim_name = 'SGD'

        optimizer = globals()[optim_config.optim_name]

        optimizer =  optimizer(self.model.parameters(), **optim_config.optim_param)


        if not hasattr(optim_config, 'factor_decay'):
            optim_config.factor_decay = 0.9
            optim_config.min_lr = optim_config.optim_param.lr

            self.logger.info('No decay for Learning rate \n')

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="max",
                                                               factor = optim_config.factor_decay,
                                                               min_lr= optim_config.min_lr,
                                                               patience= optim_config.patience,
                                                               verbose = True)

        return optimizer, scheduler



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


                if phase=='val':
                    self.lr_scheduler.step(acc)

                    if acc > self.best_acc:
                        epochs_without_improving = 0

                        self.best_acc = acc
                        self.save_checkpoint(self.checkpoint_filename)
                    else:
                        epochs_without_improving += 1
                        self.logger.info('epochs to Unfreeze backbone' + str(5-epochs_without_improving) + '\n')

                        if (not train_complete_model) and (self.training_type == 'fine_tuning') and (epochs_without_improving > 5):
                            train_complete_model = True
                            self.optimizer.param_groups[0]['lr'] = self.initial_lr
                            for param in self.model.parameters():
                                param.requires_grad = True

                            self.logger.info('Unfreeze backbone \n')


            self.TB_writer.add_scalar('LR: ', self.optimizer.param_groups[0]['lr'], epoch)
