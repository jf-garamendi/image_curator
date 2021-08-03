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

class Ag005_EncDec(Ag004_Image):

    def __init__(self, config):
        super().__init__(config)




    def train(self):
        """
        Main training loop
        :return:
        """

        self.train_encDec()
        self.train_classifier()

    def train_encDec(self):
        epochs_without_improving = 0
        already_reset = False
        for epoch in range(self.encDec_current_epoch+1, self.encDec_total_epochs+1):
            self.encDec_current_epoch = epoch
            print('EncDec Epoch {}/{}'.format(epoch, self.total_epochs ))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train_encDec()  # Set model to training mode
                else:
                    self.model.eval_encDec()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                with torch.set_grad_enabled(phase=='train'):
                    loss= self.run_one_encDec_epoch(data_loader=self.data_loaders[phase], phase=phase)


                print('{} Loss: {:.4f} '.format(phase, loss))

                self.TB_writer.add_scalar('encDec Loss: ' + phase, loss, epoch)


                if phase=='val':
                    self.lr_scheduler.step(loss)

                    if loss < self.best_encDec_loss:
                        epochs_without_improving = 0

                        self.best_acc = acc
                        self.save_checkpoint(self.checkpoint_filename)
                    else:
                        epochs_without_improving += 1
                        self.logger.info('epochs to Unfreeze backbone' + str(5-epochs_without_improving) + '\n')

                        if (not already_reset) and (self.training_type == 'fine_tuning') and (epochs_without_improving > 5):
                            already_reset = True
                            self.optimizer.param_groups[0]['lr'] = self.initial_lr
                            for param in self.model.parameters():
                                param.requires_grad = True

                            self.logger.info('Unfreeze backbone \n')


            self.TB_writer.add_scalar('LR: ', self.optimizer.param_groups[0]['lr'], epoch)
