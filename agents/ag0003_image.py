import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.ag0001_image import Ag001_Image
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

    def train(self):
        """
        Main training loop
        :return:
        """
        epochs_without_improving = 0
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

                    if (self.training_type == 'fine_tuning') and (epochs_without_improving > 5):
                        for param in self.model.parameters():
                            param.requires_grad = True




            self.TB_writer.add_scalar('LR: ', self.optimizer.param_groups[0]['lr'], epoch)
            self.lr_scheduler.step()



