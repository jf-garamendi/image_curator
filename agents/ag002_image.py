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

class Ag002_Image(Ag001_Image):

    def __init__(self, config):
        super().__init__(config)




    ########################################################################
    # SET FUNCTIONS

    def build_dataLoaders(self, data_config):
        # Define Transforms
        image_transforms = transforms.Compose([
                #todo: mirar en el dataset el histograma de tamaños
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

