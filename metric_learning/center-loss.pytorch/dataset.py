import os
import random
import tarfile
from math import ceil, floor

from torch.utils import data
import numpy as np
import glob
import traceback

from utils import image_loader, download
def create_datasets(images_root, train_val_split=0.9):
    names = os.listdir(images_root)
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    training_set = []
    validation_set = []
    for klass, name in enumerate(names):
        def add_class(image):
            image_path = os.path.join(images_root, name, image)
            return (image_path, klass, name)

        images_of_person = os.listdir(os.path.join(images_root, name))
        total = len(images_of_person)

        training_set += map(
                add_class,
                images_of_person[:ceil(total * train_val_split)])
        validation_set += map(
                add_class,
                images_of_person[floor(total * train_val_split):])

    return training_set, validation_set, len(names)

class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        try:
            image = image_loader(self.datasets[index][0])
            if self.transform:
                image = self.transform(image)
            return (image, self.datasets[index][1], self.datasets[index][2])
        except:
            traceback.print_exc()
            return self.__getitem__(np.random.randint(0, len(self)))

class PairedDataset(data.Dataset):

    def __init__(self, dataroot, pairs_cfg, transform=None, loader=None):
        self.dataroot = dataroot
        self.pairs_cfg = pairs_cfg
        self.transform = transform
        self.loader = loader if loader else image_loader

        self.image_names_a = []
        self.image_names_b = []
        self.matches = []

        self._prepare_dataset()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        return (self.transform(self.loader(self.image_names_a[index])),
                self.transform(self.loader(self.image_names_b[index])),
                self.matches[index])

    def _prepare_dataset(self):
        raise NotImplementedError

class LFWPairedDataset(PairedDataset):
    def _prepare_dataset(self):
        pairs = self._read_pairs(self.pairs_cfg)

        for pair in pairs:
            if len(pair) == 3:
                match = True
                name1, name2, index1, index2 = \
                    pair[0], pair[0], int(pair[1]), int(pair[2])

            else:
                match = False
                name1, name2, index1, index2 = \
                    pair[0], pair[2], int(pair[1]), int(pair[3])

            self.image_names_a.append(os.path.join(
                    self.dataroot, 'lfw-deepfunneled',
                    name1, "{}_{:04d}.jpg".format(name1, index1)))

            self.image_names_b.append(os.path.join(
                    self.dataroot, 'lfw-deepfunneled',
                    name2, "{}_{:04d}.jpg".format(name2, index2)))
            self.matches.append(match)

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return pairs

class CuratorPairedDataset(PairedDataset):
    
    def _prepare_dataset(self):
        set_ = os.listdir(self.dataroot)
        datas = {}
        for s in set_:
            datas[s] = glob.glob(os.path.join(self.dataroot, s, "*.jpg"))

        # Eq
        for i in range(5000):
            a = set_[np.random.randint(0, len(set_))]

            match = True

            folder_a_path = datas[a]
            file_a_path = folder_a_path[np.random.randint(0, len(folder_a_path))]
            file_b_path = folder_a_path[np.random.randint(0, len(folder_a_path))]

            self.image_names_a.append(file_a_path)

            self.image_names_b.append(file_b_path)
            self.matches.append(match)

        # Diff
        for i in range(5000):
            a = set_[np.random.randint(0, len(set_))]
            b = set_[np.random.randint(0, len(set_))]

            match = a == b

            folder_a_path = datas[a]
            file_a_path = folder_a_path[np.random.randint(0, len(folder_a_path))]

            folder_b_path = datas[b]
            file_b_path = folder_b_path[np.random.randint(0, len(folder_b_path))]

            self.image_names_a.append(file_a_path)

            self.image_names_b.append(file_b_path)
            self.matches.append(match)

