"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse

from numpy.core.shape_base import block
from utils.config import *

from agents import *
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets'))
from dataset_with_fixed_classes import CustomImageFolder

class ImageFolderWithPaths(CustomImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0], )

TEST_DIR = '/media/totolia/datos_3/photoslurp/dataset/images_oriented/bonprix/test'
class FromLoss:
    def __init__(self, agent):
        self.agent = agent
        self.agent.model.eval()

        # Define Transforms
        image_transforms = transforms.Compose([
                #todo: mirar en el dataset el histograma de tamaños
                transforms.Resize((self.agent.config.dataset.W, self.agent.config.dataset.H)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        dataset = ImageFolderWithPaths(root = TEST_DIR,
                                      transform = image_transforms
                                     )

        self.dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=self.agent.config.dataset.batch_size,
                                  num_workers=self.agent.config.dataset.num_workers)
    
    def __loss(self, outputs, labels):
        loss = 0
        for loss_fn, loss_weight in zip(self.agent.losses_fn, self.agent.losses_weight):
            loss += torch.tensor(loss_weight).to(self.agent.device) *  loss_fn(outputs, labels)

        return loss

    def run(self, topk):
        # problem: Internal imageFolder order classes for ABC. approved = 0 and reject = 1
        print(self.dataloader.dataset.class_to_idx)
        names_list = []
        loss_list = []
        gt_list = []

        with torch.no_grad():
            for inputs, labels, paths in tqdm(self.dataloader, leave=False, desc="val"):
                inputs = inputs.to(self.agent.device)
                labels = labels.to(self.agent.device).float() # approved have to be 1
                
                outputs = self.agent.model(inputs)
                loss = [float(self.__loss(outputs[i:i+1], labels[i:i+1]).cpu().numpy()) for i in range(inputs.shape[0])]

                names_list += paths
                loss_list += loss
                gt_list += labels.cpu().numpy().tolist()

        loss_list = np.array(loss_list)

        # Accept
        filter_idx = np.where(np.array(gt_list) == 1)[0]
        loss_list_accepted = loss_list[filter_idx]
        names_list_accepted = [names_list[idx] for idx in filter_idx]

        idx_sort = np.argsort(loss_list_accepted)
        sorted_loss_list_accepted = loss_list_accepted[idx_sort]
        sorted_names_list_accepted = [names_list_accepted[idx] for idx in idx_sort]

        # Reject
        filter_idx = np.where(np.array(gt_list) == 0)[0]
        loss_list_rejected = loss_list[filter_idx]
        names_list_rejected = [names_list[idx] for idx in filter_idx]

        idx_sort = np.argsort(loss_list_rejected)
        sorted_loss_list_rejected = loss_list_rejected[idx_sort]
        sorted_names_list_rejected = [names_list_rejected[idx] for idx in idx_sort]

        # Best top-k
        best_accepted = dict(zip(sorted_names_list_accepted[:topk],  sorted_loss_list_accepted[:topk]))
        worst_accepted = dict(zip(sorted_names_list_accepted[-topk:],  sorted_loss_list_accepted[-topk:]))

        best_rejected = dict(zip(sorted_names_list_rejected[:topk],  sorted_loss_list_rejected[:topk]))
        worst_rejected = dict(zip(sorted_names_list_rejected[-topk:],  sorted_loss_list_rejected[-topk:]))

        return {
            'accepted': {
                'best': best_accepted,
                'worst': worst_accepted,
            },

            'rejected': {
                'best': best_rejected,
                'worst': worst_rejected,
            }
        }




def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.general.agent]
    agent = agent_class(config)
    
    ev = FromLoss(agent)
    topk = 10
    ev_result = ev.run(topk=topk)
    fig, axs = plt.subplots(4, topk)
    for i, (
        (accept_best_key, accept_best_value),
        (accept_worst_key, accept_worst_value),
        (reject_best_key, reject_best_value),
        (reject_worst_key, reject_worst_value),
    ) in enumerate(zip(
        ev_result['accepted']['best'].items(),
        ev_result['accepted']['worst'].items(),
        ev_result['rejected']['best'].items(),
        ev_result['rejected']['worst'].items()
    )):
        axs[0, i].imshow(cv2.imread(accept_best_key)[..., ::-1])
        axs[0, i].set_title(f'Accept/Best {accept_best_value:.2f}')
        axs[0, i].axis('off')

        axs[1, i].imshow(cv2.imread(accept_worst_key)[..., ::-1])
        axs[1, i].set_title(f'Accept/Worst {accept_worst_value:.2f}')
        axs[1, i].axis('off')

        axs[2, i].imshow(cv2.imread(reject_best_key)[..., ::-1])
        axs[2, i].set_title(f'Reject/Best {reject_best_value:.2f}')
        axs[2, i].axis('off')

        axs[3, i].imshow(cv2.imread(reject_worst_key)[..., ::-1])
        axs[3, i].set_title(f'Reject/Worse {reject_worst_value:.2f}')
        axs[3, i].axis('off')
    plt.show(block=True)

    print(json.dumps(ev_result, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
