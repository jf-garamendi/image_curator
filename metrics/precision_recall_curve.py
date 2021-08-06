"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from torch.nn.modules.activation import Threshold

from torch.nn.modules.linear import Linear
from utils.config import *

from agents import *
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)), '../datasets')
from dataset_with_fixed_classes import CustomImageFolder

TEST_DIR = '/media/totolia/datos_3/photoslurp/dataset/images_oriented/bonprix/test'
class Eval:
    def __init__(self, agent):
        self.agent = agent
        self.agent.model.eval()

        print(self.agent.config)

        # Define Transforms
        image_transforms = transforms.Compose([
                #todo: mirar en el dataset el histograma de tamaños
                transforms.Resize((self.agent.config.dataset.W, self.agent.config.dataset.H)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        dataset = CustomImageFolder(root = TEST_DIR,
                                      transform = image_transforms
                                     )

        self.dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=self.agent.config.dataset.batch_size,
                                  num_workers=self.agent.config.dataset.num_workers)
    
    def run(self):
        # problem: Internal imageFolder order classes for ABC. approved = 0 and reject = 1
        print(self.dataloader.dataset.class_to_idx)
        with torch.no_grad():
            pred_list = []
            gt_list = []
            for inputs, labels in tqdm(self.dataloader, leave=False, desc="val"):
                inputs = inputs.to(self.agent.device)
                labels = labels.to(self.agent.device) # approved have to be 1
                
                outputs = self.agent.model(inputs)

                pred_list += outputs.cpu().float().numpy().tolist()
                gt_list += labels.cpu().float().numpy().tolist()
        
        pred_list = np.array(pred_list)
        gt_list = np.array(gt_list)

        #precision, recall, thresholds = precision_recall_curve(gt_list, pred_list)
        #thresholds = np.append(thresholds, [0])
        precision = []
        recall = []
        thresholds = []
        gt_bool = gt_list > 0
        for th in np.linspace(0.1, 0.9, 8):
            p = pred_list > th
            tp = ((p == gt_bool) & p).sum()
            fp = ((p != gt_bool) & p).sum()
            fn = ((p != gt_bool) & ~p).sum()

            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)

            precision.append(prec)
            recall.append(rec)
            thresholds.append(th)


        np.set_printoptions(threshold=sys.maxsize, precision=2, suppress=True)
        print(np.stack((precision, recall, thresholds), axis=-1))

        no_skill = len(gt_list[gt_list == 1]) / len(gt_list)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Model')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.show(block=True)
        # poss threshold: Precision: 0.869326   Recall: 0.95757576 Thres: 0.54772514




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
    
    ev = Eval(agent)
    ev.run()


if __name__ == '__main__':
    main()
