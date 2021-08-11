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
from PIL import Image
import glob
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets'))
from dataset_with_fixed_classes import CustomImageFolder

TEST_DIR = '/media/totolia/datos_3/photoslurp/dataset/test_pending'
OUTPUT_FOLDER = '/media/totolia/datos_3/photoslurp/dataset/filtered_test_pending'
os.makedirs(os.path.join(OUTPUT_FOLDER, 'approved'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'rejected'), exist_ok=True)
THRESHOLD = 0.54
class Eval:
    def __init__(self, agent):
        self.agent = agent
        self.agent.model.eval()

        print(self.agent.config)

        # Define Transforms
        self.image_transforms = transforms.Compose([
                #todo: mirar en el dataset el histograma de tamaños
                transforms.Resize((self.agent.config.dataset.W, self.agent.config.dataset.H)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def run(self):
        with torch.no_grad():
            prob_list = []
            path_list = []
            for im_path in tqdm(glob.glob(os.path.join(TEST_DIR, "*")), leave=False, desc="evaluating pending samples"):
                im = Image.open(im_path)
                inputs = self.image_transforms(im)
                inputs = inputs.to(self.agent.device).unsqueeze(0)
                
                outputs = self.agent.model(inputs)

                prob_list.append(float(outputs.cpu().float().numpy()))
                path_list.append(im_path)
        
        prob_list = np.array(prob_list)

        # Accepted
        idx_filtered = np.where(prob_list >= THRESHOLD)[0]
        valid_probs = prob_list[idx_filtered]
        valid_paths = [path_list[idx] for idx in idx_filtered]
        
        idx_sort = np.argsort(valid_probs)[::-1]
        sorted_probs = valid_probs[idx_sort]
        sorted_paths = [valid_paths[idx] for idx in idx_sort]

        for i, (prob, im_path) in enumerate(zip(sorted_probs, sorted_paths)):
            im_name = os.path.basename(im_path)
            name_no_ext, ext = os.path.splitext(im_name)

            new_im_path = os.path.join(OUTPUT_FOLDER, 'approved', f'{i:04}_{prob:.2f}_{name_no_ext}{ext}')
            shutil.copyfile(im_path, new_im_path)


        # Rejected
        idx_filtered = np.where(prob_list < THRESHOLD)[0]
        valid_probs = prob_list[idx_filtered]
        valid_paths = [path_list[idx] for idx in idx_filtered]

        idx_sort = np.argsort(valid_probs)
        sorted_probs = valid_probs[idx_sort]
        sorted_paths = [valid_paths[idx] for idx in idx_sort]
        
        for i, (prob, im_path) in enumerate(zip(sorted_probs, sorted_paths)):
            im_name = os.path.basename(im_path)
            name_no_ext, ext = os.path.splitext(im_name)

            new_im_path = os.path.join(OUTPUT_FOLDER, 'rejected', f'{i:04}_{prob:.2f}_{name_no_ext}{ext}')
            shutil.copyfile(im_path, new_im_path)

        




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
