

import torch.nn as nn
from torchvision.models import *
from torch import squeeze

class Resnet_001(nn.Module):
    def __init__(self, model_name, train_mode="fine_tuning"):
        # train_mode: "fine_tuning", "scratch"
        super().__init__()

        model = globals()[model_name]
        pretrain = not (train_mode == "scratch")
        if not pretrain:
            self.logger.info('Training from scratch \n')

        self.backbone = model(pretrained=pretrain)

        # Freeze parameters. LAter, in the agent will be unfreeze depending if the model is training for transfer learning
        # or fine tuning
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.backbone.fc = nn.Sequential(nn.Flatten(),
                                 nn.Linear(num_ftrs, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128,1),
                                 nn.Sigmoid())


    def forward(self, x):
        output = self.backbone(x)
        return squeeze(output, dim=1)




