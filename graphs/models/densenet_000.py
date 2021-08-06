import logging

import torch.nn as nn
from torchvision.models import *
from torch import squeeze

class Densenet_000(nn.Module):
    def __init__(self, model_name, train_mode="fine_tuning"):
        super().__init__()

        self.logger = logging.getLogger("Densenet_000 Model")

        model = globals()[model_name]
        pretrain = not (train_mode == "scratch")
        self.backbone = model(pretrained=pretrain)

        if pretrain:
            # Freeze parameters. LAter, in the agent will be unfreeze depending if the model is training for transfer learning
            # or fine tuning
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.logger.info('Training from scratch \n')




        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.backbone.classifier = nn.Sequential(nn.Flatten(),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128,1),
                                 nn.Sigmoid())


    def forward(self, x):
        output = self.backbone(x)
        return squeeze(output, dim=1)




