

import torch.nn as nn
from torchvision.models import *

class Resnet_001(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        model = globals()[model_name]
        self.backbone = model(pretrained=True)

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
        return self.backbone(x)




