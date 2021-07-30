

import torch.nn as nn
from torchvision.models import *

class Resnet_000(nn.Module):
    def __init__(self, model_name, nClasses=2):
        super().__init__()

        model = globals()[model_name]
        self.model = model(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, nClasses)

    def forward(self, x):
        return self.model(x)




