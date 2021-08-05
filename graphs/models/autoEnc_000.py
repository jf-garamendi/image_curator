

import torch.nn as nn
from torchvision.models import *
from torch import squeeze

class Resnet_000(nn.Module):
    def __init__(self, H=512, W=512, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.linear(H*W*3, 128) #AQUI ME HE QUEDADO

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=3),
            nn.ReLU(True),
            nn.Tanh(),

        )

        self.classifier = nn.Sequential(nn.Flatten(),
                                 nn.Linear(64*128, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128,1),
                                 nn.Sigmoid())




    def forward(self, x, classify = False):
        #  wich_output: classification or image

        output = None
        features = self.encoder(x)
        output = self.decoder(features)

        if classify:
            output = self.classifier(features)
            output = squeeze(output, dim=1)

        return output




