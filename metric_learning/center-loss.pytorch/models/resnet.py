import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101

from .base import CuratorModel
from device import device


class ResnetCuratorModel(CuratorModel):

    IMAGE_SHAPE = (512, 512)

    def __init__(self, num_classes, feature_dim):
        super().__init__(num_classes, feature_dim)

        self.extract_feature = nn.Linear(
            self.feature_dim*4*4, self.feature_dim)
        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = x.view(x.size(0), -1)

        feature = self.extract_feature(x)
        logits = self.classifier(feature) if self.num_classes else None

        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return logits, feature_normed


class Resnet101CuratorModel(CuratorModel):
    IMAGE_SHAPE = (512, 512)
    FEATURE_DIM = 2048

    def __init__(self, num_classes):
        super().__init__(num_classes, self.FEATURE_DIM)
        self.base = resnet101(pretrained=True)

        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = nn.Linear(self.FEATURE_DIM, num_classes)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feature = x.view(x.size(0), -1)

        logits = self.classifier(feature) if self.num_classes else None

        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return logits, feature_normed
