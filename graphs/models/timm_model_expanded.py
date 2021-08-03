import torch.nn as nn
import timm
from copy import deepcopy
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.resnet import Bottleneck
from torch.nn import Linear, Sequential, AvgPool2d, Conv2d, BatchNorm2d
import torch.nn.functional as F

class TimmPytorchExpandedModel(nn.Module):
    def __init__(self, model_name, nClasses=2, freeze=False):
        super().__init__()

        self.model = timm.create_model(model_name, num_classes=nClasses, pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.layer5 = Bottleneck(
            inplanes=2048,
            planes=512,
            dilation=3,
            stride=2,
            downsample=Sequential(
                AvgPool2d(kernel_size=2, stride=2, padding=0),
                Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )
        self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)
        self.fc = Linear(in_features=2048, out_features=nClasses, bias=True)

        if freeze:
            for l in [
                self.model.layer4,
                # self.model.global_pool,
                # self.model.fc
            ]:
                for param in l.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model.forward_features(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.layer5(x)
        x = self.global_pool(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc(x)
        
        return x




