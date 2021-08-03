import torch.nn as nn
import timm

class TimmPytorchModel(nn.Module):
    def __init__(self, model_name, nClasses=2):
        super().__init__()

        self.model = timm.create_model(model_name, num_classes=nClasses, pretrained=True)
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, nClasses)

    def forward(self, x):
        return self.model(x)




