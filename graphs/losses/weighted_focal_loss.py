"""
An example for loss class definition, that will be used in the agent
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class Weighted_Focal_Loss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, config=None,alpha=.25, gamma=2):
        super(Weighted_Focal_Loss, self).__init__()

        if (config is None):
            config.alpha = 0.25 
            config.gamma = 2

        if not hasattr(config, 'alpha'):
            config.alpha = .25

        if not hasattr(config, 'gamma'):
            config.gamma = 2



        self.alpha = torch.tensor([1-config.alpha, config.alpha]).cuda()
        self.gamma = config.gamma
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.loss(inputs, targets)

        pt = torch.exp(-BCE_loss)
        idx = targets.long().view(-1)
        at = self.alpha.gather(0, idx)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
