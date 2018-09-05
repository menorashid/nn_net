import os
import torch
import numpy as np
# from torchvision import transforms
from helpers import util, visualize
import torch.nn as nn

class MultiCrossEntropy(nn.Module):
    def __init__(self,class_weights=None):
        super(MultiCrossEntropy, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim = 1)
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

    def forward(self, gt, pred):
        pred = self.LogSoftmax(pred)

        if self.class_weights is not None:
            assert self.class_weights.size(1)==pred.size(1)
            loss = self.class_weights*-1*gt*pred
        else:
            loss = -1*gt* pred

        loss = torch.sum(loss, dim = 1)
        loss = torch.mean(loss)
        return loss
