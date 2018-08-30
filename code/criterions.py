import os
import torch
import numpy as np
# from torchvision import transforms
from helpers import util, visualize
import torch.nn as nn

class MultiCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiCrossEntropy, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim = 1)
        

    def forward(self, gt, pred):
    	# print pred.size()
        pred = self.LogSoftmax(pred)
        # print pred.size()
        loss = -1*gt* pred
        # print loss.size()
        loss = torch.sum(loss, dim = 1)
        # print loss.size()
        loss = torch.mean(loss)
        # print  loss.size()
        return loss
