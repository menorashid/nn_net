from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
# from graph_layer_flexible_temp import Graph_Layer
# from graph_layer_flexible_temp import Graph_Layer_Wrapper
# from normalize import Normalize

import numpy as np

class Wsddn(nn.Module):
    def __init__(self,
                n_classes,
                deno = None,
                in_out = None):
        super(Wsddn, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        
        if in_out is None:
            in_out = [2048,512]
        
        self.linear_layer = []
        # nn.Sequential(*)
        self.linear_layer.append(nn.Linear(in_out[0], in_out[1], bias = True))
        self.linear_layer.append(nn.ReLU())
        self.linear_layer.append(nn.Dropout(0.5))
        self.linear_layer = nn.Sequential(*self.linear_layer)
        
        self.det_branch = nn.Sequential(*[nn.Linear(in_out[1],self.num_classes), nn.Softmax(dim=0)])
        self.class_branch = nn.Sequential(*[nn.Linear(in_out[1],self.num_classes), nn.Softmax(dim=1)])

    def forward(self, input):
        # print input.size()

        x = self.linear_layer(input)
        
        x_class = self.class_branch(x) #n_instances x n_classes softmax along classes
        x_det = self.det_branch(x) #n_instances x n_classes softmax along instances

        x = x_class*x_det

        pmf = self.make_pmf(x)
        # print pmf.size()

        return x_class, pmf

    def make_pmf(self, x):
        if self.deno is None:
            k = x.size(0)
            pmf = x
        else:
            k = max(1,x.size(0)//self.deno)
            pmf,_ = torch.sort(x, dim=0, descending=True)
            pmf = pmf[:k,:]
        
        # print torch.min(x), torch.max(x)
        pmf = torch.sum(pmf, dim = 0)
        # print torch.min(pmf), torch.max(pmf)
        # /k
        # print pmf.size()
        # raw_input()
        return pmf


class Network:
    def __init__(self, n_classes, deno = None, in_out = None, init = False):
        model = Wsddn(n_classes, deno,in_out)

        self.model = model


    def get_lr_list(self, lr):
        modules = [self.model.linear_layer, self.model.class_branch, self.model.det_branch, ]
        lr_list = []
        # [p for p in self.model.last_graphs.parameters() if p.requires_grad]
        for idx_module, module in enumerate(modules):
            lr_list += [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr[idx_module]}]
        
        return lr_list


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 20, deno = 8)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((16,2048))
    input = torch.Tensor(input).cuda()
    input = Variable(input)
    output,pmf = net.model(input)
    print output.shape, pmf.shape

if __name__=='__main__':
    main()