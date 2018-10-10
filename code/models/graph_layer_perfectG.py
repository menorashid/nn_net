from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out

        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.Softmax = nn.Softmax(dim = 1)

        
    def forward(self, x, sim_feat):
        # print x.size()
        # print sim_feat.size()
        
        G = self.get_affinity(sim_feat)
        # print G.size()
        # print torch.min(G), torch.max(G), 1/torch.sum(sim_feat/torch.max(sim_feat))
        # raw_input()

        temp = torch.mm(G,x)
        
        out = torch.mm(temp,self.weight)
        
        return out

    def get_affinity(self,input):
        input = input.view(input.size(0),1)   
        # class_idx = torch.max(input).cpu().numpy()
        # print torch.min(input), torch.max(input)
        input = input/max(1.,torch.max(input))
        # print torch.min(input), torchvisionh.max(input)
        
        G = torch.mm(input,torch.t(input)) 
        # if torch.max(G)==0:
           # print class_idx 
           # G = G+  torch.eye(input.size(0)).cuda()
        # + torch.eye(input.size(0)).cuda()
        # G[torch.eye(G.size(0))>0]=1.
        # if torch.max(input)==0:
        #     print 'hello'
        # print torch.mm(input,torch.t(input))
        # print torch.diagonal(G)


        # D = torch.diagflat(torch.rsqrt(torch.sum(G,dim = 1)))
        # # if is_cuda:
        # D = D.cuda()
        # G = torch.mm(torch.mm(D,G),D)

        # ming = torch.min(G).cpu().numpy()
        # maxg = torch.max(G).cpu().numpy()
        # # print ming, maxg
        # if not(maxg>0 and ming==0):
        #     print ming, maxg
        #     print input
        #     mm = torch.mm(input,torch.t(input))
        #     print torch.min(mm), torch.max(mm)
        #     mm = mm+torch.eye(input.size(0)).cuda()
        #     print torch.min(mm), torch.max(mm)
        #     print torch.min(D), torch.max(D)

        #     raw_input()
        
        sums = torch.sum(G,dim = 1, keepdim = True)
        sums[sums==0]=1.
        G = G/sums
        
        return G



class Graph_Layer_Wrapper(nn.Module):
    def __init__(self,in_size, n_out = None, non_lin = 'HT'):
        super(Graph_Layer_Wrapper, self).__init__()
        self.graph_layer = Graph_Layer(in_size, n_out)
        if non_lin=='HT':
            self.non_linearity = nn.Hardtanh()
        elif non_lin=='rl':
            self.non_linearity = nn.ReLU()
        else:
            error_message = str('non_lin %s not recognized', non_lin)
            raise ValueError(error_message)
    
    def forward(self, x, sim_feat):
        sim_feat = self.non_linearity(sim_feat)
        out = self.graph_layer(x, sim_feat)
        return out

    def get_affinity(self,input):
        return self.graph_layer.get_affinity(input)        