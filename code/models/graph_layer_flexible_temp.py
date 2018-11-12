from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import time
from normalize import Normalize
import numpy as np
from torch.autograd import Variable

class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None, method = 'cos', k = None,
        affinity_dict = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        print self.weight.size()
        self.method = method
        self.affinity_dict = affinity_dict
        if self.affinity_dict is not None:
            self.affinity_dict = Variable(torch.Tensor(np.load(self.affinity_dict))).cuda()
            # print torch.min(self.affinity_dict), torch.max(self.affinity_dict)



        
    def forward(self, x, sim_feat, to_keep = None, alpha = None):

        G = self.get_affinity(sim_feat, to_keep = to_keep, alpha = alpha)
        temp = torch.mm(G,x)
        out = torch.mm(temp,self.weight)
        
        return out


    # def get_to_keep(self,input_sizes):
        
    #     k_all = [max(1,size_curr)//self.deno for idx_size_curr,size_curr in enumerate(input_sizes)]
    #     k_all = int(np.mean(k_all))
    #     # print k_all
    #     # raw_input()
    #     # k_all = []
    #     # for idx_size_curr,size_curr in enumerate(input_sizes):
    #     #     deno_curr = max(1,size_curr)//self.deno
    #     #     k_all += [deno_curr]*size_curr 

    #     return k_all
    


    def get_affinity(self,input, to_keep = None, alpha = None, nosum = False):
        
        # print input.size()
        if 'affinity_dict' in self.method:
            assert self.affinity_dict is not None
            if next(self.parameters()).is_cuda and 'cuda' not in self.affinity_dict.type():
                self.affinity_dict = self.affinity_dict.cuda()
        
            aff_size = self.affinity_dict.size(0)
            in_size = input.size(0)
            
            # expand aff row wise
            index = input.view(in_size,1).expand(in_size,aff_size).long()
            G = torch.gather(self.affinity_dict,0,index)
            
            # expand aff col wise
            index = input.view(1,in_size).expand(in_size,in_size).long()
            G = torch.gather(G,1,index)
            
            
        else:
            if 'cos' in self.method:
                input = F.normalize(input)
            
            G = torch.mm(input,torch.t(input))
            
            if 'zero_self' in self.method:
                eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
                G = G*eye_inv
        


            
        if to_keep is not None:
            if type(to_keep) == type(()):
                to_keep,input_sizes = to_keep
                topk = []
                indices = []
                
                for vid_idx in range(len(input_sizes)):
                    end_idx = sum(input_sizes[:vid_idx+1])
                    start_idx = sum(input_sizes[:vid_idx])
                    size_vid = input_sizes[vid_idx]
                    to_keep_curr = min(size_vid, to_keep)
                    
                    try:
                        # if alpha is not None:
                        #     _, indices_curr = torch.topk(torch.abs(alphamat[:,start_idx:end_idx]), to_keep_curr, dim = 1) 
                        # else:
                        _, indices_curr = torch.topk(torch.abs(G[:,start_idx:end_idx]), to_keep_curr, dim = 1) 
                    except:
                        print G.size()
                        print input_sizes
                        print start_idx, end_idx, size_vid, to_keep_curr, to_keep
                        raw_input()

                    indices_curr = indices_curr+start_idx   
                    
                    indices.append(indices_curr)
                    
                indices = torch.cat(indices,dim = 1)
                topk = torch.gather(G, 1, indices)
                

                G = G*0
                G = G.scatter(1, indices, topk)
            elif type(to_keep)==float:
                # print torch.sum(G==0)
                # print to_keep
                # print 'torch.min(G),torch.max(G)'
                # print torch.min(G),torch.max(G)
                if to_keep>0:
                    G[torch.abs(G)<to_keep] = 0
                else:
                    G[G<to_keep] = 0
                # print 'torch.min(G),torch.max(G)'
                # print torch.min(G),torch.max(G)
                # print torch.sum(G==0)
                # raw_input()


        if not nosum:
            sums = torch.sum(G,dim = 1, keepdim = True)
            sums[sums==0]=1
            G = G/sums


        if alpha is not None:
            assert to_keep is None
            # print G[:2,:2]
            # print torch.min(alpha),torch.max(alpha)
            alpha = alpha.view(1,alpha.size(0))
            G = G*alpha
            diag_vals = torch.diagonal(G)

            alpha1 = alpha.view(alpha.size(1),1)
            G = G*alpha1
            # print G[:2,:2]
            eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
            G = G*eye_inv
            # print G[:2,:2]
            G = G+torch.diag(diag_vals)
            # print G[:2,:2]

            # raw_input()


        # print torch.min(G), torch.max(G)    
        # raw_input()
        return G



class Graph_Layer_Wrapper(nn.Module):
    def __init__(self,in_size, n_out = None, non_lin = 'HT', method = 'cos', aft_nonlin = None,
        affinity_dict = None):

        super(Graph_Layer_Wrapper, self).__init__()
        
        n_out = in_size if n_out is None else n_out

        self.graph_layer = Graph_Layer(in_size, n_out = n_out, method = method,affinity_dict = affinity_dict)

        

        self.aft = None
        if aft_nonlin is not None:
            self.aft = []

            to_pend = aft_nonlin.split('_')
            for tp in to_pend:
                if tp.lower()=='ht':
                    self.aft.append(nn.Hardtanh())
                elif tp.lower()=='rl':
                    self.aft.append(nn.ReLU())
                elif tp.lower()=='l2':
                    self.aft.append(Normalize())
                elif tp.lower()=='ln':
                    self.aft.append(nn.LayerNorm(n_out))
                elif tp.lower()=='bn':
                    self.aft.append(nn.BatchNorm1d(n_out, affine = False, track_running_stats = False))
                else:
                    error_message = str('non_lin %s not recognized', non_lin)
                    raise ValueError(error_message)
            self.aft = nn.Sequential(*self.aft)


        # self.do = nn.Dropout(0.5)
        if non_lin is None:
            self.non_linearity = None
        elif non_lin=='HT':
            self.non_linearity = nn.Hardtanh()
        elif non_lin.lower()=='rl':
            self.non_linearity = nn.ReLU()
        else:
            error_message = str('non_lin %s not recognized', non_lin)
            raise ValueError(error_message)
    
    def forward(self, x, sim_feat, to_keep = None, alpha = None):
        if self.non_linearity is not None:
            sim_feat = self.non_linearity(sim_feat)
        # sim_feat = self.do(sim_feat)
        out = self.graph_layer(x, sim_feat, to_keep = to_keep, alpha = alpha)
        
        if hasattr(self,'aft') and self.aft is not None:
            out = self.aft(out)

        return out

    def get_affinity(self,input,to_keep = None, alpha = None, nosum = False):
        if self.non_linearity is not None:
            input = self.non_linearity(input)
        # input = self.non_linearity(input)
        # sim_feat = self.do(sim_feat)
        return self.graph_layer.get_affinity(input, to_keep = to_keep, alpha = alpha, nosum = nosum)        