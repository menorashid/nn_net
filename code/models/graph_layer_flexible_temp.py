from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import time

class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None, method = 'cos', k = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        print self.weight.size()
        self.method = method


        
    def forward(self, x, sim_feat, to_keep = None, alpha = None):
        G = self.get_affinity(sim_feat, to_keep = to_keep, alpha = alpha)
        temp = torch.mm(G,x)
        out = torch.mm(temp,self.weight)
        
        return out


    def get_to_keep(self,input_sizes):
        
        k_all = [max(1,size_curr)//self.deno for idx_size_curr,size_curr in enumerate(input_sizes)]
        k_all = int(np.mean(k_all))
        # print k_all
        # raw_input()
        # k_all = []
        # for idx_size_curr,size_curr in enumerate(input_sizes):
        #     deno_curr = max(1,size_curr)//self.deno
        #     k_all += [deno_curr]*size_curr 

        return k_all
    


    def get_affinity(self,input, to_keep = None, alpha = None):

        # if alpha is not None:
        #     print alpha.size()
        #     print input.size()
        #     raw_input()

        if 'cos' in self.method:
            input = F.normalize(input)
        
        G = torch.mm(input,torch.t(input))
        # G = torch.ones(input.size(0),input.size(0)).cuda()
        # print G.size()
        
        if 'zero_self' in self.method:
            eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
            G = G*eye_inv
        
        if alpha is not None:
            alpha = alpha.view(1,alpha.size(0))
            # print alpha.size()
            # print G[0,-10:]
            G = G*alpha
            alpha1 = alpha.view(alpha.size(1),1)
            G = G*alpha1
            # print alpha.size()
            # print alpha1.size()
            # G = torch.mm(alpha1,alpha)
            # sums = torch.sum(summat,dim=1,keepdim=True)

            # print G[10:20,10:20]
            # print G[10,:]
            # print G[0,-10:]
            # raw_input()
        # alpha = None
        if to_keep is not None:
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


        sums = torch.sum(G,dim = 1, keepdim = True)
        sums[sums==0]=1

        G = G/sums
        # print torch.min(G), torch.max(G)    
        return G



class Graph_Layer_Wrapper(nn.Module):
    def __init__(self,in_size, n_out = None, non_lin = 'HT', method = 'cos'):
        super(Graph_Layer_Wrapper, self).__init__()
        self.graph_layer = Graph_Layer(in_size, n_out = n_out, method = method)
        # self.do = nn.Dropout(0.5)
        if non_lin=='HT':
            self.non_linearity = nn.Hardtanh()
        elif non_lin.lower()=='rl':
            self.non_linearity = nn.ReLU()
        else:
            error_message = str('non_lin %s not recognized', non_lin)
            raise ValueError(error_message)
    
    def forward(self, x, sim_feat, to_keep = None, alpha = None):
        sim_feat = self.non_linearity(sim_feat)
        # sim_feat = self.do(sim_feat)
        out = self.graph_layer(x, sim_feat, to_keep = to_keep, alpha = alpha)
        return out

    def get_affinity(self,input,to_keep = None, alpha = None):
        input = self.non_linearity(input)
        # sim_feat = self.do(sim_feat)
        return self.graph_layer.get_affinity(input, to_keep = to_keep, alpha = alpha)        