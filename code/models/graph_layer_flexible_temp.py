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


        
    def forward(self, x, sim_feat, to_keep = None):
        G = self.get_affinity(sim_feat, to_keep)
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
    


    def get_affinity(self,input, to_keep = None):

        # if to_keep is not None:
        #     print to_keep
        #     print input.size()
        #     # print sum(to_keep)
        #     raw_input()

            # input=input[:5]
        #     to_keep = 2

        if 'cos' in self.method:
            input = F.normalize(input)
        
        G = torch.mm(input,torch.t(input)) 
        
        if 'zero_self' in self.method:
            eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
            G = G*eye_inv
        
        
        

        
        if to_keep is not None:
            to_keep,input_sizes = to_keep
            topk = []
            indices = []
            
            # G_org = G
            # G_abs = torch.abs(G)

            for vid_idx in range(len(input_sizes)):
                end_idx = sum(input_sizes[:vid_idx+1])
                start_idx = sum(input_sizes[:vid_idx])
                size_vid = input_sizes[vid_idx]
                to_keep_curr = min(size_vid, to_keep)
                # print vid_idx, start_idx,end_idx, size_vid, to_keep_curr
                # G_rel = G[:,start_idx:end_idx]
                # print G_rel.size()
                try:
                    _, indices_curr = torch.topk(torch.abs(G[:,start_idx:end_idx]), to_keep_curr, dim = 1) 
                except:
                    print G.size()
                    print input_sizes
                    print start_idx, end_idx, size_vid, to_keep_curr, to_keep
                    raw_input()

                indices_curr = indices_curr+start_idx   
                # print type(indices_curr), indices_curr.size()
                # print to_keep_curr
                # print topk_curr.size(),torch.min(topk_curr),torch.max(topk_curr)
                # topk_curr = torch.gather(G_org, 1, indices_curr)
                # # G_org[indices_curr.tolist()]
                # # torch.index_select(G_org, 1, indices_curr)
                # print topk_curr.size(),torch.min(topk_curr),torch.max(topk_curr)



                
                indices.append(indices_curr)
                # topk.append(topk_curr)
                # print indices_curr.size()
                # print topk_curr.size()
                # print indices_curr[0]
                # raw_input()

            # torch.cat(topk,dim = 1)
            indices = torch.cat(indices,dim = 1)
            topk = torch.gather(G, 1, indices)
            
            # print G.size()
            # print topk.size()
            # print indices.size()
            # raw_input()


            G = G*0
            G = G.scatter(1, indices, topk)



        G = G/torch.sum(G,dim = 1, keepdim = True)
        
        # if to_keep is not None:        
        #     print G

        #     raw_input()        
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
    
    def forward(self, x, sim_feat, to_keep = None):
        sim_feat = self.non_linearity(sim_feat)
        # sim_feat = self.do(sim_feat)
        out = self.graph_layer(x, sim_feat, to_keep = to_keep)
        return out

    def get_affinity(self,input,to_keep = None):
        input = self.non_linearity(input)
        # sim_feat = self.do(sim_feat)
        return self.graph_layer.get_affinity(input, to_keep = to_keep)        