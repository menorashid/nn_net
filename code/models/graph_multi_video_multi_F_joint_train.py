from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible_temp import Graph_Layer
from graph_layer_flexible_temp import Graph_Layer_Wrapper
from normalize import Normalize

import numpy as np

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'HT',
                 normalize = [True,True]
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify

        if in_out is None:
            in_out = [2048,64]
        if feat_dim is None:
            feat_dim = [2048,64]

        num_layers = len(in_out)-1
        
        print 'NUM LAYERS', num_layers, in_out

        self.linear_layers = nn.ModuleList()
        self.linear_layers_after = nn.ModuleList()
        for idx_layer_num,layer_num in enumerate(range(num_layers)):

            if non_lin =='HT':
                non_lin_curr = nn.Hardtanh()
            elif non_lin =='RL':
                non_lin_curr = nn.ReLU()
            else:
                error_message = str('Non lin %s not valid', non_lin)
                raise ValueError(error_message)

            idx_curr = idx_layer_num*2

            self.linear_layers.append(nn.Linear(feat_dim[idx_curr], feat_dim[idx_curr+1], bias = False))
            
            last_linear = []
            last_linear.append(non_lin_curr)
            if normalize[0]:
                last_linear.append(Normalize())
            last_linear.append(nn.Dropout(0.5))
            last_linear.append(nn.Linear(feat_dim[idx_curr+1],n_classes))
            last_linear = nn.Sequential(*last_linear)
            self.linear_layers_after.append(last_linear)


        
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[num_layer],n_out = in_out[num_layer+1], non_lin = non_lin, method = method))
        
        last_graph = []
        if non_lin =='HT':
            last_graph.append(nn.Hardtanh())
        elif non_lin =='RL':
            last_graph.append(nn.ReLU())
        else:
            error_message = str('Non lin %s not valid', non_lin)
            raise ValueError(error_message)

        if normalize[1]:
            last_graph.append(Normalize())

        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes))
        last_graph = nn.Sequential(*last_graph)
        self.last_graph = last_graph

        self.num_branches = num_layers+1
        
        # if type(num_switch)==type(1):
        #     num_switch = [num_switch]*self.num_branches

        # self.num_switch = num_switch
        # self.epoch_counters = [0]* self.num_branches
        # self.focus = focus
        # self.epoch_last = 0
        
        print 'self.num_branches', self.num_branches
        # print 'self.num_switch', self.num_switch
        # print 'self.epoch_counters', self.epoch_counters
        # print 'self.focus', self.focus
        # print 'self.epoch_last', self.epoch_last


    def get_to_keep(self,input_sizes):
        
        k_all = [max(1,size_curr//self.deno) for idx_size_curr,size_curr in enumerate(input_sizes)]
        k_all = int(np.mean(k_all))
        
        return k_all
        
    # def get_graph_out(self,input,input_sizes,graph_idx=None,printit=False):
    #     is_cuda = next(self.parameters()).is_cuda
        
    #     if self.sparsify:
    #         to_keep = (self.get_to_keep(input_sizes),input_sizes)
    #     else:
    #         to_keep = None

    #     if is_cuda:
    #         input = input.cuda()
        
    #     if graph_idx is None:
    #         graph_idx = len(self.graph_layers)-1

    #     input_graph = input
    #     for idx_graph_layer in range(graph_idx+1):
    #         graph_layer = self.graph_layers[idx_graph_layer]
    #         layer_curr = self.linear_layers[idx_graph_layer][0]

    #         if printit:
    #             print 'self.linear_layers[idx_graph_layer]',self.linear_layers[idx_graph_layer]
    #             print 'layer_curr',layer_curr
    #             print 'input_graph.size()',input_graph.size()

    #         features_out = layer_curr(input_graph)    
    #         input_graph = graph_layer(input_graph, features_out,to_keep = to_keep)

    #     if printit:
    #         print 'input_graph.size(),idx_graph_layer, graph_idx',input_graph.size(), graph_idx

    #     return input_graph
        
    # def forward_graph(self, input_chunks):

    #     pmf = []
    #     x_all = []

    #     is_cuda = next(self.parameters()).is_cuda
    #     # print 'Graph branch'

    #     for input in input_chunks:
    #         input_sizes = [input_curr.size(0) for input_curr in input]
            
            
    #         input = torch.cat(input,0)

    #         out_graph = self.get_graph_out(input,input_sizes,printit=False)
    #         # print out_graph.size()

    #         x = self.last_graph(out_graph)
    #         x_all.append(x)
            
    #         for idx_sample in range(len(input_sizes)):
    #             if idx_sample==0:
    #                 start = 0
    #             else:
    #                 start = sum(input_sizes[:idx_sample])
    #             end = start+input_sizes[idx_sample]
    #             x_curr = x[start:end,:]
    #             pmf += [self.make_pmf(x_curr).unsqueeze(0)]
        
        
    #     return x_all, pmf


    # def forward_linear(self, input_chunks, idx_linear):
        
        pmf = []
        x_all = []
        
        is_cuda = next(self.parameters()).is_cuda
        # print 'Linear branch',idx_linear

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)
            if is_cuda:
                input = input.cuda()

            # print self.linear_layers[idx_linear]
            
            out_graph = self.get_graph_out(input,input_sizes,idx_linear-1)

            x = self.linear_layers[idx_linear](out_graph)
            x_all.append(x)
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf += [self.make_pmf(x_curr).unsqueeze(0)]
        
        return x_all, pmf

    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        # if epoch_num is not None:
        #     if epoch_num>self.epoch_last:
        #         self.epoch_counters[self.focus]+=1

        #     if self.epoch_counters[self.focus]>0 and not self.epoch_counters[self.focus]%self.num_switch[self.focus]:
        #         self.epoch_counters[self.focus]=0
        #         self.focus = (self.focus+1)%self.num_branches
        #         print 'FOCUS',epoch_num,self.focus, self.epoch_last,self.epoch_counters

        # if branch_to_test>-1:
        #     self.focus = branch_to_test


        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        if self.graph_size is None:
            graph_size = len(input)
        elif self.graph_size=='rand':
            import random
            graph_size = random.randint(1,len(input))
        else:
            graph_size = min(self.graph_size, len(input))

        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda
        # print 'Graph branch'
        
        pmf_all = [[] for i in range(self.num_branches)]
        x_all_all = [[] for i in range(self.num_branches)]
        
        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if self.sparsify:
                to_keep = (self.get_to_keep(input_sizes),input_sizes)
            else:
                to_keep = None

            if is_cuda:
                input = input.cuda()
            
            assert len(self.graph_layers)==(self.num_branches-1)
            # print 'len(self.graph_layers)',len(self.graph_layers)

            input_graph = input
            for col_num in range(len(self.graph_layers)):
                graph_layer = self.graph_layers[col_num]
                linear_layer = self.linear_layers[col_num]
                linear_layer_after = self.linear_layers_after[col_num]
                
                # print 'col_num',col_num
                # print 'input_graph.size()',input_graph.size()
                # print 'self.linear_layers[col_num]',self.linear_layers[col_num]
                # print 'self.graph_layers[col_num]',self.graph_layers[col_num]
                # print 'self.linear_layers_after[col_num]',self.linear_layers_after[col_num]


                feature_out = self.linear_layers[col_num](input_graph)
                input_graph = self.graph_layers[col_num](input_graph, feature_out, to_keep = to_keep)
                out_col = self.linear_layers_after[col_num](feature_out)
                x_all_all[col_num].append(out_col)
                
                # print 'feature_out.size()',feature_out.size()
                # print 'input_graph.size()',input_graph.size()
                # print 'out_col.size()',out_col.size()
                # raw_input()

            x_all_all[len(self.graph_layers)].append(self.last_graph(input_graph))

            for branch_num in range(len(x_all_all)):
                # print ' branch_num', branch_num
                # print ' len(x_all_all[branch_num])', len(x_all_all[branch_num])

                x = x_all_all[branch_num][-1]
                # print ' x.size()', x.size()

                for idx_sample in range(len(input_sizes)):
                    if idx_sample==0:
                        start = 0
                    else:
                        start = sum(input_sizes[:idx_sample])

                    end = start+input_sizes[idx_sample]
                    x_curr = x[start:end,:]
                    pmf_all[branch_num] += [self.make_pmf(x_curr).unsqueeze(0)]
                # print ' len(pmf_all[branch_num]), pmf_all[branch_num][0].size()', len(pmf_all[branch_num]), pmf_all[branch_num][0].size()
            # raw_input()
            
        if strip:
            for idx_pmf, pmf in enumerate(pmf_all):
                assert len(pmf)==1
                pmf_all[idx_pmf] = pmf[0].squeeze()

        for idx_x, x in enumerate(x_all_all):
            x_all_all[idx_x] = torch.cat(x,dim=0)
        

        if branch_to_test>-1:
            x_all_all = x_all_all[branch_to_test]
            pmf_all = pmf_all[branch_to_test]


        # if epoch_num is not None:
        #     self.epoch_last = epoch_num


        # for branch_num in range(self.num_branches):
        #     print branch_num
        #     print x_all_all[branch_num].size()
        #     print len(pmf_all[branch_num])
        #     for i in range(len(pmf_all[branch_num])):
        #         print pmf_all[branch_num][i].size()



        # if (self.focus+1)<self.num_branches:
        #     x, pmf = self.forward_linear(input_chunks,self.focus)
        # else:
        #     x, pmf = self.forward_graph(input_chunks)

        # if strip:
        #     assert len(pmf)==1
        #     pmf = pmf[0].squeeze()

        # x = torch.cat(x,dim=0)
        
        # if epoch_num is not None:
        #     self.epoch_last = epoch_num

        # # raw_input()

        if ret_bg:
            return x_all_all, pmf_all, None
        else:
            return x_all_all, pmf_all
        

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        
        pmf,_ = torch.sort(x, dim=0, descending=True)
        pmf = pmf[:k,:]
        
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        return pmf

    # def get_similarity(self,input,idx_graph_layer = 0):

    #     is_cuda = next(self.parameters()).is_cuda
    #     input_sizes = [input_curr.size(0) for input_curr in input]
    #     input = torch.cat(input,0)

    #     if self.sparsify:
    #         to_keep = (self.get_to_keep(input_sizes),input_sizes)
    #     else:
    #         to_keep = None

    #     print to_keep    
        
    #     if is_cuda:
    #         input = input.cuda()
        
    #     feature_out = self.linear_layers[idx_graph_layer][0](input)
    #     sim_mat = self.graph_layers[idx_graph_layer].get_affinity(feature_out,to_keep = to_keep)
        
    #     return sim_mat
    
    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].graph_layer.weight.grad
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'HT',
                 normalize = [True,True]
                 ):
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin, normalize)
        
    def get_lr_list(self, lr):
        
        
        lr_list = []

        lr_list+= [{'params': [p for p in self.model.linear_layers.parameters() if p.requires_grad], 'lr': lr[0]}]

        lr_list+= [{'params': [p for p in self.model.linear_layers_after.parameters() if p.requires_grad], 'lr': lr[0]}]
        
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[1]}]        
        # lr_list+= [{'params': [p for p in self.model.last_linear.parameters() if p.requires_grad], 'lr': lr[2]}]
        lr_list+= [{'params': [p for p in self.model.last_graph.parameters() if p.requires_grad], 'lr': lr[1]}]

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
    # print output.shape


    print output.data.shape

if __name__=='__main__':
    main()