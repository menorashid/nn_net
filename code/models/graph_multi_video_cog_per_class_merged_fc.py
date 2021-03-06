from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
# from graph_layer_flexible import Graph_Layer
from graph_layer_flexible import Graph_Layer_Wrapper
from normalize import Normalize
import random
import numpy as np
import time

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 # feat_dim = None,
                 sparsify = 0.5,
                 aft_nonlin = 'RL',
                 sigmoid = False,
                 feat_ret= False):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.feat_ret = feat_ret
        
        self.deno = deno
        self.graph_size = 1
        self.sparsify = sparsify
        
        if in_out is None:
            in_out = [2048,64]
        
        # if feat_dim is None:
        #     feat_dim = [2048,256]

        # self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)
        self.graph_layers = nn.ModuleList()
        for class_num in range(self.num_classes):
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], aft_nonlin = aft_nonlin, type_layer = 'cooc', non_lin = None))
        # self.graph_layers = nn.ModuleList(*self.graph_layers)
        
        # self.last_graph = nn.ModuleList()
        # for class_num in range(self.num_classes):
        last_graph = []
        # if aft_nonlin is not None:
        #     to_pend = aft_nonlin.split('_')
        #     for tp in to_pend:
        #         if tp.lower()=='ht':
        #             last_graph.append(nn.Hardtanh())
        #         elif tp.lower()=='rl':
        #             last_graph.append(nn.ReLU())
        #         elif tp.lower()=='l2':
        #             last_graph.append(Normalize())
        #         elif tp.lower()=='ln':
        #             last_graph.append(nn.LayerNorm(n_out))
        #         elif tp.lower()=='bn':
        #             last_graph.append(nn.BatchNorm1d(n_out, affine = False, track_running_stats = False))
        #         elif tp.lower()=='sig':
        #             last_graph.append(nn.Sigmoid())
        #         else:
        #             error_message = str('non_lin %s not recognized', non_lin)
        #             raise ValueError(error_message)
            
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1]*self.num_classes,self.num_classes, bias = True))
        if sigmoid:
            last_graph.append(nn.Sigmoid())
        self.last_graph = nn.Sequential(*last_graph)
            # self.last_graphs.append(last_graph)
        
    def forward(self, input_chunks, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if type(input_chunks)!=type([]):
            input_chunks = [input_chunks]
            strip = True

        identity = False
        method = None
        
        # print 'self.graph_size' , self.graph_size
        assert self.graph_size ==1
        graph_size = self.graph_size

        # print 'graph_size', graph_size, self.training
        
        # input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda

        pmf_all = []
        x_all = []
        out_graph_all = []
        input_sizes_all = np.array([input_curr[0].size(0) for input_curr in input_chunks])
        
        for input_curr in input_chunks:
            # print len(input_chunks)
            # print 'len(input_curr)',len(input_curr)
            graphs = input_curr[1]
            input = input_curr[0]

            if is_cuda:
                input = input.cuda()
                graphs = graphs.cuda()
            
            # print len(graphs)
            # print graphs.size()
            assert graphs.size(0)==self.num_classes
            
            # x_curr = []
            out_graph = []
            
            for class_num in range(self.num_classes):
                graph_curr = graphs[class_num]
                # if is_cuda:
                #     graph_curr = graph_curr.cuda()

                to_keep = self.sparsify
                out_graph_curr = self.graph_layers[class_num](input, graphs[class_num], to_keep = to_keep)
                # x_class_curr = self.last_graphs[class_num](out_graph_curr)
                # print x_class_curr.size()
                # print out_graph_curr.size()
                # x_curr.append(x_class_curr)
                out_graph.append(out_graph_curr)

            out_graph = torch.cat(out_graph, 1)
            x_curr = self.last_graph(out_graph)
            # torch.cat(x_curr, 1)
            # 
            # print x_curr.size(), out_graph.size()
            # raw_input()
            x_all.append(x_curr)
            out_graph_all.append(out_graph)
            pmf_all += [self.make_pmf(x_curr).unsqueeze(0)]
            
                
            
        if strip:
            pmf_all = pmf_all[0].squeeze()
            x_all = x_all[0]
            
        if hasattr(self,'feat_ret') and self.feat_ret:
            pmf_all = [pmf_all, [None,out_graph_all,input_sizes_all]]
        
        if ret_bg:
            return x_all, pmf_all, None
        else:
            return x_all, pmf_all
        

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        
        pmf,_ = torch.sort(x, dim=0, descending=True)
        pmf = pmf[:k,:]
        
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        return pmf

    def get_similarity(self,input,idx_graph_layer = 0,sparsify = False, nosum = False):

        # if sparsify is None:
        #     sparsify = self.sparsify

        is_cuda = next(self.parameters()).is_cuda

        input_sizes = [input_curr.size(0) for input_curr in input]
        input = torch.cat(input,0)


        if is_cuda:
            input = input.cuda()
        
        assert idx_graph_layer<len(self.graph_layers)
        
        if hasattr(self, 'layer_bef') and self.layer_bef is not None:
            input = self.layer_bef(input)

        feature_out = self.linear_layer(input)
        
        if sparsify:
            to_keep = self.sparsify[idx_graph_layer]                
        else:
            to_keep = None

        sim_mat = self.graph_layers[idx_graph_layer].get_affinity(feature_out, to_keep = to_keep,nosum = nosum)

        return sim_mat
    
    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].graph_layer.weight.grad
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()

    def out_f(self, input):
        # is_cuda = next(self.parameters()).is_cuda
        # if is_cuda:
        #     input = input.cuda()
        
        identity = False
        method = None
        feature_out = self.linear_layer(input)
        to_keep = self.sparsify
        out_graph = self.graph_layer(input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
        
        if self.graph_sum:
            [out_graph, graph_sum] = out_graph       
        
        return out_graph

    def out_f_f(self, input):
        # is_cuda = next(self.parameters()).is_cuda
        # if is_cuda:
        #     input = input.cuda()
        
        identity = False
        method = None
        feature_out = self.linear_layer(input)
        # to_keep = self.sparsify
        # out_graph = self.graph_layer(input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
        
        # if self.graph_sum:
        #     [out_graph, graph_sum] = out_graph       
        
        return feature_out

class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 sparsify = False,
                 aft_nonlin = 'RL',
                 sigmoid = False,
                 feat_ret = False
                 ):
        # print 'network graph_size', graph_size
        self.model = Graph_Multi_Video(n_classes, deno, in_out, sparsify, aft_nonlin,sigmoid, feat_ret)
        print self.model
        # time.sleep(4)
        # raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        modules = []
        
        modules+=[self.model.graph_layers, self.model.last_graph]

        for i,module in enumerate(modules):
            print i, lr[i]
            lr_list+= [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr[i]}]


        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 20, deno = 8, graph_size = 2)
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