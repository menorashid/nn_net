import os
import torch
import numpy as np
# from torchvision import transforms
from helpers import util, visualize
import torch.nn as nn

class MultiCrossEntropy(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None):
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

class MultiCrossEntropy_noSoftmax(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None):
        super(MultiCrossEntropy_noSoftmax, self).__init__()
        # self.Log = nn.Log(dim = 1)
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

    def forward(self, gt, pred):
        pred = torch.log(pred)

        if self.class_weights is not None:
            assert self.class_weights.size(1)==pred.size(1)
            loss = self.class_weights*-1*gt*pred
        else:
            loss = -1*gt* pred

        loss = torch.sum(loss, dim = 1)
        loss = torch.mean(loss)
        return loss


class Wsddn_Loss(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None):
        super(Wsddn_Loss, self).__init__()
        
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

    def forward(self, gt, pred):
        gt[gt>0]=1.
        gt[gt<=0]=-1.

        in_log_val = torch.clamp(gt*(pred - 0.5)+0.5,1e-10,1)
        loss = -1*torch.log(in_log_val)
        
        if self.class_weights is not None:
            assert self.class_weights.size(1)==pred.size(1)
            loss = self.class_weights*loss

        loss = torch.sum(loss, dim = 1)
        loss = torch.mean(loss)
        if loss.eq(float('-inf')).any() or loss.eq(float('inf')).any():
            print torch.min(pred), torch.max(pred)
            print torch.min(in_log_val), torch.max(in_log_val)
            print torch.log(torch.min(in_log_val)),torch.log(torch.max(in_log_val))

            raw_input()


        return loss


class Wsddn_Loss_WithL1(Wsddn_Loss):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None, window_size = 3):
        num_branches = max(num_branches,1)
        super(Wsddn_Loss_WithL1, self).__init__(class_weights=class_weights, loss_weights = loss_weights, num_branches = num_branches)
        self.loss_weights = loss_weights
        # self.att_weight = loss_weights[-1]
        
    def forward(self, gt, preds, att):
        
        loss_regular = super(Wsddn_Loss_WithL1,self).forward(gt, preds)
        # max_preds = preds[

        max_preds = torch.cat([max_pred_curr.unsqueeze(0) for max_pred_curr,_ in att],0)
        dots = torch.cat([dot_curr.unsqueeze(0) for _,dot_curr in att],0)

        max_preds = 0.5*(max_preds**2)

        loss_spatial = torch.sum(max_preds*dots)
        # print loss_spatial
        loss_spatial = loss_spatial/(max_preds.size(0)*max_preds.size(1))
        # print loss_spatial

        # print max_preds.size()
        
        # print att[0][0]
        # print max_preds[0]

        
        # print dots.size()
        # print att[0][1]
        # print dots[0]


        # for fc, max_idx in att:
        #     print 'in loss'
        #     print fc.size()
        #     print max_idx.size()
        #     print max_idx
        

        


        # l1 = torch.mean(torch.abs(att))
        # l1 = self.att_weight*l1
        # loss_all = l1+loss_regular
        # print loss_spatial
        # print loss_regular
        
        loss_all = self.loss_weights[0]*loss_regular + self.loss_weights[1]*loss_spatial
        
        return loss_all

class MultiCrossEntropyMultiBranch(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2):
        super(MultiCrossEntropyMultiBranch, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim = 1)
        self.num_branches = num_branches

        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

        if loss_weights is None:
            self.loss_weights = [1 for i in range(self.num_branches)]
        else:
            self.loss_weights = loss_weights

    def forward(self, gt, preds):
        loss_all = 0
        assert len(preds) == self.num_branches
        for idx_pred, pred in enumerate(preds):
            pred = self.LogSoftmax(pred)
            # print pred.size()
            if self.class_weights is not None:
                assert self.class_weights.size(1)==pred.size(1)
                loss = self.class_weights*-1*gt*pred
            else:
                loss = -1*gt* pred

            loss = torch.sum(loss, dim = 1)
            loss = torch.mean(loss)
            loss_all += loss*self.loss_weights[idx_pred]
        return loss_all

class MultiCrossEntropyMultiBranchWithL1(MultiCrossEntropyMultiBranch):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5):
        num_branches = max(num_branches,1)
        super(MultiCrossEntropyMultiBranchWithL1, self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        self.att_weight = loss_weights[-1]
        
    def forward(self, gt, preds, att):
        if self.num_branches ==1:
            preds = [preds]

        loss_regular = super(MultiCrossEntropyMultiBranchWithL1,self).forward(gt, preds)
        # print 'min_val',torch.min(torch.abs(att))
        

        l1 = torch.mean(torch.abs(att))
        # print 'l1',l1,'loss_regular',loss_regular
        l1 = self.att_weight*l1
        # print 'l1',l1,'loss_regular',loss_regular
        loss_all = l1+loss_regular
        # raw_input()
        # loss_all = 0
        # assert len(preds) == self.num_branches
        # for idx_pred, pred in enumerate(preds):
        #     pred = self.LogSoftmax(pred)
        #     # print pred.size()
        #     if self.class_weights is not None:
        #         assert self.class_weights.size(1)==pred.size(1)
        #         loss = self.class_weights*-1*gt*pred
        #     else:
        #         loss = -1*gt* pred

        #     loss = torch.sum(loss, dim = 1)
        #     loss = torch.mean(loss)
        #     loss_all += loss*self.loss_weights[idx_pred]
        return loss_all

class MCE_CenterLoss_Combo(nn.Module):
    def __init__(self, n_classes, feat_dim, bg, lambda_param, alpha_param, class_weights = None):
        super(MCE_CenterLoss_Combo, self).__init__()  
        self.lambda_param = lambda_param
        self.alpha_param = alpha_param
        self.mce = MultiCrossEntropy(class_weights)
        self.cl = CenterLoss(n_classes, feat_dim, bg)
        self.optimizer_centloss = torch.optim.SGD(self.cl.parameters(), lr=alpha_param)

    def forward(self, gt, pred):
        [gt, gt_all, features, class_pred] = gt
        loss_mce = self.mce(gt, pred)
        # loss_cl = self.cl(gt_all, features, class_pred)
        loss_total = loss_mce
         # + self.lambda_param*loss_cl
        # self.optimizer_centloss.zero_grad()
        return loss_total

    # def backward(self):
    #     pass
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        # for param in self.cl.parameters():
        #     param.grad.data *= (1./self.lambda_param)
        # self.optimizer_centloss.step()





class CenterLoss(nn.Module):
    def __init__(self, n_classes, feat_dim, bg):
        super(CenterLoss, self).__init__()
        
        self.bg = bg

        if self.bg:
            n_classes+=1

        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.n_classes, self.feat_dim).cuda())
        # [None]*self.n_classes

    # def make_clusters(self,features, gt):
    #     # find features of each gt class (both for multilabel)
    #     for class_num in range(self.n_classes):
    #         if self.centers[class_num] is not None:
    #             continue
    #         rel_features = features[gt[:,class_num]>0,:]
    #         if rel_features.size(0)==0:
    #             continue
    #         rel_features = torch.mean(rel_features,dim =0)
    #         self.centers[class_num] = rel_features

    # def update_clusters(self, features, gt):
    #     for class_num in range(self.n_classes):
            
    #         rel_features = features[gt[:,class_num]>0,:]
    #         if rel_features.size(0)==0:
    #             continue
            
    #         rel_features = torch.mean(rel_features,dim =0)
    #         self.centers[class_num] = rel_features        


    def forward(self, gt, features, class_pred = None):
        
        num_instances = features.size(0)

        is_cuda = features.is_cuda
        
        if self.bg:
            assert class_pred is not None
            assert class_pred.size(1)==self.n_classes
            assert class_pred.size(1) - gt.size(1)==1

            zeros = torch.zeros(gt.size(0),1)
            if is_cuda:
                zeros = zeros.cuda()
            
            gt = torch.cat([gt,zeros],dim = 1)
            bin_bg = torch.argmax(class_pred, dim = 1)==(class_pred.size(1)-1)
            gt[bin_bg,:-1] = 0
            gt[bin_bg,-1] = 1

        # self.make_clusters(features,gt)
        
        loss_total = 0
        for class_num in range(self.n_classes):

            rel_features = features[gt[:,class_num]>0,:]
            if rel_features.size(0)==0:
                continue
            
            center_rel = self.centers[class_num]
            center_rel = center_rel.view(1,center_rel.size(0)).expand(rel_features.size(0),-1)
            distance = torch.sum(torch.sum(torch.pow(rel_features - center_rel,2),1),0)
            loss_total += distance

        loss_total = loss_total/num_instances        
        
        return loss_total    