import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Model
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
import time
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import itertools


def MILL(element_logits, seq_len, batch_size, labels, device):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over, 
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    k = np.ceil(seq_len/8).astype('int32')
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CASL(x, element_logits, seq_len, n_similar, labels, device):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    
    print ('x', x.size())
    print ('element_logits', element_logits.size())
    print ('seq_len', seq_len)
    print ('n_similar', n_similar)
    print ('labels', labels.size())
    print ('device',device)
    print (labels)



    sim_loss = 0.
    n_tmp = 0.
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]], dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)
        print (atn1.size(), atn2.size())

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        print ('n1',n1.size())
        print ('n2',n2.size())
        print ('Hf1',Hf1.size())
        print ('Hf2',Hf2.size())
        print ('Lf1',Lf1.size())
        print ('Lf2',Lf2.size())

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))

        print ('d1',d1.size())
        print ('d2',d2.size())
        print ('d3',d3.size())

        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        n_tmp = n_tmp + torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))

        input()

    sim_loss = sim_loss / n_tmp
    return sim_loss

def MyLoss(x, element_logits, seq_len, n_similar, labels, device):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    
    # print ('x', x.size())
    # print ('element_logits', element_logits.size())
    # print ('seq_len', seq_len)
    # print ('n_similar', n_similar)
    # print ('labels', labels.size())
    # print ('device',device)
    # print (labels)
    

    batch_size = x.size(0)
    sim_loss = 0.
    n_tmp = 0.
    for i in range(0, batch_size):
        for j in range(i+1, batch_size):
            # print (batch_size, i, j)
            joint_label = Variable(labels[i,:])+Variable(labels[j,:])
            same_label = torch.max(torch.sum(joint_label.unsqueeze(0),dim = 0))>1
            if same_label:
                # print ('continuing')
                continue

            ssd = [i,j]
            labels_curr = [labels[idx,:] for idx in ssd]
            alpha = [element_logits[idx][:seq_len[idx]] for idx in ssd]
            x_curr = [x[idx][:seq_len[idx]] for idx in ssd]
            n_tmp +=1
            sim_loss += single_double_loss(alpha, labels_curr, x_curr)

            # atn1 = F.softmax(element_logits[i][:seq_len[i]], dim=0)
            # atn2 = F.softmax(element_logits[j][:seq_len[j]], dim=0)
            
            # n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
            # n2 = torch.FloatTensor([np.maximum(seq_len[j]-1, 1)]).to(device)
            # Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)
            # Hf2 = torch.mm(torch.transpose(x[j][:seq_len[j]], 1, 0), atn2)
            # Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
            # Lf2 = torch.mm(torch.transpose(x[j][:seq_len[j]], 1, 0), (1 - atn2)/n2)

            # d1 = 1 - (torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0)))
            # d2 = 1 - (torch.sum(Hf1*Lf1, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0)))
            # d3 = 1 - (torch.sum(Hf2*Lf2, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0)))
            # print ('d1',d1[joint_label>0])
            # print ('d2',d2[joint_label>0])
            # print ('d3',d3[joint_label>0])
            # print ('d1-d2',(d1-d2)[joint_label>0])
            # print ('d1-d3',(d1-d3)[joint_label>0])

            # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*joint_label)
            # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*joint_label)
           
            # torch.sum(joint_label)

            # input()

    sim_loss = sim_loss / max(n_tmp,1)
    return sim_loss


def MyLoss_triple(x, element_logits, seq_len, n_similar, labels, device):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    batch_size = x.size(0)
    sim_loss = 0.
    n_tmp = 0.

    for triple in itertools.combinations(range(batch_size),3):
        triple = list(triple)
        # print (triple)
        # print (labels[triple,:])
        joint_label = torch.sum(labels[triple,:],dim = 0)
        # print (joint_label)
        # input()

        if torch.max(joint_label)!=2 or torch.sum(joint_label==1)==0:
            # print ('zero continue')
            continue

        for same_idx in itertools.combinations(triple,2):
            diff = [idx for idx in triple if idx not in same_idx]
            assert len(diff)==1
            
            ssd = list(same_idx)+ diff
            
            # check same is same
            same_sum = torch.sum(labels[same_idx,:], dim = 0)
            if torch.max(same_sum)!=2 or torch.sum(same_sum==1)!=0:
                continue

            #check diff is diff
            if torch.max(joint_label[labels[ssd[-1],:]>0])!=1:
                print ('second continue')
                continue
            
            #check seq len is atleast 2
            if np.sum(seq_len[ssd]<2)>0:
                print ('third continue')
                continue

            labels_curr = [labels[idx,:] for idx in ssd]
            alpha = [element_logits[idx][:seq_len[idx]] for idx in ssd]
            x_curr = [x[idx][:seq_len[idx]] for idx in ssd]
            n_tmp +=1

            sim_loss += single_triplet_loss(alpha, labels_curr, x_curr)
        
        # input()
    # print (n_tmp, sim_loss)
    sim_loss = sim_loss/max(n_tmp,1)
    # print (sim_loss)
    # input()
    return sim_loss

def cos_sim(vecs):

    vecs = [vec_curr/torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in vecs]
    d1 = 1 - torch.mm(torch.transpose(vecs[0],0,1),vecs[1])
    return d1

def single_double_loss(alpha, labels, x, delta = 0.5, margin = 0.5):  

    assert len(alpha)==len(labels)==len(x)==2

    atns = [F.softmax(alpha_curr, dim = 0) for alpha_curr in alpha]
    Hfs = [torch.mm(torch.transpose(x_curr, 1, 0), atns[idx_x_curr])[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    Lfs = [torch.mm(torch.transpose(x_curr, 1, 0), 1-atns[idx_x_curr])[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    
    h1h2 = cos_sim(Hfs)
    
    h1l1 = cos_sim([Hfs[0], Lfs[0]])
    h2l2 = cos_sim([Hfs[1], Lfs[1]])
    
    first_terms = [h1h2, h1h2]
    second_terms = [h1l1, h2l2]
    assert len(first_terms)==len(second_terms)

    loss = 0
    for idx_term, first_term in enumerate(first_terms):
        second_term = second_terms[idx_term]
        relud = torch.nn.functional.relu(first_term - second_term + margin)
        loss+= 1./len(first_terms)*torch.mean(relud)
    
    return loss
    

def single_triplet_loss(alpha, labels, x, delta = 0.5, margin = 0.5):

    atns = [F.softmax(alpha_curr, dim = 0) for alpha_curr in alpha]
    Hfs = [torch.mm(torch.transpose(x_curr, 1, 0), atns[idx_x_curr])[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    Lfs = [torch.mm(torch.transpose(x_curr, 1, 0), 1-atns[idx_x_curr])[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    
    h1h1 = cos_sim(Hfs[:2])
    h1h2a = cos_sim([Hfs[0], Hfs[2]])
    h1h2b = cos_sim([Hfs[1], Hfs[2]])

    h1l1a = cos_sim([Hfs[0], Lfs[0]])
    h1l1b = cos_sim([Hfs[1], Lfs[1]])
    h2l2 =  cos_sim([Hfs[2], Lfs[2]])

    # lab12a = (labels[0]+labels[2])>0
    # lab12b = (labels[1]+labels[2])>0
    # lab11 = (labels[0]+labels[1])>0

    first_terms = [h1h2a, h1h2a, h1h2b, h1h2b, h1h1, h1h1]
    second_terms = [h1l1a, h2l2, h1l1b, h2l2, h1h2a, h1h2b]
    assert len(first_terms)==len(second_terms)

    loss = 0
    for idx_term, first_term in enumerate(first_terms):
        second_term = second_terms[idx_term]
        # print (first_term.size(), second_term.size())
        relud = torch.nn.functional.relu(first_term - second_term + margin)
        # print (relud)
        loss+= 1./len(first_terms)*torch.mean(relud)
        # print (loss, torch.mean(relud))
        # input()
    return loss
    




def train(itr, dataset, args, model, optimizer, logger, device):

    features, labels = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits = model(Variable(features))
        
    milloss = MILL(element_logits, seq_len, args.batch_size, labels, device)
    casloss = MyLoss(final_features, element_logits, seq_len, args.num_similar, labels, device)

    total_loss = args.Lambda * milloss + (1-args.Lambda) * casloss
        
    logger.log_value('milloss', milloss, itr)
    logger.log_value('tripletloss', casloss, itr)
    logger.log_value('total_loss', total_loss, itr)

    print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

