from helpers import util, visualize
import random

import torch.utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import models
import matplotlib.pyplot as plt
import time
import os
import itertools
import glob
import sklearn.metrics
import analysis.evaluate_thumos as et


class Exp_Lr_Scheduler:
    def __init__(self, optimizer,step_curr, init_lr, decay_rate, decay_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.step_curr = step_curr
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        

    def step(self):
        self.step_curr += 1
        # print 'STEPPING',len(self.optimizer.param_groups)
        for idx_param_group,param_group in enumerate(self.optimizer.param_groups): 
            # print 'outside',idx_param_group,self.init_lr[idx_param_group],param_group['lr']
            if self.init_lr[idx_param_group]!=0:
                new_lr = self.init_lr[idx_param_group] * self.decay_rate **(self.step_curr/float(self.decay_steps))
                new_lr = max(new_lr ,self.min_lr)
                # print idx_param_group,param_group['lr'], new_lr
                param_group['lr'] = new_lr
            # print param_group['lr']

def test_model_core(model, test_dataloader, criterion, log_arr):
        model.eval()

        preds = []
        labels_all = []

        for num_iter_test,batch in enumerate(test_dataloader):
            samples = batch['features']
            labels = batch['label'].cuda()

            preds_mini = []
            for sample in samples:
                out,pmf = model.forward(sample.cuda())
                preds.append(pmf.unsqueeze(0))
                preds_mini.append(pmf.unsqueeze(0))

            labels_all.append(labels)

            loss = criterion(labels, torch.cat(preds_mini,0))
            loss_iter = loss.data[0]
            
            str_display = 'val iter: %d, val loss: %.4f' %(num_iter_test,loss_iter)
            log_arr.append(str_display)
            print str_display
            
        preds = torch.cat(preds,0)        
        labels_all = torch.cat(labels_all,0)
            
        loss = criterion(labels_all, preds)
        loss_iter = loss.data[0]

        str_display = 'val total loss: %.4f' %(loss_iter)
        log_arr.append(str_display)
        print str_display
        

        preds = torch.nn.functional.softmax(preds).data.cpu().numpy()
        labels_all = labels_all.data.cpu().numpy()
        labels_all[labels_all>0]=1
        assert len(np.unique(labels_all)==2)
        # print labels_all.shape, np.min(labels_all), np.max(labels_all)
        # print preds.shape, np.min(preds), np.max(preds)

        accuracy = sklearn.metrics.average_precision_score(labels_all, preds)
        
        # print accuracy.shape
        # print accuracy
        
        str_display = 'val accuracy: %.4f' %(accuracy)
        log_arr.append(str_display)
        print str_display
        
        model.train(True)

        return accuracy, loss_iter

def test_model_overlap(model, test_dataloader, criterion, log_arr, overlap_thresh=0.1, bin_trim = None, first_thresh = 0, second_thresh = 0.05):

    model.eval()

    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    
    outs = []
    min_all = None
    max_all = None

    det_events_class = []
    det_time_intervals = []
    det_conf = []
    det_vid_names = []
    idx_test = 0

    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label'].cuda()

        preds_mini = []
        for sample in samples:

            out, pmf = model.forward(sample.cuda())


            # print out.size(),torch.min(out), torch.max(out)
            # print pmf.size(),torch.min(pmf), torch.max(pmf)

            # raw_input()

            if bin_trim is not None:
                # print out.size()
                out = out[:,np.where(bin_trim)[0]]
                pmf = pmf[np.where(bin_trim)[0]]


                # print out.size()
                # raw_input()

            # out = torch.nn.functional.softmax(out,dim = 1)

            pmf = pmf.data.cpu().numpy()
            out = out.data.cpu().numpy()

            bin_not_keep = pmf<first_thresh
            # print pmf, bin_not_keep
            out[:,bin_not_keep]= second_thresh-1
            for r in range(out.shape[0]):
                out[r,~bin_not_keep]=util.softmax(out[r,~bin_not_keep])

            # print out.shape
            # print out[0]

            # print out
            # raw_input()
            # print np.sum(out,1)
            # raw_input()
            # print out.shape

            start_seq = np.array(range(0,out.shape[0]))*16./25.
            end_seq = np.array(range(1,out.shape[0]+1))*16./25.
            time_intervals = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
            # print time_intervals.shape

            det_conf.append(out)
            det_vid_names.extend([det_vid_names_ac[idx_test]]*out.shape[0])
            det_time_intervals.append(time_intervals)
            idx_test +=1

    det_conf = np.concatenate(det_conf,axis =0)
    det_time_intervals = np.concatenate(det_time_intervals,axis = 0)
    class_keep = np.argmax(det_conf , axis = 1)
    # det_conf = np.max(det_conf,axis = 1)
    
    aps = et.test_overlap(det_vid_names, det_conf, det_time_intervals, second_thresh,log_arr = log_arr)
    return aps
    
        

def test_model(out_dir_train,
                model_num, 
                test_data, 
                batch_size_val = None,
                criterion = nn.CrossEntropyLoss(),
                gpu_id = 0,
                num_workers = 0,
                post_pend = '',
                trim_preds = None):
    
    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend)
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_file = os.path.join(out_dir_results,'log.txt')
    out_file = os.path.join(out_dir_results,'aps.npy')
    log_arr=[]

    model = torch.load(model_file)

    if batch_size_val is None:
        batch_size_val = len(test_data)

    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size = batch_size_val,
                        collate_fn = test_data.collate_fn,
                        shuffle = False, 
                        num_workers = num_workers)
    
    torch.cuda.device(gpu_id)
    
    model = model.cuda()
    
    criterion = criterion.cuda()

    if trim_preds is not None:
        bin_trim = np.in1d(np.array(trim_preds[0]),np.array(trim_preds[1]))
        new_trim = np.array(trim_preds[0])[bin_trim]
        old_trim = np.array(trim_preds[1])
        assert np.all(new_trim==old_trim)
    else:
        bin_trim = None
        

    aps = test_model_overlap(model, test_dataloader, criterion, log_arr, bin_trim = bin_trim)
    np.save(out_file, aps)
    util.writeFile(log_file, log_arr)
            

def train_model(out_dir_train,
                train_data,
                test_data,
                batch_size = None,
                batch_size_val = None,
                num_epochs = 100,
                save_after = 20,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = 0.0001,
                dec_after = 100, 
                model_name = 'alexnet',
                criterion = nn.CrossEntropyLoss(),
                gpu_id = 0,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
                network_params = None,
                weight_decay = 0):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    
    log_file_writer = open(log_file,'wb')

    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]
    plot_val_arr =  [[],[]]
    plot_val_acc_arr = [[],[]]
    plot_strs_posts = ['Loss']
    plot_acc_file = os.path.join(out_dir_train,'val_accu.jpg')

    network = models.get(model_name,network_params)

    if model_file is not None:
        network.model = torch.load(model_file)

    model = network.model
    
    if batch_size is None:
        batch_size = len(train_data)

    if batch_size_val is None:
        batch_size_val = len(test_data)

    train_dataloader = torch.utils.data.DataLoader(train_data, 
                        batch_size = batch_size,
                        collate_fn = train_data.collate_fn,
                        shuffle = True, 
                        num_workers = num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size = batch_size_val,
                        collate_fn = test_data.collate_fn,
                        shuffle = False, 
                        num_workers = num_workers)
    
    torch.cuda.device(gpu_id)
    
    model = model.cuda()
    model.train(True)
    
    optimizer = torch.optim.Adam(network.get_lr_list(lr),weight_decay=weight_decay)

    if dec_after is not None:
        print dec_after
        if dec_after[0] is 'step':
            print dec_after
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after[1], gamma=dec_after[2])
        elif dec_after[0] is 'exp':
            print 'EXPING',dec_after
            exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,epoch_start*len(train_dataloader),[lr_curr for lr_curr in lr if lr_curr!=0],dec_after[1],dec_after[2],dec_after[3])
    
    criterion = criterion.cuda()

    for num_epoch in range(epoch_start,num_epochs):

        plot_arr_epoch = []
        for num_iter_train,batch in enumerate(train_dataloader):

            samples = batch['features']
            labels = batch['label'].cuda()

            preds = []
            for sample in samples:
                out,pmf = model.forward(sample.cuda())
                preds.append(pmf.unsqueeze(0))
            preds = torch.cat(preds,0)        

            loss = criterion(labels, preds)
            loss_iter = loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            
            plot_arr_epoch.append(loss_iter)
            str_display = 'lr: %.6f, iter: %d, loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)
            log_arr.append(str_display)
            print str_display

        plot_arr[0].append(num_epoch)
        plot_arr[1].append(np.mean(plot_arr_epoch))

        if num_epoch % plot_after== 0 and num_iter>0:
            
            for string in log_arr:
                log_file_writer.write(string+'\n')
            
            log_arr = []

            if len(plot_val_arr[0])==0:
                visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
            else:
                
                lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Val '] for plot_str_posts in plot_strs_posts]

                # print len(plot_arr),len(plot_val_arr)
                plot_vals = [(arr[0],arr[1]) for arr in [plot_arr]+[plot_val_arr]]
                # print plot_vals
                # print lengend_strs
                visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

                visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])
                


        if (num_epoch+1) % test_after == 0 or num_epoch==0:

            accuracy, loss_iter = test_model_core(model, test_dataloader, criterion, log_arr)
            
            # num_iter = num_epoch*len(train_dataloader)+len(train_dataloader)
            
            plot_val_arr[0].append(num_epoch); plot_val_arr[1].append(loss_iter)
            plot_val_acc_arr[0].append(num_epoch); plot_val_acc_arr[1].append(accuracy)
           

        if (num_epoch+1) % save_after == 0 or num_epoch==0:
            out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
            print 'saving',out_file
            torch.save(model,out_file)

        if dec_after is not None and dec_after[0]=='reduce':
            # exp_lr_scheduler
            if accuracy>=best_val:
                best_val = accuracy
                out_file_best = os.path.join(out_dir_train,'model_bestVal.pt')
                print 'saving',out_file_best
                torch.save(model,out_file_best)            
            exp_lr_scheduler.step(loss_iter)

        elif dec_after is not None and dec_after[0]!='exp':
            exp_lr_scheduler.step()
    
    out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
    print 'saving',out_file
    torch.save(model,out_file)
    
    for string in log_arr:
        log_file_writer.write(string+'\n')
                
    if len(plot_val_arr[0])==0:
        visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
    else:
        
        lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Val '] for plot_str_posts in plot_strs_posts]

        # print len(plot_arr),len(plot_val_arr)
        plot_vals = [(arr[0],arr[1]) for arr in [plot_arr]+[plot_val_arr]]
        # print plot_vals
        # print lengend_strs
        visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

        visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])

    log_file_writer.close()









    