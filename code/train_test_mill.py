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
import debugging_graph as dg

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

def test_model_core(model, test_dataloader, criterion, log_arr, multibranch = 1):
    model.eval()

    preds = []
    
    labels_all = []
    loss_iter_total = 0.
    model_name = model.__class__.__name__.lower()
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label'].cuda()
        
        # print model.__class__.__name__.lower()
        # print 'centerloss' in model.__class__.__name__.lower()
        # raw_input()

        if 'centerloss' in model_name:
            preds_mini,extra = model.forward(samples, labels)
            preds.append(preds_mini)
            labels = [labels]+extra
        else:
            if multibranch>1:
                preds_mini = [[] for i in range(multibranch)]
            else:
                preds_mini = []

            for idx_sample, sample in enumerate(samples):
                if 'perfectg' in model_name:
                    out,pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()])
                else:
                    out,pmf = model.forward(sample.cuda())
            

                if multibranch>1:
                    preds.append(torch.nn.functional.softmax(pmf[0].unsqueeze(0)).data.cpu().numpy())
                        
                    for idx in range(len(pmf)):
                        # print idx, pmf[idx].size(), len(preds_mini[idx])
                        preds_mini[idx].append(pmf[idx].unsqueeze(0))
                else:
                    preds.append(torch.nn.functional.softmax(pmf.unsqueeze(0)).data.cpu().numpy())
                    preds_mini.append(pmf.unsqueeze(0))

            if multibranch>1:
                # print 'hello;',len(preds_mini[0]), len(preds_mini[1])
                preds_mini = [torch.cat(preds_curr,0) for preds_curr in preds_mini]
                # print preds_mini[0].size(), preds_mini[1].size()

            else:
                preds_mini = torch.cat(preds_mini,0)
        
        
        loss = criterion(labels, preds_mini)
        labels_all.append(labels.data.cpu().numpy())
        loss_iter = loss.data[0]
        loss_iter_total+=loss_iter    
        str_display = 'val iter: %d, val loss: %.4f' %(num_iter_test,loss_iter)
        log_arr.append(str_display)
        print str_display
        
    preds = np.concatenate(preds,axis = 0)
    labels_all = np.concatenate(labels_all,axis = 0)
    
    # if 'centerloss' not in criterion.__class__.__name__.lower():    
    #     loss = criterion(labels_all, preds)
    #     loss_iter = loss.data[0]

    #     str_display = 'val total loss: %.4f' %(loss_iter)
    #     log_arr.append(str_display)
    #     print str_display
    

    # labels_all = labels_all.data.cpu().numpy()
    loss_iter = loss_iter_total/len(test_dataloader)
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
    
    del preds
    torch.cuda.empty_cache()

    model.train(True)

    return accuracy, loss_iter

def test_model_overlap_old(model, test_dataloader, criterion, log_arr, overlap_thresh=0.1, bin_trim = None, first_thresh = 0, second_thresh = 0.5):

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

    threshes_all = []
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

            out = torch.nn.functional.softmax(out,dim = 1)

            pmf = pmf.data.cpu().numpy()
            out = out.data.cpu().numpy()

            bin_not_keep = pmf<first_thresh
            
            # for r in range(out.shape[0]):
            #     out[r,~bin_not_keep]=util.softmax(out[r,~bin_not_keep])

            max_vals = np.max(out, axis = 0)
            min_vals = np.min(out, axis = 0)
            threshes = max_vals - (max_vals - min_vals)*second_thresh
            
            out[:,bin_not_keep]= np.min(threshes)-1
            for idx_class in range(threshes.shape[0]):
                out[out[:,idx_class]<threshes[idx_class],idx_class] = threshes[idx_class]-1
            threshes = np.tile(threshes[np.newaxis,:],[out.shape[0],1])
            threshes_all.append(threshes)


            

            start_seq = np.array(range(0,out.shape[0]))*16./25.
            end_seq = np.array(range(1,out.shape[0]+1))*16./25.
            time_intervals = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
            # print time_intervals.shape

            det_conf.append(out)
            det_vid_names.extend([det_vid_names_ac[idx_test]]*out.shape[0])
            det_time_intervals.append(time_intervals)
            idx_test +=1

    threshes_all = np.concatenate(threshes_all,0)
    det_conf = np.concatenate(det_conf,axis =0)
    det_time_intervals = np.concatenate(det_time_intervals,axis = 0)
    class_keep = np.argmax(det_conf , axis = 1)

    # np.savez('../scratch/debug_det.npz', det_vid_names = det_vid_names, det_conf = det_conf, det_time_intervals = det_time_intervals)
    # raw_input()

    aps = et.test_overlap(det_vid_names, det_conf, det_time_intervals, threshes_all,log_arr = log_arr)
    
    return aps

def merge_detections(bin_keep, det_conf, det_time_intervals):
    bin_keep = bin_keep.astype(int)
    bin_keep_rot = np.roll(bin_keep, 1)
    bin_keep_rot[0] = 0
    diff = bin_keep - bin_keep_rot
    # diff[-3]=1
    idx_start_all = list(np.where(diff==1)[0])
    idx_end_all = list(np.where(diff==-1)[0])
    if len(idx_start_all)>len(idx_end_all):
        assert len(idx_start_all)-1==len(idx_end_all)
        idx_end_all.append(bin_keep.shape[0])
    
    assert len(idx_start_all)==len(idx_end_all)
    num_det = len(idx_start_all)
    
    det_conf_new = np.zeros((num_det,))
    det_time_intervals_new = np.zeros((num_det,2))

    for idx_curr in range(num_det):
        idx_start = idx_start_all[idx_curr]
        idx_end = idx_end_all[idx_curr]

        det_conf_rel = det_conf[idx_start:idx_end]
        det_conf_new[idx_curr]=np.max(det_conf_rel)

        # print det_time_intervals.shape, idx_start
        det_time_intervals_new[idx_curr,0]=det_time_intervals[idx_start,0]
        # print idx_end, det_time_intervals.shape, idx_curr, num_det
        det_time_intervals_new[idx_curr,1]=det_time_intervals[idx_end,0] if idx_end<det_time_intervals.shape[0] else det_time_intervals[idx_end-1,1]

        # print bin_keep[idx_start:idx_end]
        # print diff[idx_start:idx_end]
        assert np.all(bin_keep[idx_start:idx_end]==1)

    # print det_conf.shape
    # print det_time_intervals.shape
    # print det_conf_new.shape
    # print det_time_intervals_new.shape

    # raw_input()
    return det_conf_new, det_time_intervals_new

def visualize_dets(model, test_dataloader, dir_viz, first_thresh , second_thresh, bin_trim = None,  det_class = -1, branch_to_test =-1):

    model.eval()
    model_name = model.__class__.__name__.lower()

    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    
    outs = []
    min_all = None
    max_all = None

    det_events_class = []
    det_time_intervals_all = []
    det_conf_all = []
    det_vid_names = []
    out_shapes = []
    idx_test = 0

    threshes_all = []
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label']

        preds_mini = []
        for idx_sample, sample in enumerate(samples):
            # print idx_test
            if branch_to_test>-1:
                out, pmf, bg = model.forward(sample.cuda(), ret_bg = True, branch_to_test = branch_to_test)
            elif 'perfectg' in model_name:
                out, pmf, bg = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()], ret_bg = True)
                # out, pmf, bg = model.forward(sample.cuda(), ret_bg = True)
            else:
                out, pmf, bg = model.forward(sample.cuda(), ret_bg = True)
            # out = out-bg
            if bin_trim is not None:
                out = out[:,np.where(bin_trim)[0]]
                pmf = pmf[np.where(bin_trim)[0]]

            # out = torch.nn.functional.softmax(out,dim = 1)

            start_seq = np.array(range(0,out.shape[0]))*16./25.
            end_seq = np.array(range(1,out.shape[0]+1))*16./25.
            det_time_intervals_meta = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
            

            pmf = pmf.data.cpu().numpy()
            out = out.data.cpu().numpy()

            if det_class==-1:
                class_idx = np.where(labels[idx_sample].numpy())[0][0]
                class_idx_gt = class_idx
            elif det_class ==-2:
                bg = bg.data.cpu().numpy()
                bg = bg[:,:1]
                out = np.concatenate([out,bg],axis = 1)
                class_idx = out.shape[1] - 1
                class_idx_gt = np.where(labels[idx_sample].numpy())[0][0]
            else:
                class_idx = det_class
                class_idx_gt = class_idx

            if det_class>=-1:
                bin_not_keep = pmf<first_thresh
                # print pmf
                # print bin_not_keep
                # print class_idx
                # raw_input()
                # for class_idx in range(pmf.size):
                if bin_not_keep[class_idx]:
                    idx_test +=1
                    print 'PROBLEM'
                    continue
            
            det_conf = out[:,class_idx]
            if second_thresh<0:
                thresh = 0
            else:
                thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*second_thresh
            bin_second_thresh = det_conf>thresh
            
            # det_conf, det_time_intervals = merge_detections(bin_second_thresh, det_conf, det_time_intervals_meta)
            det_time_intervals = det_time_intervals_meta

            det_vid_names.extend([det_vid_names_ac[idx_test]]*det_conf.shape[0])
            det_events_class.extend([class_idx_gt]*det_conf.shape[0])
            out_shapes.extend([out.shape[0]]*det_conf.shape[0])
            
            det_conf_all.append(det_conf)
            det_time_intervals_all.append(det_time_intervals)

            idx_test +=1
            

    det_conf_all = np.concatenate(det_conf_all,axis =0)
    det_time_intervals_all = np.concatenate(det_time_intervals_all,axis = 0)
    det_events_class_all = np.array(det_events_class)
    out_shapes = np.array(out_shapes)

    et.viz_overlap(dir_viz, det_vid_names, det_conf_all, det_time_intervals_all, det_events_class_all,out_shapes)

    # np.savez('../scratch/debug_det_graph.npz', det_vid_names = det_vid_names, det_conf = det_conf, det_time_intervals = det_time_intervals, det_events_class = det_events_class)




def test_model_overlap(model, test_dataloader, criterion, log_arr,first_thresh , second_thresh , bin_trim = None , multibranch =1, branch_to_test = -1):

    # print 'FIRST THRESH', first_thresh
    # print 'SECOND THRESH', second_thresh
    # raw_input()

    model.eval()
    model_name = model.__class__.__name__.lower()

    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    
    outs = []
    min_all = None
    max_all = None

    det_events_class = []
    det_time_intervals_all = []
    det_conf_all = []
    det_vid_names = []
    idx_test = 0

    threshes_all = []
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label'].cuda()

        preds_mini = []
        for idx_sample, sample in enumerate(samples):

            if 'centerloss' in model_name:
                out, pmf = model.forward_single_test(sample.cuda())
            else:    
                
                if multibranch>1:
                    out, pmf = model.forward(sample.cuda(), branch_to_test = branch_to_test)
                else:
                    if 'perfectg' in model_name:
                        out,pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()])
                    else:    
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
            if second_thresh>=0:
                out = torch.nn.functional.softmax(out,dim = 1)

            start_seq = np.array(range(0,out.shape[0]))*16./25.
            end_seq = np.array(range(1,out.shape[0]+1))*16./25.
            det_time_intervals_meta = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
            

            pmf = pmf.data.cpu().numpy()
            out = out.data.cpu().numpy()


            bin_not_keep = pmf<first_thresh
            # print np.min(pmf), np.max(pmf)
            for class_idx in range(pmf.size):
                if bin_not_keep[class_idx]:
                    # print 'PROBLEM'
                    continue

                det_conf = out[:,class_idx]
                if second_thresh<0:
                    thresh = 0
                else:
                    thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*second_thresh
                bin_second_thresh = det_conf>thresh
                # print thresh, np.sum(bin_second_thresh)


                det_conf, det_time_intervals = merge_detections(bin_second_thresh, det_conf, det_time_intervals_meta)
                # det_time_intervals = det_time_intervals_meta
                
                det_vid_names.extend([det_vid_names_ac[idx_test]]*det_conf.shape[0])
                det_events_class.extend([class_idx]*det_conf.shape[0])
                det_conf_all.append(det_conf)
                det_time_intervals_all.append(det_time_intervals)

            idx_test +=1

    # threshes_all = np.concatenate(threshes_all,0)
    det_conf = np.concatenate(det_conf_all,axis =0)
    det_time_intervals = np.concatenate(det_time_intervals_all,axis = 0)
    det_events_class = np.array(det_events_class)
    # class_keep = np.argmax(det_conf , axis = 1)

    # np.savez('../scratch/debug_det_graph.npz', det_vid_names = det_vid_names, det_conf = det_conf, det_time_intervals = det_time_intervals, det_events_class = det_events_class)
    # raw_input()

    aps = et.test_overlap(det_vid_names, det_conf, det_time_intervals,det_events_class,log_arr = log_arr)
    
    return aps
        

def test_model(out_dir_train,
                model_num, 
                test_data, 
                batch_size_val = None,
                criterion = nn.CrossEntropyLoss(),
                gpu_id = 0,
                num_workers = 0,
                post_pend = '',
                trim_preds = None,
                first_thresh = 0,
                second_thresh = 0.5, 
                visualize = False,
                det_class = -1, 
                multibranch = 1,
                branch_to_test = -1):
    
    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend+'_'+str(first_thresh)+'_'+str(second_thresh))
    
    if branch_to_test>-1:
        out_dir_results = out_dir_results +'_'+str(branch_to_test)

    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    if branch_to_test>-1:
        append_name = '_'+str(branch_to_test)
    else:
        append_name = ''    

    log_file = os.path.join(out_dir_results,'log'+append_name+'.txt')
    out_file = os.path.join(out_dir_results,'aps'+append_name+'.npy')
    log_arr=[]

    model = torch.load(model_file)
    if multibranch==1 and branch_to_test>-1:
        model.focus = branch_to_test
    

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
        
    if not visualize:
        aps = test_model_overlap(model, test_dataloader, criterion, log_arr ,first_thresh = first_thresh, second_thresh = second_thresh, bin_trim = bin_trim, multibranch = multibranch, branch_to_test = branch_to_test)
        np.save(out_file, aps)
        util.writeFile(log_file, log_arr)
    else:
        dir_viz = os.path.join(out_dir_results, '_'.join([str(val) for val in ['viz',det_class, first_thresh, second_thresh]]))
        util.mkdir(dir_viz)
        if multibranch>1:
            branch_to_test_pass = branch_to_test
        else:
            branch_to_test_pass = -1
        visualize_dets(model, test_dataloader,  dir_viz,first_thresh = first_thresh, second_thresh = second_thresh,bin_trim = bin_trim,det_class = det_class, branch_to_test = branch_to_test_pass)


def visualize_sim_mat(out_dir_train,
                model_num, 
                test_data, 
                batch_size_val = None,
                gpu_id = 0,
                num_workers = 0,
                post_pend = '', first_thresh = 0, second_thresh = 0.5):
    
    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend+'_'+str(first_thresh)+'_'+str(second_thresh))
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    

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
    
    dir_viz = os.path.join(out_dir_results, '_'.join([str(val) for val in ['viz_sim_mat']]))
    util.mkdir(dir_viz)


    visualize_sim_mat_inner(model, test_dataloader,  dir_viz)
     
            
def visualize_sim_mat_inner(model, test_dataloader, dir_viz):

    model.eval()
    model_name = model.__class__.__name__.lower()
    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    
    outs = []
    min_all = None
    max_all = None

    det_events_class = []
    det_time_intervals_all = []
    det_conf_all = []
    det_vid_names = []
    out_shapes = []
    idx_test = 0

    threshes_all = []
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label']

        preds_mini = []
        for idx_sample, sample in enumerate(samples):
            # print idx_test
            vid_name = det_vid_names_ac[idx_test]
            out_shape_curr = sample.size(0)

            class_idx = np.where(labels[idx_sample].numpy())[0][0]

            if 'perfectg' in model_name:
                sim_mat = model.get_similarity(batch['gt_vec'][idx_sample].cuda())
            else:
                sim_mat = model.get_similarity(sample.cuda())
            # print sim_mat.size()
            sim_mat = sim_mat.data.cpu().numpy()
            dg.save_sim_viz(vid_name, out_shape_curr, sim_mat, class_idx, dir_viz)

            idx_test +=1
    
    dg.make_htmls(dir_viz)        

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
                weight_decay = 0, 
                multibranch = 1):

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
    model_str = str(model)
    log_file_writer.write(model_str+'\n')
    print model_str
    # out_file = os.path.join(out_dir_train,'model_-1.pt')
    # print 'saving',out_file
    # torch.save(model,out_file)    
    # return

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

            if 'centerloss' in model_name:
                preds,extra = model.forward(samples, labels)
                labels = [labels]+extra

            else:
                preds = []
                if multibranch>1:
                    preds = [[] for i in range(multibranch)]
                for idx_sample, sample in enumerate(samples):
                    # print labels[idx_sample]
                    if 'alt_train' in model_name:
                        out,pmf = model.forward(sample.cuda(), epoch_num=num_epoch)
                    elif 'perfectG' in model_name:
                        out,pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()])
                    else:    
                        out,pmf = model.forward(sample.cuda())

                    if multibranch>1:
                        for idx in range(len(pmf)):
                            preds[idx].append(pmf[idx].unsqueeze(0))
                    else:
                        preds.append(pmf.unsqueeze(0))
                
                if multibranch>1:
                    preds = [torch.cat(preds_curr,0) for preds_curr in preds]        
                else:
                    preds = torch.cat(preds,0)        

            loss = criterion(labels, preds)
            loss_iter = loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.printGraphGrad()
            # grad_rel = model.graph_layers[0].graph_layer.weight.grad
            # print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()

            # print criterion.__class__.__name__.lower()
            # if 'centerloss' in criterion.__class__.__name__.lower():
            #     criterion.backward()
            
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

            accuracy, loss_iter = test_model_core(model, test_dataloader, criterion, log_arr, multibranch  = multibranch)
            
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









    