from train_test_mill import *
import models
from criterions import *
import os
from helpers import util,visualize
from dataset import *
import numpy as np
import torch
from globals import * 

def get_data(dataset, limit, all_classes, just_primary, gt_vec, k_vec):

    if dataset =='ucf':
        dir_files = '../data/ucf101/train_test_files'
        n_classes = 20
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train')
        test_train_file = os.path.join(dir_files, 'test')
        test_file = os.path.join(dir_files, 'test')
            
        files = [train_file, test_train_file, test_file]

        if all_classes:
            n_classes = 101
            post_pends = [pp+val for pp,val in zip(post_pends,['_all','_all',''])]
            classes_all_list = util.readLinesFromFile(os.path.join(dir_files,'classes_all_list.txt'))
            classes_rel_list = util.readLinesFromFile(os.path.join(dir_files,'classes_rel_list.txt'))
            trim_preds = [classes_all_list,classes_rel_list]

        if just_primary:
            post_pends = [pp+val for pp,val in zip(post_pends,['_just_primary','_just_primary','_just_primary'])]
        
        # post_pends = [pp+val for pp,val in zip(post_pends,['_corrected','_corrected','_corrected'])]
        if not all_classes:
            post_pends = [pp+val for pp,val in zip(post_pends,['_ultra_correct','_ultra_correct','_ultra_correct'])]
                

        if gt_vec:
            post_pends = [pp+val for pp,val in zip(post_pends,['_gt_vec','_gt_vec','_gt_vec'])]

        if k_vec is not None:
            # print 
            post_pends = [pp+val for pp,val in zip(post_pends,['_'+k_vec]*3)]


        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        if gt_vec or (k_vec is not None):
            train_data = UCF_dataset_gt_vec(train_file, limit)
            test_train_data = UCF_dataset_gt_vec(test_train_file, limit)
            test_data = UCF_dataset_gt_vec(test_file, None)
        else:
            # all_classes
            train_data = UCF_dataset(train_file, limit)
            test_train_data = UCF_dataset(test_train_file, limit)
            test_data = UCF_dataset(test_file, None)
    elif dataset =='activitynet':
        dir_files = '../data/activitynet/train_test_files'
        n_classes = 100
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train')
        test_train_file = os.path.join(dir_files, 'val')
        test_file = os.path.join(dir_files, 'val')
            
        files = [train_file, test_train_file, test_file]

        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        train_data = UCF_dataset(train_file, limit)
        test_train_data = UCF_dataset(test_train_file, limit)
        test_data = UCF_dataset(test_file, None)

    return train_data, test_train_data, test_data, n_classes, trim_preds



def train_simple_mill_all_classes(model_name,
                                    lr,
                                    dataset,
                                    network_params,
                                    limit,
                                    epoch_stuff=[30,60],
                                    res=False,
                                    class_weights = False,
                                    batch_size = 32,
                                    batch_size_val = 32,
                                    save_after = 1,
                                    model_file = None,
                                    gpu_id = 0,
                                    exp = False,
                                    test_mode = False,
                                    test_after = 1,
                                    all_classes = False,
                                    just_primary = False,
                                    model_nums = None,
                                    retrain = False,
                                    viz_mode = False,
                                    det_class = -1,
                                    second_thresh = 0.5,
                                    first_thresh = 0,
                                    post_pend = '',
                                    viz_sim = False,
                                    test_post_pend = '', 
                                    multibranch = 1,
                                    loss_weights = None,
                                    branch_to_test = 0,
                                    gt_vec = False,
                                    k_vec = None,
                                    attention = False):

    num_epochs = epoch_stuff[1]

    # test_mode = test_mode or viz_mode or viz_sim

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr

    train_data, test_train_data, test_data, n_classes, trim_preds = get_data(dataset, limit, all_classes, just_primary, gt_vec, k_vec)
    
    network_params['n_classes']=n_classes

    train_file = train_data.anno_file
    
    print train_file
    print test_data.anno_file
    print class_weights
    # raw_input()

    if class_weights:
        class_weights_val = util.get_class_weights_au(util.readLinesFromFile(train_file),n_classes)
    else:
        class_weights_val = None

    if attention:
        if multibranch>1:
            criterion = MultiCrossEntropyMultiBranchWithL1(class_weights= class_weights_val, loss_weights = loss_weights, num_branches = multibranch)
            criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
        else:
            criterion = MultiCrossEntropyMultiBranchWithL1(class_weights= class_weights_val, loss_weights = loss_weights, num_branches = 1)
            criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    else:
        if multibranch>1:
            criterion = MultiCrossEntropyMultiBranch(class_weights= class_weights_val, loss_weights = loss_weights, num_branches = multibranch)
            criterion_str = 'MultiCrossEntropyMultiBranch'
        else:
            criterion = MultiCrossEntropy(class_weights= class_weights_val)
            criterion_str = 'MultiCrossEntropy'


    # criterion = MCE_CenterLoss_Combo(n_classes, feat_dim = 2048, bg = True, lambda_param = 0.0, alpha_param = 0.5, class_weights= class_weights_val)
    # criterion_str = 'Multi_Center_Combo'


    init = False
    

    
    out_dir_meta = os.path.join('../experiments',model_name)
    util.mkdir(out_dir_meta)

    out_dir_meta_str = [model_name]
    for k in network_params.keys():
        out_dir_meta_str.append(k)
        if type(network_params[k])==type([]):
            out_dir_meta_str.extend(network_params[k])
        else:
            out_dir_meta_str.append(network_params[k])
    out_dir_meta_str.append(dataset)
    out_dir_meta_str = '_'.join([str(val) for val in out_dir_meta_str])
    
    out_dir_meta = os.path.join(out_dir_meta,out_dir_meta_str)
    # print out_dir_meta
    util.mkdir(out_dir_meta)
    


    strs_append_list = ['all_classes',all_classes,'just_primary',just_primary,'limit',limit,'cw',class_weights, criterion_str, num_epochs]+dec_after+lr
    
    if loss_weights is not None:
        strs_append_list += ['lw']+['%.2f' % val for val in loss_weights]
    
    strs_append_list+=[post_pend] if len(post_pend)>0 else []
    strs_append = '_'.join([str(val) for val in strs_append_list])

    out_dir_train =  os.path.join(out_dir_meta,strs_append)
    final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
    


    if os.path.exists(final_model_file) and not test_mode and not retrain:
        print 'skipping',final_model_file
        return 
    else:
        print 'not skipping', final_model_file


    # network_params = dict(n_classes=n_classes,deno = deno)

    # if 'alt_train' in model_name:
    #     network_params['num_switch'] = num_switch
    # if in_out is not None:
    #     network_params['in_out'] = in_out
    # if graph_size is not None:
    #     network_params['graph_size'] = graph_size
    
    # print network_params
    # raw_input()
        

    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_train_data,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = save_after,
                disp_after = 1,
                plot_after = 1,
                test_after = test_after,
                lr = lr,
                dec_after = dec_after,
                model_name = model_name,
                criterion = criterion,
                gpu_id = gpu_id,
                num_workers = 0,
                model_file = model_file,
                epoch_start = epoch_start,
                network_params = network_params, 
                multibranch = multibranch)

    if not test_mode:
        train_model(**train_params)
    
    if model_nums is None :
        model_nums = [num_epochs-1] 
    # print model_nums
    # model_nums = [0]+[i-1 for i in range(save_after,num_epochs,save_after)]
    # if model_nums[-1]!=(num_epochs-1):
    #     model_nums.append(num_epochs-1)
    # return
    for model_num in model_nums:

        print 'MODEL NUM',model_num

        test_params = dict(out_dir_train = out_dir_train,
                model_num = model_num,
                test_data = test_data,
                batch_size_val = batch_size_val,
                criterion = criterion,
                gpu_id = gpu_id,
                num_workers = 0,
                trim_preds = trim_preds,
                visualize = False,
                det_class = det_class,
                second_thresh = second_thresh,
                first_thresh = first_thresh,
                post_pend=test_post_pend,
                multibranch = multibranch,
                branch_to_test =branch_to_test,
                dataset = dataset)
        test_model(**test_params)
        if viz_mode:
            test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    test_data = test_data,
                    batch_size_val = batch_size_val,
                    criterion = criterion,
                    gpu_id = gpu_id,
                    num_workers = 0,
                    trim_preds = trim_preds,
                    visualize = True,
                    det_class = det_class,
                    second_thresh = second_thresh,
                    first_thresh = first_thresh,
                    post_pend=test_post_pend,
                    multibranch = multibranch,
                    branch_to_test =branch_to_test,
                    dataset = dataset)
            test_model(**test_params)
            test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    test_data = test_data,
                    batch_size_val = batch_size_val,
                    gpu_id = gpu_id,
                    num_workers = 0,
                    second_thresh = second_thresh,
                    first_thresh = first_thresh)
            visualize_sim_mat(**test_params)



def ens_moredepth_concat_sim_experiments():
    model_name = 'graph_multi_video_same_F_ens_dll_moredepth_concat_sim'

    lr = [0.001, 0.001]
    multibranch = 1
    loss_weights = None
    # [1/float(multibranch)]*multibranch
    branch_to_test = -1

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = 500
    save_after = 10
    
    test_mode = False

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,128]
    network_params['feat_dim'] = [2048,64]
    network_params['num_graphs'] = 1
    network_params['graph_size'] = 1
    network_params['num_branches'] = multibranch
    network_params['non_lin'] = 'HT'
    network_params['non_lin_aft'] = 'RL'
    network_params['aft_nonlin']='HT_L2'
    network_params['scaling_method']='n'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias_sym'
    
    first_thresh=0.

    class_weights = True
    test_after = 5
    
    all_classes = False
    
    
    second_thresh = 0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)



def ens_moredepth_experiments():
    model_name = 'graph_multi_video_same_F_ens_dll_moredepth'

    lr = [0.001,0.001, 0.01]
    multibranch = 1
    loss_weights = None
    # [1/float(multibranch)]*multibranch
    branch_to_test = -1

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    
    epoch_stuff = [300,300]
    dataset = 'ucf'
    limit  = None
    save_after = 100
    
    test_mode = False

    model_nums = [99,199,299]
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,256]
    network_params['feat_dim'] = [2048,512]
    network_params['num_graphs'] = 1
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = [0.5]
    # ,None,None]
    # network_params['layer_bef'] = [2048,512]
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_L2'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias'
    
    first_thresh=0.1

    class_weights = True
    test_after = 5
    
    all_classes = False
    
    
    second_thresh = 0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)


def ens_experiments():
    model_name = 'graph_multi_video_same_F_ens_dll'
    # # lr = [0.001]
    lr = [0.001, 0.001,0.001]
    multibranch = 4
    loss_weights = [1/float(6)]*3+[1/2.]
    # 
    branch_to_test = -4

    # model_name = 'graph_multi_video_same_i3dF_ens_sll'
    # lr = [0.001]
    # model_name = 'graph_multi_video_diff_F_ens_sll'
    # lr = [0.001,0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [500,500]
    dataset = 'ucf'
    limit  = 500
    save_after = 50
    
    test_mode = True

    model_nums = range(99,epoch_stuff[1],100)
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,256]
    network_params['feat_dim'] = [2048,512]
    # network_params['layer_bef'] = [2048,2048]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    # network_params['sparsify'] = list(np.arange(0.5,1.0,0.1))[::-1]
    network_params['sparsify'] = [0.75,0.5,0.25,'lin']
    # loss_weights = network_params['sparsify']
    # [0.9,0.8,0.7,0.6,0.5]
    # ,0.75,0.5]
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_l2'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias'
    
    first_thresh=0.1

    class_weights = True
    test_after = 5
    
    all_classes = False
    # just_primary = False
    # gt_vec = False

    
    second_thresh = 0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)


def ens_att_experiments():
    model_name = 'graph_multi_video_attention_soft'
    lr = [0.001, 0.001, 0.001]
    multibranch = 1
    loss_weights = [1]*multibranch + [0.001]
    branch_to_test = -1
    attention = True

    k_vec = None

    gt_vec = False
    just_primary = False
    all_classes = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [300,300]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = False

    model_nums = None
    retrain = True
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,256]
    network_params['feat_dim'] = [2048,512]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['att'] = 256
    # network_params['sparsify'] = [0.5]
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_l2'
    post_pend = 'ABS_bias'
    
    first_thresh=0
    second_thresh = 0.5
    det_class = -1
    
    class_weights = True
    test_after = 5
    
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec,
                        attention = attention)

def testing_exp():
    # model_name = 'graph_multi_video_multi_F_joint_train_gaft'
    # lr = [0.001,0.001]
    # multibranch = 2
    # loss_weights = [0,1]
    # branch_to_test = 1


    # model_name = 'graph_multi_video_i3dF_gaft'
    # lr = [0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0


    # model_name = 'graph_multi_video_same_F'
    # lr = [0.001,0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0
    # for idx_class in [1]:
    # range(18,20):
    k_vec = None

    model_name = 'graph_multi_video_cooc_ofe_olg'
    lr = [0.001]
    loss_weights = None
    multibranch = 1
    branch_to_test = 0
    k_vec = 'k_100'


    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [300,300]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = False

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    # network_params['pretrained'] = 'ucf'
    network_params['in_out'] = [2048,16]
    network_params['feat_dim'] = '100'
    # [2048,32]
    # 
    network_params['post_pend'] = 'negexp'
    # [2048,32]
    network_params['graph_size'] = 2
    # network_params['gk'] = 8
    network_params['method'] = 'affinity_dict'
    # network_params['num_switch'] = [5,5]
    # network_params['focus'] = 0
    network_params['sparsify'] = False
    network_params['non_lin'] = None
    # network_params['normalize'] = [True, True]
    network_params['aft_nonlin']='HT_l2'
    # network_params['attention'] = False

    post_pend = 'ABS_bias'
    
    first_thresh=0

    

    class_weights = True
    test_after = 5
    
    all_classes = False
    # just_primary = False
    # gt_vec = False

    
    
    
    second_thresh =0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)

def super_simple_experiment():
    # model_name = 'just_mill_flexible'
    # model_name = 'graph_perfectG'
    # model_name = 'graph_pretrained_F_random'
    # model_name = 'graph_pretrained_F_ucf_64'
    # model_name = 'graph_pretrained_F_activitynet'
    # model_name = 'graph_multi_video_pretrained_F_ucf_64_zero_self'
    # model_name = 'graph_multi_video_pretrained_F_flexible'
    model_name = 'graph_multi_video_pretrained_F_flexible_alt_train_temp'
    # model_name = 'graph_pretrained_F'
    # model_name = 'graph_sim_direct_mill_cosine'
    # model_name = 'graph_sim_i3d_sim_mat_mill'
    # model_name = 'graph_sim_mill'
    # model_name = 'graph_same_G_multi_cat'
    # model_name = 'graph_2_G_multi_cat'
    # epoch_stuff = [25,25]
    # save_after = 5

    
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    # lr = [0.001,0.001]
    lr = [0.001,0.001]
    epoch_stuff = [500,500]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = True
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    in_out = None

    network_params = {}
    network_params['deno'] = 8
    # network_params['layer_sizes'] = [2048,64]
    # ,2048,64]
    network_params['pretrained'] = 'default'
    network_params['in_out'] = [2048,64,2048,64]
    network_params['graph_size'] = 2
    # network_params['k']
    # graph_size = 1
    network_params['method'] = 'cos'
    network_params['num_switch'] = [5,5]
    network_params['focus'] = 0
    network_params['sparsify'] = True
    network_params['non_lin'] = 'HT'
    network_params['normalize'] = [True, True]
    
    post_pend = 'ABS_EASYLR'
    loss_weights = None
    multibranch = 1
    branch_to_test = 1

    # in_out = [2048,64]
    # post_pend = '_'.join([str(val) for val in in_out])
    # post_pend += '_seeded'
    # graph_size = None
    # post_pend += '_new_model_fix_ht_cos_norm'

    # graph_size = 32
    # post_pend += '_bw_32_bs_'+str(graph_size)
    first_thresh=0


    

    class_weights = True
    test_after = 5
    
    all_classes = False
    just_primary = False
    gt_vec = False

    model_nums = [99]
    
    
    second_thresh =0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test)

def separate_supervision_experiment():
    # model_name = 'just_mill_2_1024'
    # model_name = 'graph_sim_direct_mill_cosine'
    # model_name = 'graph_sim_i3d_sim_mat_mill'
    # model_name = 'graph_sim_mill'
    # model_name = 'graph_same_G_multi_cat'
    # model_name = 'graph_same_G_multi_cat_separate_supervision_unit_norm'
    model_name = 'graph_same_G_sepsup_alt_train_2_layer'
    # epoch_stuff = [25,25]
    # save_after = 5


    lr = [1e-4,1e-4,1e-4,1e-4]
    epoch_stuff = [400,400]
    dataset = 'ucf'
    limit  = 500
    deno = 8
    save_after = 25
    
    loss_weights = None
    multibranch = 1
    num_switch = 5
    branch_to_test = 1
    test_mode = True

    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''
    post_pend = '512_1024'

    class_weights = True
    test_after = 10
    all_classes = False
    just_primary = False
    model_nums = [374]
    
    second_thresh =0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        deno = deno,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend, 
                        loss_weights = loss_weights, 
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        num_switch = num_switch)

def create_comparative_viz(dirs, class_names, dir_strs, out_dir_html):

    for class_name in class_names:
        out_file_html = os.path.join(out_dir_html, class_name+'.html')
        ims_html = []
        captions_html = []

        im_list = glob.glob(os.path.join(dirs[0],class_name, '*.jpg'))
        im_list = [os.path.split(im_curr)[1] for im_curr in im_list]
        for im in im_list:
            row_curr = [util.getRelPath(os.path.join(dir_curr,class_name,im),dir_server) for dir_curr in dirs]
            caption_curr = [dir_str+' '+im[:im.rindex('.')] for dir_str in dir_strs]
            ims_html.append(row_curr)
            captions_html.append(caption_curr)

        visualize.writeHTML(out_file_html, ims_html, captions_html, height = 150, width = 200)

def scripts_comparative():
    # dir_meta= '../experiments/graph_sim_direct_mill_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001'
    # dir_meta = dir_meta.replace(str_replace[0],str_replace[1])

    # dirs = ['results_model_24_0_0.5/viz_sim_mat', 'results_model_24_0_0.5/viz_-1_0_0.5']
    # dirs = [os.path.join(dir_meta, dir_curr) for dir_curr in dirs]

    # dir_meta_new= '../experiments/just_mill_2_1024_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001'
    # dir_meta_new = dir_meta_new.replace(str_replace[0],str_replace[1])
    # dirs_new= ['results_model_99_0_0.5/viz_sim_mat', 'results_model_99_0_0.5/viz_-1_0_0.5']
    # dirs += [os.path.join(dir_meta_new, dir_curr) for dir_curr in dirs_new]


    # dir_meta_new= '../experiments/graph_sim_direct_mill_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001__noRelu'
    # dir_meta_new = dir_meta_new.replace(str_replace[0],str_replace[1])
    # dirs_new= ['results_model_24_0_0.5/viz_sim_mat', 'results_model_24_0_0.5/viz_-1_0_0.5']
    # dirs += [os.path.join(dir_meta_new, dir_curr) for dir_curr in dirs_new]


    # out_dir_html = os.path.join(dir_meta,'comparative_htmls')
    # util.mkdir(out_dir_html)

    
    # dir_strs = ['graph_sim_24','graph_pred_24','mill_2_layer_sim_99', 'mill_2_layer_pred_99','graph_sim_24_nr','graph_pred_24_nr']


    dir_meta = '../experiments/graph_same_G_multi_cat_separate_supervision_unit_norm_ucf/all_classes_False_just_primary_False_deno_8_limit_500_cw_True_MultiCrossEntropyMultiBranch_200_lw_1_1_step_200_0.1_0.0001_ht_cosine_normalizedG'

    dir_meta = dir_meta.replace(str_replace[0],str_replace[1])

    dirs = ['results_model_199_0_0.5_0/viz_-1_0_0.5', 'results_model_199_0_0.5_1/viz_-1_0_0.5','results_model_199_0_0.5/viz_sim_mat']
    dirs = [os.path.join(dir_meta, dir_curr) for dir_curr in dirs]
    out_dir_html = os.path.join(dir_meta,'comparative_htmls')
    util.mkdir(out_dir_html)
    dir_strs = ['pred_0','pred_1','sim']

    create_comparative_viz(dirs, class_names, dir_strs, out_dir_html) 



def exps_for_visualizing_W():
    model_name = 'graph_multi_video_same_i3dF'
    lr = [0.001,0.01]
    
    # model_name = 'just_mill_flexible'
    # lr = [0.001,0.001]
    # epoch_stuff = [100,100]

    multibranch = 1
    loss_weights = None
    branch_to_test = -1
    
    gt_vec = False
    just_primary = False
    all_classes = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = None
    save_after = 50
    
    test_mode = True

    model_nums = [99]
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    # network_params = {}
    # network_params['deno'] = 8
    # network_params['layer_sizes'] = [2048,2]
    

    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,2]
    # network_params['feat_dim'] = [2048,64]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = 0.5

    network_params['non_lin'] = None
    network_params['aft_nonlin']='HT_l2'
    post_pend = 'ABS_bias_wb'
    
    first_thresh=0
    second_thresh = 0.5
    det_class = -1
    
    class_weights = True
    test_after = 5
    
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        )



def ens_experiments_pool():
    model_name = 'graph_multi_video_same_F_ens_pool'
    # # lr = [0.001]
    lr = [0.001, 0.001,0.001,0.001]
    multibranch = 1
    loss_weights = None
    # [1/float(multibranch)]*multibranch
    # 
    branch_to_test = -1

    # model_name = 'graph_multi_video_same_i3dF_ens_sll'
    # lr = [0.001]
    # model_name = 'graph_multi_video_diff_F_ens_sll'
    # lr = [0.001,0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = False

    model_nums = [99]
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,256]
    network_params['feat_dim'] = [2048,512]
    # network_params['layer_bef'] = [2048,1024]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = [0.5,'lin']
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_L2'
    network_params['pool_method']='avg_cat'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias'
    
    first_thresh=0.1

    class_weights = True
    test_after = 5
    
    all_classes = False
    # just_primary = False
    # gt_vec = False

    
    second_thresh = 0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)


def main():
    # print 'hello hello baby'
    # scripts_comparative()
    
    # separate_supervision_experiment()
    # super_simple_experiment()
    # testing_exp()
    ens_experiments()
    # ens_experiments_pool()
    # ens_moredepth_experiments()
    # ens_att_experiments()
    # ens_moredepth_concat_sim_experiments()
    # exps_for_visualizing_W()

if __name__=='__main__':
    main()