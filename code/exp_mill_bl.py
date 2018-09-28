from train_test_mill import *
import models
from criterions import *
import os
from helpers import util,visualize
from dataset import *
import numpy as np
import torch
from globals import * 

def train_simple_mill_all_classes(model_name,
                                    lr,
                                    dataset,
                                    deno,
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
                                    post_pend = '',
                                    viz_sim = False,
                                    test_post_pend = ''):

    out_dir_meta = '../experiments/'+model_name+'_'+dataset
    util.mkdir(out_dir_meta)
    num_epochs = epoch_stuff[1]

    test_mode = test_mode or viz_mode or viz_sim

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr

    if dataset =='ucf' and all_classes:
        dir_files = '../data/ucf101/train_test_files'
        n_classes = 101
        train_file = os.path.join(dir_files, 'train_all.txt')
        test_train_file = os.path.join(dir_files, 'test_all.txt')
        test_file = os.path.join(dir_files, 'test.txt')

        if just_primary:
            old_files = [train_file, test_train_file, test_file]
            new_files = [file_curr[:file_curr.rindex('.')]+'_just_primary.txt' for file_curr in old_files]
            train_file, test_train_file, test_file = new_files

        classes_all_list = util.readLinesFromFile(os.path.join(dir_files,'classes_all_list.txt'))
        classes_rel_list = util.readLinesFromFile(os.path.join(dir_files,'classes_rel_list.txt'))
        train_data = UCF_dataset(train_file, limit)
        test_train_data = UCF_dataset(test_train_file, limit)
        test_data = UCF_dataset(test_file, None)
        trim_preds = [classes_all_list,classes_rel_list]
    else:
        dir_files = '../data/ucf101/train_test_files'
        n_classes = 20
        train_file = os.path.join(dir_files, 'train.txt')
        test_file = os.path.join(dir_files, 'test.txt')
        if just_primary:
            old_files = [train_file, test_file]
            new_files = [file_curr[:file_curr.rindex('.')]+'_just_primary.txt' for file_curr in old_files]
            train_file, test_file = new_files
        train_data = UCF_dataset(train_file, limit)
        test_train_data = UCF_dataset(test_file, limit)
        test_data = UCF_dataset(test_file, None)
        trim_preds = None

    print train_file
    print test_file

    print class_weights
    if class_weights:
        class_weights_val = util.get_class_weights_au(util.readLinesFromFile(train_file))
    else:
        class_weights_val = None

    criterion = MultiCrossEntropy(class_weights= class_weights_val)
    criterion_str = 'MultiCrossEntropy'

    # criterion = MCE_CenterLoss_Combo(n_classes, feat_dim = 2048, bg = True, lambda_param = 0.0, alpha_param = 0.5, class_weights= class_weights_val)
    # criterion_str = 'Multi_Center_Combo'


    init = False

    strs_append_list = ['all_classes',all_classes,'just_primary',just_primary,'deno',deno,'limit',limit,'cw',class_weights, criterion_str, num_epochs]+dec_after+lr
    strs_append_list+=[post_pend] if len(post_pend)>0 else []
    strs_append = '_'.join([str(val) for val in strs_append_list])

    out_dir_train =  os.path.join(out_dir_meta,strs_append)
    final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
    


    if os.path.exists(final_model_file) and not test_mode and not retrain:
        print 'skipping',final_model_file
        return 
    else:
        print 'not skipping', final_model_file


    

    network_params = dict(n_classes=n_classes,deno = deno)

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
                network_params = network_params)

    if not test_mode:
        train_model(**train_params)
    
    if model_nums is None :
        model_nums = [num_epochs-1] 
    # print model_nums
    # model_nums = [0]+[i-1 for i in range(save_after,num_epochs,save_after)]
    # if model_nums[-1]!=(num_epochs-1):
    #     model_nums.append(num_epochs-1)

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
                post_pend=test_post_pend)
        test_model(**test_params)
        if not viz_mode:
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
                    post_pend=test_post_pend)
            test_model(**test_params)
            test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    test_data = test_data,
                    batch_size_val = batch_size_val,
                    gpu_id = gpu_id,
                    num_workers = 0,
                    second_thresh = second_thresh)
            visualize_sim_mat(**test_params)

def super_simple_experiment():
    # model_name = 'just_mill_2_1024'
    # model_name = 'graph_sim_direct_mill_cosine'
    model_name = 'graph_sim_direct_mill_cosine_do'
    # epoch_stuff = [25,25]
    # save_after = 5


    lr = [0.0001]
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = 500
    deno = 8
    save_after = 25
    
    test_mode = False
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''
    post_pend = ''

    class_weights = True
    test_after = 10
    all_classes = False
    just_primary = True
    model_nums = None
    
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
                        test_post_pend = test_post_pend)


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
    dir_meta= '../experiments/graph_sim_direct_mill_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001'
    dir_meta = dir_meta.replace(str_replace[0],str_replace[1])

    dirs = ['results_model_24_0_0.5/viz_sim_mat', 'results_model_24_0_0.5/viz_-1_0_0.5']
    dirs = [os.path.join(dir_meta, dir_curr) for dir_curr in dirs]

    dir_meta_new= '../experiments/just_mill_2_1024_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001'
    dir_meta_new = dir_meta_new.replace(str_replace[0],str_replace[1])
    dirs_new= ['results_model_99_0_0.5/viz_sim_mat', 'results_model_99_0_0.5/viz_-1_0_0.5']
    dirs += [os.path.join(dir_meta_new, dir_curr) for dir_curr in dirs_new]


    dir_meta_new= '../experiments/graph_sim_direct_mill_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001__noRelu'
    dir_meta_new = dir_meta_new.replace(str_replace[0],str_replace[1])
    dirs_new= ['results_model_24_0_0.5/viz_sim_mat', 'results_model_24_0_0.5/viz_-1_0_0.5']
    dirs += [os.path.join(dir_meta_new, dir_curr) for dir_curr in dirs_new]


    out_dir_html = os.path.join(dir_meta,'comparative_htmls')
    util.mkdir(out_dir_html)

    
    dir_strs = ['graph_sim_24','graph_pred_24','mill_2_layer_sim_99', 'mill_2_layer_pred_99','graph_sim_24_nr','graph_pred_24_nr']

    create_comparative_viz(dirs, class_names, dir_strs, out_dir_html) 



def main():

    # scripts_comparative()
    
    super_simple_experiment()


if __name__=='__main__':
    main()