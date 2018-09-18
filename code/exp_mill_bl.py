from train_test_mill import *
import models
from criterions import MultiCrossEntropy
import os
from helpers import util,visualize
from dataset import *
import numpy as np
import torch

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
                                    just_primary = False):

    out_dir_meta = '../experiments/'+model_name+'_'+dataset
    util.mkdir(out_dir_meta)
    num_epochs = epoch_stuff[1]

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

    init = False

    strs_append_list = ['all_classes',all_classes,'just_primary',just_primary,'deno',deno,'limit',limit,'cw',class_weights, criterion_str, num_epochs]+dec_after+lr
    strs_append = '_'.join([str(val) for val in strs_append_list])

    out_dir_train =  os.path.join(out_dir_meta,strs_append)
    final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
    


    if os.path.exists(final_model_file) and not test_mode:
        print 'skipping',final_model_file
        # return 
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
                trim_preds = trim_preds)
        test_model(**test_params)

def super_simple_experiment():
    model_name = 'graph_sim_mill_big'
    lr = [0.0001]
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = 500
    deno = 8
    save_after = 25
    test_mode = True
    class_weights = True
    test_after = 10
    all_classes = False
    just_primary = True
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
                        just_primary = just_primary)


def main():

    # num_feat = 10

    # arr = np.array(range(num_feat))[np.newaxis,:]
    # arr = np.triu(np.concatenate([np.roll(arr,num) for num in range(num_feat)],0))
    # arr = num_feat - (np.flipud(np.fliplr(arr))+arr)
    # arr = arr.astype(float)/np.sum(arr,axis = 1, keepdims = True)

    

    # print arr.shape
    # print arr
    


    # dir_files = '../data/ucf101/train_test_files'
    # train_file = os.path.join(dir_files,'train.txt')

    # weights = util.get_class_weights_au(util.readLinesFromFile(train_file))

    super_simple_experiment()


if __name__=='__main__':
    main()