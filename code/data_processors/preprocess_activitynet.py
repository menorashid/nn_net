import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import numpy as np
import scipy.io
import json
from pprint import pprint

dir_meta = '../data/activitynet'
anno_dir = os.path.join(dir_meta,'annos')
json_file = os.path.join(anno_dir,'activity_net.v1-2.min.json')

def get_i3d():
    i3d_file = '../data/i3d_features/ActivityNet1.2-I3D-JOINTFeatures.npy'
    features = np.load(i3d_file)
    return features
    
def split_into_datasets(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    print data['version']
    database = data['database']
    train_data = []
    
    for k in database:
        if str(database[k]['subset'])=='testing':
            continue

        train_data.append(database[k])
    return train_data


def get_rel_anno_info():
    train_data = split_into_datasets(json_file)

    ids = []
    times_all = []
    labels_all = []
    train_val = []
    durations = []

    for vid in train_data:
        id_curr = str(vid['url'])
        id_curr = id_curr[id_curr.rindex('?')+1:].replace('=','_')
        ids.append(id_curr)
        durations.append(vid['duration'])
        labels = []
        times = []
        
        train_val.append(str(vid['subset']))

        for segment in vid['annotations']:
            time_curr = segment['segment']
            class_curr = segment['label']
            times.append(time_curr)
            labels.append(class_curr)

        # print len(times)
        times = np.array(times)
        labels_all.append(labels)
        times_all.append(times)

    labels = []
    for labels_curr in labels_all:
        labels = labels+labels_curr
    labels = list(set(labels))
    labels.sort()

    return ids, durations, train_val, labels_all, times_all, labels
    
    # # for k in labels:
    # #     print k
    # print len(labels)

    # for id_curr in ids:
    #     if id_curr.startswith('v')
    #         count = count+1


def save_npys():
    out_dir_features = os.path.join(dir_meta,'i3d')
    util.mkdir(out_dir_features)
    out_dir_train = os.path.join(out_dir_features,'train_data')
    out_dir_val = os.path.join(out_dir_features,'val_data')
    util.mkdir(out_dir_train)
    util.mkdir(out_dir_val)
    out_dirs = [out_dir_train,out_dir_val]

    features = get_i3d()
    ids, durations, train_val, labels_all, times_all = get_rel_anno_info()
    assert len(features)==len(ids)

    train_val = np.array(train_val)
    train_val_bool = np.zeros(train_val.shape).astype(int)
    train_val_bool[train_val=='validation']= 1

    for idx_id_curr,id_curr in enumerate(ids):
        features_curr = features[idx_id_curr]
        duration_curr = durations[idx_id_curr]
        feature_len = features_curr.shape[0]
        pred_len = duration_curr*25//16
        
        diff = np.abs(pred_len - feature_len)
        # diffs.append(diff)
        if diff>2:
            print 'Continuing',diff, feature_len, pred_len, duration_curr, train_val_bool[idx_id_curr]
            continue

        assert diff<=2

        out_file_curr = os.path.join(out_dirs[train_val_bool[idx_id_curr]],id_curr+'.npy')
        
        np.save(out_file_curr, features_curr)

def write_train_test_files():
    out_dir_features = os.path.join(dir_meta,'i3d')
    util.mkdir(out_dir_features)
    out_dir_train = os.path.join(out_dir_features,'train_data')
    out_dir_val = os.path.join(out_dir_features,'val_data')
    util.mkdir(out_dir_train)
    util.mkdir(out_dir_val)
    out_dirs = [out_dir_train,out_dir_val]

    out_dir_anno =os.path.join(dir_meta,'train_test_files')
    util.mkdir(out_dir_anno)
    anno_files = [os.path.join(out_dir_anno,'train.txt'),os.path.join(out_dir_anno,'val.txt')]

    ids, durations, train_val, labels_all, times_all,labels = get_rel_anno_info()

    print len(ids),len(set(ids))

    for dir_curr,anno_file in zip(out_dirs,anno_files):
        anno_lines = []

        npy_files = glob.glob(os.path.join(dir_curr,'*.npy'))
        for npy_file in npy_files:
            anno_curr = np.zeros((len(labels),))

            id_curr = os.path.split(npy_file)[1]
            id_curr = id_curr[:id_curr.rindex('.')]
            idx_id_curr = ids.index(id_curr)
            labels_curr = labels_all[idx_id_curr]
            labels_curr = list(set(labels_curr))
            for label_curr in labels_curr:
                anno_curr[labels.index(str(label_curr))] = 1

            assert np.sum(anno_curr)>0
            assert np.sum(anno_curr)==len(labels_curr)

            anno_curr = [str(int(val)) for val in anno_curr]

            line_curr = ' '.join([npy_file]+anno_curr)
            anno_lines.append(line_curr)

        print anno_file, len(anno_lines)
        util.writeFile(anno_file, anno_lines)



def write_gt_numpys():
    out_dir_gt = os.path.join(dir_meta,'gt_npys')
    util.mkdir(out_dir_gt)
    
    required_str = 'val'
    out_file = os.path.join(out_dir_gt,required_str+'.npz')
    
    ids, durations, train_val, labels_all, times_all,labels = get_rel_anno_info()

    print np.unique(train_val)
    # gt_vid_names = loaded['gtvideonames'][0]
    # gt_class_names = loaded['gt_events_class'][0]
    # gt_time_intervals = loaded['gt_time_intervals'][0]
    
    gt_vid_names =[]
    gt_class_names = []
    gt_time_intervals = []

    for idx_vid in range(len(ids)):
        if not train_val[idx_vid].startswith(required_str):
            # print 'continuing',train_val[idx_vid]
            continue

        time_rel = times_all[idx_vid]
        label_rel = labels_all[idx_vid]

        num_instances = len(time_rel)
        
        id_curr = ids[idx_vid]
        id_curr = [id_curr for idx in range(num_instances)]
        
        assert len(label_rel)==len(time_rel)==len(id_curr)

        gt_vid_names+=id_curr
        
        gt_time_intervals.append(time_rel)
        
        label_rel = [str(label_curr) for label_curr in label_rel]
        gt_class_names += label_rel

    gt_time_intervals = np.concatenate(gt_time_intervals, axis = 0)
    gt_class_names = np.array(gt_class_names)
    gt_vid_names = np.array(gt_vid_names)
    print gt_class_names[0], gt_class_names.shape
    print gt_vid_names[0], gt_vid_names.shape
    print gt_time_intervals.shape,gt_time_intervals[0]

    print out_file
    np.savez(out_file,gt_class_names = gt_class_names, gt_vid_names = gt_vid_names, gt_time_intervals = gt_time_intervals)

    # data = np.load(out_file)
    # gt_class_names = data['gt_class_names']
    # gt_vid_names = data['gt_vid_names']
    # gt_time_intervals = data['gt_time_intervals']
    # print gt_class_names.shape,gt_class_names[0]
    # print gt_vid_names.shape,gt_vid_names[0]
    # print gt_time_intervals.shape,gt_time_intervals[0]



def main():
    # write_gt_numpys()
    
    # ids, durations, train_val, labels_all, times_all,labels = get_rel_anno_info()
    # print type(labels),labels[0]
    # out_file_labels = os.path.join('../data/activitynet','gt_labels_sorted.txt')
    # util.writeFile(out_file_labels,labels)

    # write_train_test_files()
    # save_npys()
    # pass
    # explore_features()

    



    
    
    


if __name__=='__main__':
    main()