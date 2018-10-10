import os
import numpy as np
import scipy.io
import glob
from helpers import util, visualize
import sklearn.metrics
from globals import class_names

def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis = axis, keepdims= True))
    return e_x / e_x.sum(axis=axis, keepdims = True) # only difference

def get_similarity(features):
    # features = np.load(npy_file)
    # norms = np.linalg.norm(features, axis = 1, keepdims = True)
    # features = features/norms
    
    
    # sim_mat = sklearn.metrics.pairwise.cosine_similarity(features)
    # sim_mat = np.matmul(features, features.T)
    # norms = np.sqrt(np.sum(np.power(features, 2),1))[:, np.newaxis]
    # norms_mat = np.matmul(norms, norms.T)
    # sim_mat = sim_mat/norms_mat

    sim_mat = np.matmul(features, features.T)
    # sim_mat[np.eye(sim_mat.shape[0])>0]=0
    # print sim_mat[:5,:5]

    sim_mat = softmax(sim_mat, axis = 1)
    # print sim_mat.shape
    # raw_input()
    
    return sim_mat

def get_gt_vector(vid_name, out_shape_curr, class_idx, test = True):
    

    class_name = class_names[class_idx]
    if test:
        mat_file = os.path.join('../TH14evalkit','mat_files', class_name+'_test.mat')
    else:
        mat_file = os.path.join('../TH14evalkit', class_name+'.mat')

    loaded = scipy.io.loadmat(mat_file)
    
    gt_vid_names_all = loaded['gtvideonames'][0]
    gt_class_names = loaded['gt_events_class'][0]
    gt_time_intervals = loaded['gt_time_intervals'][0]
    gt_time_intervals = np.array([a[0] for a in gt_time_intervals])
    
    bin_keep = np.array(gt_vid_names_all) == vid_name
    gt_time_intervals = gt_time_intervals[bin_keep]
    det_times = np.array(range(0,out_shape_curr))*16./25.
    
    gt_vals = np.zeros(det_times.shape)
    for gt_time_curr in gt_time_intervals:
        idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
        idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
        gt_vals[idx_start:idx_end] = 1

    return gt_vals, det_times


def save_sim_viz(vid_name, out_shape_curr, sim_mat, class_idx, out_dir):
    gt_vals, det_times = get_gt_vector(vid_name, out_shape_curr, class_idx)
    out_dir_curr = os.path.join(out_dir, class_names[class_idx])
    util.mkdir(out_dir_curr)
    pos_rows = sim_mat[gt_vals>0,:]
    pos_rows = np.mean(pos_rows, axis = 0)

    neg_rows = sim_mat[gt_vals<1,:]
    
    neg_rows = np.mean(neg_rows, axis = 0)
    # for idx_pos_row, pos_row in enumerate(pos_rows):
    max_val = max(np.max(pos_rows),np.max(neg_rows))
    gt_vals_curr = gt_vals*max_val

    arr_plot = [(det_times, curr_arr) for curr_arr in [gt_vals_curr,pos_rows,neg_rows]]
    legend_entries = ['gt', 'pos', 'neg']
    # idx_pos_row = str(idx_pos_row)
    out_file_curr = os.path.join(out_dir_curr, vid_name+'.jpg')
    title = vid_name
    # +' '+idx_pos_row

    visualize.plotSimple(arr_plot, out_file = out_file_curr, title = title, xlabel = 'time', ylabel = 'max sim', legend_entries = legend_entries)
    print out_file_curr        
    
    # print gt_vals
    # raw_input()

    

    # idx_pos = gt_vals>0
    # idx_neg = gt_vals<1
    # sim_pos_all = []
    # sim_neg_all = []
    # for idx_idx_curr, idx_curr in enumerate(np.where(idx_pos)[0]):
    #     sim_pos = sim_mat[idx_curr, idx_pos]
    #     sim_neg = sim_mat[idx_curr, idx_neg]
    #     sim_pos_all.append(sim_pos[np.newaxis,:])
    #     sim_neg_all.append(sim_neg[np.newaxis,:])

    # sim_pos_all = np.concatenate(sim_pos_all, axis = 0)
    # sim_neg_all = np.concatenate(sim_neg_all, axis = 0)

    # sim_pos_mean = np.mean(sim_pos_all,axis = 0)
    # sim_neg_mean = np.mean(sim_neg_all, axis = 0)

    # pos_vals = np.zeros(gt_vals.shape)
    # pos_vals[gt_vals>0]=sim_pos_mean
    # neg_vals = np.zeros(gt_vals.shape)
    # neg_vals[gt_vals<1]=sim_neg_mean

    # max_val = max(np.max(pos_vals),np.max(neg_vals))
    # gt_vals = gt_vals*max_val

    # arr_plot = [(det_times, curr_arr) for curr_arr in [gt_vals,pos_vals,neg_vals]]
    # legend_entries = ['gt', 'pos', 'neg']

    # out_file_curr = os.path.join(out_dir_curr, vid_name+'.jpg')
    # title = vid_name

    # visualize.plotSimple(arr_plot, out_file = out_file_curr, title = title, xlabel = 'time', ylabel = 'max sim', legend_entries = legend_entries)
    # print out_file_curr



def save_sim_viz_mean(vid_name, out_shape_curr, sim_mat, class_idx, out_dir):
    gt_vals, det_times = get_gt_vector(vid_name, out_shape_curr, class_idx)

    out_dir_curr = os.path.join(out_dir, class_names[class_idx])
    util.mkdir(out_dir_curr)

    idx_pos = gt_vals>0
    idx_neg = gt_vals<1
    sim_pos_all = []
    sim_neg_all = []
    for idx_idx_curr, idx_curr in enumerate(np.where(idx_pos)[0]):
        sim_pos = sim_mat[idx_curr, idx_pos]
        sim_neg = sim_mat[idx_curr, idx_neg]
        sim_pos_all.append(sim_pos[np.newaxis,:])
        sim_neg_all.append(sim_neg[np.newaxis,:])

    sim_pos_all = np.concatenate(sim_pos_all, axis = 0)
    sim_neg_all = np.concatenate(sim_neg_all, axis = 0)

    sim_pos_mean = np.mean(sim_pos_all,axis = 0)
    sim_neg_mean = np.mean(sim_neg_all, axis = 0)

    pos_vals = np.zeros(gt_vals.shape)
    pos_vals[gt_vals>0]=sim_pos_mean
    neg_vals = np.zeros(gt_vals.shape)
    neg_vals[gt_vals<1]=sim_neg_mean

    max_val = max(np.max(pos_vals),np.max(neg_vals))
    gt_vals = gt_vals*max_val

    arr_plot = [(det_times, curr_arr) for curr_arr in [gt_vals,pos_vals,neg_vals]]
    legend_entries = ['gt', 'pos', 'neg']

    out_file_curr = os.path.join(out_dir_curr, vid_name+'.jpg')
    title = vid_name

    visualize.plotSimple(arr_plot, out_file = out_file_curr, title = title, xlabel = 'time', ylabel = 'max sim', legend_entries = legend_entries)
    print out_file_curr

def make_htmls(out_dir):
    for class_name in class_names:
        out_dir_curr = os.path.join(out_dir, class_name)
        visualize.writeHTMLForFolder(out_dir_curr)


def script_viewing_sim():

    dir_files = '../data/ucf101/train_test_files'
    n_classes = 20
    train_file = os.path.join(dir_files, 'train_just_primary.txt')
    test_file = os.path.join(dir_files, 'test_just_primary.txt')
    
    out_dir = '../scratch/debugging_graph_self1'
    util.mkdir(out_dir)

    train_lines = util.readLinesFromFile(test_file)
    train_npy = [line_curr.split(' ') for line_curr in train_lines]
    for line_curr in train_lines:
        line_curr = line_curr.split(' ')
        npy_file = line_curr[0]
        anno = [int(val) for val in line_curr[1:]]
        anno = np.array(anno)
        assert np.sum(anno)==1
        class_idx = np.where(anno)[0][0]
        
        out_dir_curr = os.path.join(out_dir, class_names[class_idx])
        util.mkdir(out_dir_curr)


        features = np.load(npy_file)
        out_shape_curr = features.shape[0]
        vid_name = os.path.split(npy_file)[1]
        vid_name = vid_name[:vid_name.rindex('.')]

        sim_mat = get_similarity(features)
        gt_vals, det_times = get_gt_vector(vid_name, out_shape_curr, class_idx)

        # idx_pos = np.where(gt_vals>0)[0]
    
        idx_pos = gt_vals>0
        idx_neg = gt_vals<1
        # print idx_pos
        sim_pos_all = []
        sim_neg_all = []
        for idx_idx_curr, idx_curr in enumerate(np.where(idx_pos)[0]):
            

            sim_pos = sim_mat[idx_curr, idx_pos]
            sim_neg = sim_mat[idx_curr, idx_neg]
            sim_pos_all.append(sim_pos[np.newaxis,:])
            sim_neg_all.append(sim_neg[np.newaxis,:])


            # idx_pos_leave = np.in1d
            # sim_rel = sim_mat[idx_curr, idx_pos]
            # print sim_rel.shape
            # print sim_rel
            # print sim_rel[idx_idx_curr]
            # print np.min(sim_rel), np.max(sim_rel), np.mean(sim_rel)
            # sim_rel = sim_mat[idx_curr, :]
            # print sim_rel.shape
            # print np.min(sim_rel), np.max(sim_rel), np.mean(sim_rel)

        sim_pos_all = np.concatenate(sim_pos_all, axis = 0)
        sim_neg_all = np.concatenate(sim_neg_all, axis = 0)
        # print sim_pos_all.shape
        # print sim_neg_all.shape

        sim_pos_mean = np.mean(sim_pos_all,axis = 0)
        sim_neg_mean = np.mean(sim_neg_all, axis = 0)

        pos_vals = np.zeros(gt_vals.shape)
        pos_vals[gt_vals>0]=sim_pos_mean
        neg_vals = np.zeros(gt_vals.shape)
        neg_vals[gt_vals<1]=sim_neg_mean

        arr_plot = [(det_times, curr_arr) for curr_arr in [gt_vals,pos_vals,neg_vals]]
        legend_entries = ['gt', 'pos', 'neg']
        out_file_curr = os.path.join(out_dir_curr, vid_name+'.jpg')
        title = vid_name

        visualize.plotSimple(arr_plot, out_file = out_file_curr, title = title, xlabel = 'time', ylabel = 'max sim', legend_entries = legend_entries)
        print out_file_curr
        # print sim_pos_mean
        # print sim_neg_mean 

        # break

    for class_name in class_names:
        out_dir_curr = os.path.join(out_dir, class_name)
        visualize.writeHTMLForFolder(out_dir_curr)


def readTrainTestFile(file_curr):
    lines = util.readLinesFromFile(file_curr)
    
    anno_all = []
    npy_files_all = []
    for line in lines:
        line = line.split(' ')
        npy_file = line[0]
        anno = [int(val) for val in line[1:]]
        anno_all.append(anno)
        npy_files_all.append(npy_file)

    return npy_files_all, anno_all


def script_make_gt_vecs():
    dir_files = '../data/ucf101/train_test_files'
    out_dir_gt_vec = '../data/ucf101/gt_vecs'
    util.mkdir(out_dir_gt_vec)

    n_classes = 20
    just_primary = True
    # if just_primary:
    train_file = os.path.join(dir_files, 'train_just_primary.txt')
    test_file = os.path.join(dir_files, 'test_just_primary.txt')
    out_dir_curr = os.path.join(out_dir_gt_vec,'just_primary')

    # else:
    #     train_file = os.path.join(dir_files, 'train.txt')
    #     test_file = os.path.join(dir_files, 'test.txt')
    #     out_dir_curr = os.path.join(out_dir_gt_vec,'regular')


    files = [train_file, test_file]
    test_status = [False, True]
    util.mkdir(out_dir_curr)
    post_pend = '_gt_vec'

    for file_curr, test_stat  in zip(files, test_status):
        npy_files, annos = readTrainTestFile(file_curr)
        for npy_file, anno_curr in zip(npy_files, annos):
            vid_name = os.path.split(npy_file)[1]
            vid_name = vid_name[:vid_name.rindex('.')]
            out_file_curr = os.path.join(out_dir_curr,vid_name+'.npy')

            if not os.path.exists(out_file_curr):
                out_shape_curr = np.load(npy_file).shape[0]
                class_idx = anno_curr.index(1)
                gt_vec,_ = get_gt_vector(vid_name, out_shape_curr, class_idx, test = test_stat)
                gt_vec = gt_vec*(class_idx+1)
                assert np.max(gt_vec)>0
                np.save(out_file_curr, gt_vec)
        
        new_file_curr = file_curr[:file_curr.rindex('.')]+post_pend+'.txt'
        new_lines = []
        for npy_file, anno_curr in zip(npy_files, annos):
            vid_name = os.path.split(npy_file)[1]
            vid_name = vid_name[:vid_name.rindex('.')]

            out_file_curr = os.path.join(out_dir_curr,vid_name+'.npy')
            new_line = [npy_file, out_file_curr]+[str(val) for val in anno_curr]
            new_line = ' '.join(new_line)
            new_lines.append(new_line)
        # print len(new_lines), new_file_curr, new_lines[0]

        util.writeFile(new_file_curr, new_lines)

    

def main():
    # script_make_gt_vecs()
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary/'
    npys = ['../data/ucf101/gt_vecs/just_primary/video_validation_0000666.npy']

    # npys = glob.glob(os.path.join(dir_gt_vecs,'*.npy'))
    for npy in npys:

        np_curr = np.load(npy)
        print np_curr.shape
        print npy,np.min(np_curr),np.max(np_curr) 
        assert np.min(np_curr)==0
        assert np.max(np_curr)>0

            
            
        



    




if __name__=='__main__':
    main()