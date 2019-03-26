import os
from helpers import util,visualize
import numpy as np
from globals import * 
import glob
def merge_det_fake(bin_keep):
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
    # num_det = len(idx_start_all)
    return idx_start_all, idx_end_all



def plot_video_in_range(dir_meta,dir_meta_bl, vid_name, out_dir, time_range):

    posts = ['GT','Merged']
    files= [os.path.join(dir_meta,vid_name+'_'+post_curr+'.npy') for post_curr in posts]
    for file_curr in files:
        if not os.path.exists(file_curr):
            print 'not exist',file_curr
            return
    data = [np.load(file_curr) for file_curr in files]
    [gt,merged] = data
    colors = ['b','g']

    if dir_meta_bl is not None:
        posts = ['GT','Merged']
        files_bl= [os.path.join(dir_meta_bl,vid_name+'_'+post_curr+'.npy') for post_curr in posts]
        data_bl = [np.load(file_curr) for file_curr in files_bl]
        data.append(data_bl[1])
        colors.append('r')

    
    # [gt,merged] = data

    # print gt.shape
    # print gt

    idx_file = 1
    x_vals_all = []
    widths_all = []
    y_vals = []

    for idx_file in range(len(data)):
        arr_curr = data[idx_file]
        # post_curr = posts[idx_file]

        # print arr_curr[1,:]
        bin_keep = arr_curr[1,:]>0
        idx_start_all,idx_end_all = merge_det_fake(bin_keep)
        widths = [idx_end_all[idx_curr] - idx_start_all[idx_curr] for idx_curr in range(len(idx_start_all))]
        x_vals_all.append(idx_start_all)
        widths_all.append(widths)
        y_vals.append(1)
        # print widths
        # print idx_start_all,idx_end_all

        
    out_file = os.path.join(out_dir,vid_name.replace('/','_')+'.jpg')
        # print out_file
        # print out_file.replace('..','http://vision6.idav.ucdavis.edu:8000/nn_net')

    visualize.plotBarsSubplot(out_file,x_vals_all =x_vals_all ,
                widths_all = widths_all ,
                y_vals = y_vals,
                colors = colors,
                xlim = [-1,arr_curr.shape[1]+1])
                # )
                # 
                # 
        # ,xlabel='',ylabel='',title='')

    # for idx_file in range(2):
    #     arr_curr = data[idx_file]
    #     post_curr = posts[idx_file]

    #     # print arr_curr[1,:]
    #     bin_keep = arr_curr[1,:]>0
    #     idx_start_all,idx_end_all = merge_det_fake(bin_keep)
    #     widths = [idx_end_all[idx_curr] - idx_start_all[idx_curr] for idx_curr in range(len(idx_start_all))]
    #     # print widths
    #     # print idx_start_all,idx_end_all

        
    #     out_file = os.path.join(out_dir,vid_name.replace('/','_')+'_'+post_curr+'.jpg')
    #     # print out_file
    #     # print out_file.replace('..','http://vision6.idav.ucdavis.edu:8000/nn_net')

    #     visualize.plotBars(out_file,x_vals = idx_start_all ,
    #             widths = widths ,
    #             y_val = 1,
    #             color = colors[idx_file],
    #             ylim = [-1,10],
    #             xlim = [-1,arr_curr.shape[1]+1])
        # ,xlabel='',ylabel='',title='')

def comparison_figs():
    print 'hello'
    bl_dir = '../experiments/wtalc_multi_video_with_L1/wtalc_multi_video_with_L1_feat_ret_True_in_out_2048_2048_deno_8_aft_nonlin_RL_L2_n_classes_20_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchFakeL1_CASL_250_step_250_0.1_0.001_0.001_lw_1.00_0.50__caslexp/results_model_249_original_class_0.0_0.5_-2/viz_-1_0.0_0.5_-2'
    our_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_500_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_0.50__noLimit/results_model_249_original_class_0_0.5_-2/viz_-1_0_0.5_-2'



    vids = ['CleanAndJerk/video_test_0000698',
    'CleanAndJerk/video_test_0001483',
    'BaseballPitch/video_test_0001038',
    'CliffDiving/video_test_0000844',
    'CliffDiving/video_test_0001076',
    'SoccerPenalty/video_test_0001118',
    'Billiards/video_test_0001267',
    'Shotput/video_test_0000045',
    'Shotput/video_test_0000593',
    'Shotput/video_test_0001268']


    vids = ['CleanAndJerk/video_test_0000698',
    # 'CleanAndJerk/video_test_0001483',
    'BaseballPitch/video_test_0001038',
    'CliffDiving/video_test_0000844',
    'CliffDiving/video_test_0001076',
    'SoccerPenalty/video_test_0001118',
    'Shotput/video_test_0001268']

    out_dir = '../scratch/qualitative_figs/bl'
    util.mkdir(out_dir)
    dir_in = bl_dir


    out_dir = '../scratch/qualitative_figs/us_stack_just_rel'
    util.mkdir(out_dir)
    dir_in = our_dir

    # (dir_meta,dir_meta_bl, vid_name, out_dir, time_range)

    for vid_name in vids:
        plot_video_in_range(our_dir,bl_dir, vid_name, out_dir, 0)
        # break
    visualize.writeHTMLForFolder(out_dir)



def make_vid_viz():
    
    dir_videos = '../data/ucf101/test_data/rgb_10_fps_256/'
    # video_name_list = ['video_test_0001268']
    # video_name_list = ['video_test_0001268',
    #                 'video_test_0001076',
    #                 'video_test_0001118',
    #                 'video_test_0001038',
    #                 'video_test_0000698',
    #                 'video_test_0000844']
    video_name_list = ['video_test_0000964',
'video_test_0000601',
'video_test_0001343',
'video_test_0000273',
'video_test_0001460']

    # ['video_test_0000026',
    #                 'video_test_0000073']



    # dir_videos.replace('..','maheennn_net'

    out_dir_html = '../scratch/qualitative_figs'
    util.mkdir(out_dir_html)
    # /disk1/maheen-data/nn_net/data

    dir_server = '/disk1'
    str_replace = ['..','/nn_net']
   
    for vid in video_name_list:
        dir_curr = os.path.join(dir_videos,vid)
        ims = glob.glob(os.path.join(dir_curr,'*.jpg'))
        idx_to_pick = np.linspace(0,len(ims),10,endpoint = False)
        print len(ims), ims[0]
        print idx_to_pick
        ims = [ims[int(idx)] for idx in idx_to_pick]

        out_file_html = os.path.join(out_dir_html,vid+'.html')
        im_row = []
        caption_row =[]
        for im in ims:
            im_row.append(im.replace(str_replace[0],str_replace[1]))
            # caption_row.append(os.path.split(im)[1])
            caption_row.append('')
        visualize.writeHTML(out_file_html, [im_row],[caption_row],height = 256, width = 455)
        print out_file_html


def plot_graph_size():
    graph_size = [1,2,3,4,8,16,32]
    xticks = [str(val) for val in graph_size]
    overlap_str = ['%.1f'%val for val in np.arange(0.1,0.6,0.1)]
    graph_vals = [[   63.7  ,   57.0  ,   47.7  ,   36.9  ,   26.2],
    [   63.2  ,   56.7  ,   47.9  ,   37.1  ,   26.6],
    [   64.0  ,   58.2  ,   49.8  ,   39.2  ,   28.2],
    [   63.7  ,   57.3  ,   48.2  ,   36.7  ,   25.5],
    [   62.7  ,   56.5  ,   47.0  ,   36.0  ,   23.9],
    [  62.7  ,   55.5  ,   46.6  ,   35.3  ,   24.0],
    [   61.1 , 54.1 , 44.6 , 32.8 , 21.9]]
    graph_vals = np.array(graph_vals)
    graph_vals = graph_vals.T

    out_file = '../scratch/qualitative_figs/graph_size.jpg'
    ylabel = 'Average Precision'
    xlabel = 'Graph Size'
    xAndYs = []
    legend_entries = []

    for idx_graph_val,graph_val in enumerate(graph_vals):
        if idx_graph_val<4:
            continue
        x = range(len(graph_val))
        xAndYs.append((x,graph_val))
        legend_entries.append(overlap_str[idx_graph_val]+' Overlap')
    visualize.plotSimple(xAndYs, out_file = out_file, xlabel = xlabel,ylabel = ylabel, legend_entries = legend_entries, xticks = [x,xticks])
        # plotSimple(xAndYs,out_file=None,title='',xlabel='',ylabel='',legend_entries=None,loc=0,outside=False,logscale=False,colors=None,xticks=None,ylim=None,noline = False)



def graph_overfitting():

    log_file_us = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_500_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_0.50__noLimit/log_det.txt'
    log_file_them = '../experiments/graph_multi_video_with_L1_retW/graph_multi_video_with_L1_retW_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_0.50__caslexp/log_det.txt'

    files = [log_file_us, log_file_them]
    x_vals = range(0,251,10)
    xAndYs = []
    for file_curr in files:
        lines = util.readLinesFromFile(file_curr)
        lines = lines[:26]
        print lines[0]
        det_vals = []
        # for line in lines:
        #     line = [val for val in line.split(' ') if val is not '']
        #     print line
        #     det_vals.append(float(line[-1]))
        #     raw_input()
        det_vals = [float(line.split('\t')[-1]) for line in lines]
        xAndYs.append((x_vals, det_vals))

    out_file = '../scratch/qualitative_figs/graph_overfitting.jpg'
    legend_entries  = ['Ours-MCASL','CASL-Graph']
    xlabel = 'Training Epoch'
    ylabel = 'Detection Accuracy'
    title = 'Detection Accuracy at 0.5 Overlap'
    visualize.plotSimple(xAndYs, out_file = out_file, xlabel = xlabel,ylabel = ylabel, legend_entries = legend_entries, title = title)
        # for line in lines:
            
        # print len(lines)
        # rel_lines = [line for line_curr in lines if line.starts


def main():
    make_vid_viz()
    # graph_overfitting()
    # plot_graph_size()
    # comparison_figs()



    # our_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_500_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_0.50__noLimit/results_model_249_original_class_0_0.5_-2/viz_-1_0_0.5_-2'

    # in_dir = our_dir

    # print class_names

    # for class_name in class_names:
    #     dir_curr = os.path.join(in_dir,class_name)
    #     video_files = glob.glob(os.path.join(dir_curr,'*.jpg'))
    #     video_files = [os.path.split(vid)[1].replace('.jpg','') for vid in video_files]
    #     for vid_name in video_files:
    #         plot_video_in_range(dir_curr,None, vid_name, dir_curr,0)

    #     visualize.writeHTMLForFolder(dir_curr)
    


if __name__=='__main__':
    main()
