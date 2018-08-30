import sys
sys.path.append('./')
import os
import numpy as np
import scipy.io
import glob
from helpers import util, visualize


def interval_single_overlap_val_seconds(i1, i2, norm_type = 0):
    i1 = [np.min(i1), np.max(i1)]
    i2 = [np.min(i2), np.max(i2)]

    ov = 0
    if norm_type<0:
        ua = 1
    elif norm_type==1:
        ua = i1[1] - i1[0]
    elif norm_type==2:
        ua = i2[1] - i2[0]
    else:
        bu = [min(i1[0], i2[0]), max(i1[1], i2[1])]
        ua = bu[1] - bu[0]

    bi = [max(i1[0], i2[0]), min(i1[1], i2[1])]
    iw = bi[1] - bi[0]

    if iw>0:
        if norm_type<0:
            ov = iw
        else:
            ov = iw/float(ua)

    return ov

def interval_overlap_val_seconds(i1, i2, norm_type=0):
    ov = np.zeros((i1.shape[0],i2.shape[0]))

    for i in range(i1.shape[0]):
        for j in range(i2.shape[0]):
            ov[i,j] = interval_single_overlap_val_seconds(i1[i,:], i2[j,:], norm_type)
            
    return ov


def pr_ap(rec, prec):
    ap = 0
    recall_points = np.arange(0,1.1,0.1)
    # print recall_points
    for t in recall_points:
        p = prec[rec>=t]
        if p.size ==0:
            p=0
        else:
            p= np.max(p)
        
        ap = ap + p/float(recall_points.size)
        # print t,p,ap

    return ap





def test_ov():
    loaded = scipy.io.loadmat('../TH14evalkit/i1_i2.mat')
    i1 = loaded['i1']
    i2 = loaded['i2']
    ov_org = loaded['ov']
    print i1.shape,i2.shape,ov_org.shape
    ov = interval_overlap_val_seconds(i1, i2,2)
    print ov.shape
    print ov
    print ov_org
    print np.abs(ov_org-ov)

def test_pr_ap():
    loaded = scipy.io.loadmat('../TH14evalkit/rec_prec.mat')
    rec = loaded['rec']
    prec = loaded['prec']
    ap_org = loaded['ap'][0][0]
    print rec.shape, prec.shape, ap_org.shape
    ap = pr_ap(rec, prec)
    print ap
    print ap_org
    assert np.all(ap_org-ap)    

def event_det_pr(det_vid_names, det_time_intervals, det_class_names, det_conf, gt_vid_names, gt_time_intervals, gt_class_names, class_name, overlap_thresh):

    video_names = np.unique(det_vid_names+gt_vid_names)
    num_pos = gt_class_names.count(class_name)
    assert num_pos>0

    gt_class_names = np.array(gt_class_names)
    gt_vid_names = np.array(gt_vid_names)
    det_vid_names = np.array(det_vid_names)
    det_class_names = np.array(det_class_names)

    ind_gt_class = gt_class_names==class_name
    ind_amb_class = gt_class_names=='Ambiguous'
    ind_det_class = det_class_names==class_name

    tp_conf = []
    fp_conf = []
    for idx_video_name, video_name in enumerate(video_names):
        gt = np.logical_and(gt_vid_names==video_name, ind_gt_class)
        amb = np.logical_and(gt_vid_names==video_name, ind_amb_class)
        det = np.logical_and(det_vid_names==video_name, ind_det_class)
        # det = det_vid_names==video_name

        if np.sum(det)>0:
            
            ind_free = np.ones((np.sum(det),))
            ind_amb = np.zeros((np.sum(det),))
            

            det_conf_curr = det_conf[det]
            idx_sort = np.argsort(det_conf_curr)[::-1]
            idx_sort = np.where(det)[0][idx_sort]

            det_conf_curr = det_conf[idx_sort]
            
            det_vid_names_curr = det_vid_names[idx_sort]
            assert  np.unique(det_vid_names_curr).size==1 and det_vid_names_curr[0]== video_name

            if np.sum(gt)>0:
                ov = interval_overlap_val_seconds(gt_time_intervals[gt,:], det_time_intervals[idx_sort,:])
                # print ov.shape

                for k in range(ov.shape[0]):    
                    ind = np.where(ind_free>0)[0]
                    im = np.argmax(ov[k,ind])
                    vm = ov[k,ind][im]
                    if vm>overlap_thresh:
                        ind_free[ind[im]]=0

            if np.sum(amb)>0:
                ov_amb = interval_overlap_val_seconds(gt_time_intervals[amb,:], det_time_intervals[idx_sort,:])
                ind_amb = np.sum(ov_amb,0)

            tp_conf.extend(list(det_conf_curr[np.where(ind_free==0)[0]]))
            fp_conf.extend(list(det_conf_curr[np.where(np.logical_and(ind_free==1, ind_amb==0))[0]]))

    # print tp_conf, fp_conf
    conf = np.array([tp_conf + fp_conf,list(np.ones((len(tp_conf),))) + list(2*np.ones((len(fp_conf),)))])
    idx_sort = np.argsort(conf[0,:])[::-1]
    tp = np.cumsum(conf[1,idx_sort]==1)
    fp = np.cumsum(conf[1,idx_sort]==2)
    rec = tp/float(num_pos)
    prec = tp/(fp+tp).astype(float)
    ap = pr_ap(rec, prec)
    return rec, prec, ap
        




def test_event_det_pr():
    # loaded = scipy.io.loadmat('../TH14evalkit/gt_det_stuff.mat')
    class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    
    gt_ov = util.readLinesFromFile('analysis/test.txt')
    gt_ov = [float(val) for val in gt_ov]
    diffs = []
    idx_ov = 0

    for class_name in class_names:
        print class_name
        mat_file = os.path.join('../TH14evalkit',class_name+'.mat')
        loaded = scipy.io.loadmat(mat_file)
        gt_vid_names = loaded['gtvideonames'][0]
        det_vid_names = loaded['detvideonames'][0]
        gt_class_names = loaded['gt_events_class'][0]
        det_class_names = loaded['det_events_class'][0]
        gt_time_intervals = loaded['gt_time_intervals'][0]
        det_time_intervals = loaded['det_time_intervals'][0]
        det_conf = loaded['det_conf'][0]

        # class_name = 'BaseballPitch'
        # overlap_thresh = 0.1

        arr_meta = [gt_vid_names,det_vid_names,gt_class_names,det_class_names]

        arr_out = []
        for arr_curr in arr_meta:
            arr_curr = [str(a[0]) for a in arr_curr]
            arr_out.append(arr_curr)
        
        [gt_vid_names,det_vid_names,gt_class_names,det_class_names] = arr_out

        gt_time_intervals = np.array([a[0] for a in gt_time_intervals])
        det_time_intervals = np.array([a[0] for a in det_time_intervals])
        det_conf = np.array([a[0][0] for a in det_conf])

        for overlap_thresh in np.arange(0.1,0.6,0.1):
            rec, prec, ap = event_det_pr(det_vid_names, det_time_intervals, det_class_names, det_conf, gt_vid_names, gt_time_intervals, gt_class_names, class_name, overlap_thresh)
            print ap, overlap_thresh
            diffs.append(ap - gt_ov[idx_ov])
            idx_ov+=1
    
    diffs = np.abs(np.array(diffs))
    print np.mean(diffs), np.min(diffs), np.max(diffs)







def main():
    print 'hello'
    # test_ov()
    test_event_det_pr()





if __name__=='__main__':
    main()


