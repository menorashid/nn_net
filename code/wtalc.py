import numpy as np
import time
from scipy.signal import savgol_filter
import sys
import scipy.io as sio 
from analysis import evaluate_thumos as et

def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def smooth(v):
   return v
   #l = min(3, len(v)); l = l - (1-l%2)
   #if len(v) <= 3:
   #   return v
   #return savgol_filter(v, l, 1) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

def filter_segments(segment_predict, videonames, ambilist):
   ind = np.zeros(np.shape(segment_predict)[0])
   for i in range(np.shape(segment_predict)[0]):
      vn = videonames[int(segment_predict[i,0])]
      for a in ambilist:
         if a[0]==vn:
            gt = range(int(round(float(a[2])*25/16)), int(round(float(a[3])*25/16)))
            pd = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
            IoU = float(len(set(gt).intersection(set(pd))))/float(len(set(gt).union(set(pd))))
            if IoU > 0:
               ind[i] = 1
   s = [segment_predict[i,:] for i in range(np.shape(segment_predict)[0]) if ind[i]==0]
   return np.array(s)

def getLocMAP(predictions, pmfs, vid_names,first_thresh, second_thresh, th, annotation_path):

   gtsegments = np.load(annotation_path + '/segments.npy')
   gtlabels = np.load(annotation_path + '/labels.npy')
   # gtlabels = np.load(annotation_path + '/labels.npy')
   videoname = np.load(annotation_path + '/videoname.npy'); videoname = np.array([v.decode('utf-8') for v in videoname])
   subset = np.load(annotation_path + '/subset.npy'); subset = np.array([s.decode('utf-8') for s in subset])
   classlist = np.load(annotation_path + '/classlist.npy'); classlist = np.array([c.decode('utf-8') for c in classlist])
   duration = np.load(annotation_path + '/duration.npy')
   ambilist = annotation_path + '/Ambiguous_test.txt'

   ambilist = list(open(ambilist,'r'))
   ambilist = [a.strip('\n').split(' ') for a in ambilist]
   # print (ambilist)

   # # keep training gtlabels for plotting
   gtltr = []
   for i,s in enumerate(subset):
      if subset[i]=='validation' and len(gtsegments[i]):
         gtltr.append(gtlabels[i])
   gtlabelstr = gtltr
   
   
   bin_keep = np.in1d(videoname, vid_names)
   
   gtsegments = [gtsegments[idx] for idx,i in enumerate(bin_keep) if i]
   gtlabels = [gtlabels[idx] for idx,i in enumerate(bin_keep) if i]
   
   videoname = videoname[bin_keep]
   # print videoname.shape

   # # which categories have temporal labels ?
   templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

   # the number index for those categories.
   templabelidx = []
   for t in templabelcategories:
      templabelidx.append(str2ind(t,classlist))
   
   class_names_chosen = [str(val) for val in np.array(classlist)[templabelidx]]


   # process the predictions such that classes having greater than a certain threshold are detected only
   predictions_mod = []
   c_score = []
   for idx_pmf, pmf in enumerate(pmfs):
      p = predictions[idx_pmf]
      ind = pmf > first_thresh
      c_score.append(pmf)
      new_pred = np.zeros((np.shape(p)[0],np.shape(p)[1]), dtype='float32')
      predictions_mod.append(p*ind)
   predictions = predictions_mod

   detection_results = []
   for i,vn in enumerate(videoname):
      detection_results.append([])
      detection_results[i].append(vn)

   ap = []
   for c in templabelidx:
      segment_predict = []
      # Get list of all predictions for class c
      for i in range(len(predictions)):
         tmp = smooth(predictions[i][:,c])
         threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*second_thresh
         vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
         vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
         s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
         e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
         for j in range(len(s)):
            # aggr_score = np.max(tmp[s[j]:e[j]]) + 0.7*c_score[i][c]
            if e[j]-s[j]>=2:               
               segment_predict.append([i,s[j],e[j],np.max(tmp[s[j]:e[j]])])
                  # +0.7*c_score[i][c]])
               detection_results[i].append([classlist[c], s[j], e[j], np.max(tmp[s[j]:e[j]])])
                  # +0.7*c_score[i][c]])
      segment_predict = np.array(segment_predict)
      segment_predict = filter_segments(segment_predict, videoname, ambilist)

   
      # Sort the list of predictions for class c based on score
      if len(segment_predict) == 0:
         return 0
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]

      # Create gt list 
      segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j],classlist)==c]
      gtpos = len(segment_gt)

      # Compare predictions and gt
      # tp, fp = [], []

      list_taken = np.zeros((segment_predict.shape[0]))
      tp = np.zeros((segment_predict.shape[0]))
      fp = np.zeros((segment_predict.shape[0]))
      

      for i in range(len(segment_predict)):
         flag = 0.
         for j in range(len(segment_gt)):
            if segment_predict[i][0]==segment_gt[j][0]:
               gt = range(int(round(segment_gt[j][1]*25/16)), int(round(segment_gt[j][2]*25/16)))
               p = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
               IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
               # print segment_gt[j]
               # print gt
               # print p

               # gt_curr = [segment_gt[j][1], segment_gt[j][2]]
               # pred_curr = [segment_predict[i][1]*16/25, segment_predict[i][2]*16/25]
               # print pred_curr
               # print gt_curr
               # raw_input()

               # IoU = et.interval_single_overlap_val_seconds(gt_curr, pred_curr)

               if IoU >= th:
                  # print IoU, IoU_new
                  # raw_input()
                  # flag = 1.
                  flag = segment_predict[i][3]
                  list_taken[i]=1
                  tp[i]= flag
                  # fp[i] = 1-flag
                  del segment_gt[j]
                  break
         # fp.append(1.-flag)
      # tp[list_taken==1] = segment_predict[list_taken==1,3]
      # tp[list_taken==0] = 1-segment_predict[list_taken==0,3]
      fp[list_taken==0] = segment_predict[list_taken==0,3]
      # print sum(tp==0), sum(fp==0), len(segment_predict)
      # raw_input()
      tp_c = np.cumsum(tp)
      fp_c = np.cumsum(fp)
      if sum(tp)==0:
         prc = 0.
      else:
         prc = np.sum((tp_c/(fp_c+tp_c))*tp)/gtpos
      ap.append(prc)


   return ap, 100*np.mean(ap), class_names_chosen
  

def getDetectionMAP(predictions, pmfs, vid_names, first_thresh, second_thresh, annotation_path = 'Thumos14reduced-Annotations/'):
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
   dmap_list = []
   ap_list = []
   for iou in iou_list:
      ap, dmap, class_names_chosen = getLocMAP(predictions, pmfs, vid_names, first_thresh, second_thresh, iou, annotation_path)
      ap_list.append(ap)
      dmap_list.append(dmap)

   return ap_list, dmap_list, iou_list, class_names_chosen

