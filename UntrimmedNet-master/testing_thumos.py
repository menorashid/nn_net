import numpy as np
import sys
sys.path.append('../code')
from helpers import visualize,util
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['GLOG_minloglevel'] = '3' 
import caffe
import cPickle as pkl
import glob
import scipy.special
import multiprocessing

def save_im((attention_weight,attention_bias,vid_name,out_file,idx_vid_name)):
    if idx_vid_name%10==0:
        print idx_vid_name  
    vid_data = pkl.load(open(vid_name,'r'))
    flow_features = vid_data['flow_features'].T
    flow_features = flow_features[:,:,np.newaxis]
    # print flow_features.shape

    out_attention = np.matmul(attention_weight,flow_features).squeeze()
    out_attention = out_attention+attention_bias
    out_attention = scipy.special.softmax(out_attention)
    # print out_attention.shape

    # raw_input()
    # out_file = os.path.join(out_dir_viz, vid_name_only+'.jpg')
    xAndYs = [(range(flow_features.shape[0]),out_attention)]
    visualize.plotSimple(xAndYs,out_file=out_file,title='Attention',xlabel='Time Segment',ylabel='Confidence')

def main():
# caffe.set_device(0)
    caffe.set_mode_cpu()

    # net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
                    # 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    # caffe.TEST)
    out_dir_viz = '../scratch/untf_viz_spatial'
    util.mkdir(out_dir_viz)

    model_file = 'models/thumos14_spatial_untrimmednet_soft_bn_inception.caffemodel'
    prototxt = 'thumos_proto/spatial_untrimmednet_soft_bn_inception_deploy.prototxt'
    # prototxt = 'models/temporal_untrimmednet_soft_bn_inception_train_val.prototxt'

    net = caffe.Net(prototxt, model_file, caffe.TEST)

    print net.params['fc-action'][0].data.shape
    print net.params['fc-action'][1].data.shape

    attention_weight = np.array(net.params['fc-attention'][0].data)
    attention_bias = np.array(net.params['fc-attention'][1].data)

    print attention_weight.shape
    print attention_bias.shape

    data_dir = '../data/untf/test'
    vid_names = glob.glob(os.path.join(data_dir,'*_test_*.pickle'))

    args = []
    for idx_vid_name, vid_name in enumerate(vid_names):
        vid_name_only = os.path.split(vid_name)[1]
        vid_name_only = vid_name_only[:vid_name_only.rindex('.')]
    
        out_file = os.path.join(out_dir_viz, vid_name_only+'.jpg')
        if os.path.exists(out_file):
            continue

        arg_curr = (attention_weight,attention_bias,vid_name,out_file,idx_vid_name)
        args.append(arg_curr)
        # if idx_vid_name%10==0:
        #   print idx_vid_name, len(vid_names)
    
    print len(args)
    pool = multiprocessing.Pool()
    pool.map(save_im,args)
    pool.close()
    pool.join()

    visualize.writeHTMLForFolder(out_dir_viz)

    
if __name__=='__main__':
    main()

    
#   attention_arr = []
#   action_arr = []

#   for flow_curr in flow_features.T:
#       flow_curr = flow_curr[np.newaxis,:,np.newaxis,np.newaxis]
#       outputs = net.forward(start='global_pool')
#       attention_arr.append(outputs['fc-attention'].squeeze())
#       action_arr.append(outputs['fc-action'].squeeze())
    
#   attention_arr = np.array(attention_arr)
#   action_arr = np.array(action_arr)
    
    
# # fc-action: InnerProduct     (2 blobs)
# # fc-attention: InnerProduct     (2 blobs)


# # print "Network layers:"
# # for name, layer in zip(net._layer_names, net.layers):
# #     print "{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs))

# # print net.keys()

# # layer_input = np.zeros(10,1,1024)
# # vid_name = 'video_test_0000004.pickle'


