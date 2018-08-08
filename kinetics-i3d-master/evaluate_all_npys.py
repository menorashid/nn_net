# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""
from __future__ import absolute_import
from __future__ import division

import sys
import os
import glob
sys.path.append('../code/data_processors')
sys.path.append('../code')
import preprocess_ucf as pu
from helpers import util


# from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

_IMAGE_SIZE = 224

_CHECKPOINT_PATHS = {
        'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
        'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
        'flow': 'data/checkpoints/flow_scratch/model.ckpt',
        'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
        'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'


def get_nets(eval_type,imagenet_pretrained, batchsize):
    NUM_CLASSES = 400
    if eval_type == 'rgb600':
        NUM_CLASSES = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
                tf.float32,
                shape=(1, batchsize, _IMAGE_SIZE, _IMAGE_SIZE, 3))


        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                    NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, rgb_endpoints = rgb_model(
                    rgb_input, is_training=False, dropout_keep_prob=1.0)


        rgb_variable_map = {}
        for variable in tf.global_variables():

            if variable.name.split('/')[0] == 'RGB':
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    rgb_variable_map[variable.name.replace(':0', '')] = variable

        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
                tf.float32,
                shape=(1, batchsize, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
                    NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, flow_endpoints = flow_model(
                    flow_input, is_training=False, dropout_keep_prob=1.0)
    
    flow_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow':
            flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    return rgb_saver, flow_saver, rgb_endpoints, flow_endpoints, rgb_input, flow_input


def get_video_names(out_dir, batchsize):
    dir_meta = '../data/ucf101'
    # out_dir = os.path.join(dir_meta,'npys')
    # util.mkdir(out_dir)

    dir_rgb = os.path.join(dir_meta, 'rgb_ziss/jpegs_256')

    dir_flos = os.path.join(dir_meta,'flow_ziss/tvl1_flow')
    dir_flos = [os.path.join(dir_flos,'u'),os.path.join(dir_flos,'v')]

    videos = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(dir_rgb,'*')) if os.path.isdir(dir_curr)]
    print len(videos)
    
    print len(videos)
    videos = [video for video in videos if not os.path.exists(os.path.join(out_dir,video+'.npy'))]
    print len(videos)
    mid_point = len(videos)//2
    print mid_point

    # videos = videos[:mid_point]
    # videos = videos[mid_point:]
    print len(videos)
    
    for video_curr in videos:
        out_file = os.path.join(out_dir,video_curr+'.npy')
        if os.path.exists(out_file):
            continue

        rgb_data,flo_data = pu.get_numpys(video_curr, dir_rgb, dir_flos, 224)

        n = rgb_data.shape[1]
        count = n//batchsize
        # print n
        if n%batchsize:
            n_new = count * batchsize
            rgb_data = rgb_data[:,:n_new,:,:]
            flo_data = flo_data[:,:n_new,:,:]


        # print ('rgb_data.shape',rgb_data.shape)
        # print ('flo_data.shape',rgb_data.shape)
        
        #break it up
        rgb_data = np.array_split(rgb_data, count, axis=1)
        flo_data = np.array_split(flo_data, count, axis=1)
        assert rgb_data[-1].shape[1] == batchsize
        assert rgb_data[0].shape[1] == batchsize
        
        # print np.min(flo_data[0]), np.max(flo_data[0])
        # print np.min(rgb_data[0]), np.max(rgb_data[0])

        # print ('len(rgb_data)',len(rgb_data))
        
        out_arr = np.zeros((2,len(rgb_data),1024))
        # print (out_arr.shape)
        yield rgb_data, flo_data, out_arr, out_file

        # raw_input()


def main(unused_argv):

    
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = 'joint'
    imagenet_pretrained = True
    batchsize = 16
    dir_meta = '../data/ucf101'
    in_dir = os.path.join('/disk2/maheen-data/ucf101','npys')
    out_dir = os.path.join(dir_meta,'features_noaug_16_16')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    # get_video_names(out_dir,batchsize)
    # return
    rgb_saver, flow_saver, rgb_endpoints, flow_endpoints,  rgb_input, flow_input = get_nets(eval_type, imagenet_pretrained, batchsize)
    sample_paths = glob.glob(os.path.join(in_dir,'*.npz'))

    with tf.Session() as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')

            if eval_type in ['flow', 'joint']:
                if imagenet_pretrained:
                    flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
                else:
                    flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
                tf.logging.info('Flow checkpoint restored')


            for rgb_data, flo_data, out_arr, out_file in get_video_names(out_dir, batchsize):
                print out_file

                for idx_sample, (rgb_sample, flow_sample) in enumerate(zip(rgb_data,flo_data)):
                    feed_dict[rgb_input] = rgb_sample
                    feed_dict[flow_input] = flow_sample
                    rgb_units = sess.run(rgb_endpoints['Logits'], feed_dict=feed_dict)
                    flo_units = sess.run(flow_endpoints['Logits'], feed_dict=feed_dict)

                    # print (rgb_units.shape)
                    # print (flo_units.shape)
                    
                    # print out_arr.shape

                    out_arr[0,idx_sample,:] = rgb_units.squeeze()
                    out_arr[1,idx_sample,:] = flo_units.squeeze()
                    # print np.min(out_arr[0],1)
                    # print np.max(out_arr[0],1)

                    # print np.min(out_arr[1],1)
                    # print np.max(out_arr[1],1)
                np.save(out_file, out_arr)
                # raw_input()
                    
        # flo_units = sess.run(flow_endpoints['Conv3d_1a_7x7'], feed_dict=feed_dict)
        # print (flo_units.shape)

        # feed_dict[rgb_input]

        # out_logits, out_predictions = sess.run(
        #     [model_logits, model_predictions],
        #     feed_dict=feed_dict)

        # out_logits = out_logits[0]
        # out_predictions = out_predictions[0]
        # sorted_indices = np.argsort(out_predictions)[::-1]

        # print('Norm of logits: %f' % np.linalg.norm(out_logits))
        # print('\nTop classes and probabilities')
        # for index in sorted_indices[:20]:
        #   print(out_predictions[index], out_logits[index], kinetics_classes[index])


# rgb_logits, all_endpoints = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0) units = sess.run(all_endpoints['Conv3d_1a_7x7'], feed_dict=feed_dict)

if __name__ == '__main__':
    # with tf.device("/GPU:0"):
    # from tensorflow.python.client import device_lib
    # device_lib.list_local_devices()
    tf.app.run(main)
