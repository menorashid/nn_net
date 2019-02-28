import numpy as np
import glob
import utils
import time

class Dataset():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.path_to_features = self.dataset_name + '-I3D-JOINTFeatures.npy'
        self.path_to_annotations = self.dataset_name + '-Annotations/'
        self.features = np.load(self.path_to_features, encoding='bytes')
        self.segments = np.load(self.path_to_annotations + 'segments.npy')
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy')     # Specific to Thumos14
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy')
        self.subset = np.load(self.path_to_annotations + 'subset.npy')

        self.videoname = np.load(self.path_to_annotations + '/videoname.npy'); 
        self.videoname = np.array([v.decode('utf-8') for v in self.videoname])
        # print (self.videoname.shape)
        # print (self.labels.shape)
        # input()
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.currenttrainidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs,self.classlist) for labs in self.labels]

        self.train_test_idx()
        self.classwise_feature_mapping()


    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == 'validation':   # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i); break;
            self.classwiseidx.append(idx)


    def load_test_data(self):
        # if train_test ==True:
        features = []
        labels = []
        idx = []

        # Load similar pairs
        # rand_classid = np.random.choice(len(self.classwiseidx), size=n_similar)
        # for rid in rand_classid:
        #     rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)
        #     idx.append(self.classwiseidx[rid][rand_sampleid[0]])
        #     idx.append(self.classwiseidx[rid][rand_sampleid[1]])

        # Load rest pairs
        rand_sampleid = np.random.choice(len(self.testidx), size=self.batch_size)
        for r in rand_sampleid:
            idx.append(self.testidx[r])
      
        return np.array([utils.process_feat(self.features[i], self.t_max) for i in idx]), np.array([self.labels_multihot[i] for i in idx])

    def load_train_data_for_test(self):
        

        # if is_training==True:
        #     features = []
        #     labels = []
        #     idx = []

        #     # Load similar pairs
        #     rand_classid = np.random.choice(len(self.classwiseidx), size=n_similar)
        #     for rid in rand_classid:
        #         rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)
        #         idx.append(self.classwiseidx[rid][rand_sampleid[0]])
        #         idx.append(self.classwiseidx[rid][rand_sampleid[1]])

        #     # Load rest pairs
        #     rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size-2*n_similar)
        #     for r in rand_sampleid:
        #         idx.append(self.trainidx[r])
          
        #     return np.array([utils.process_feat(self.features[i], self.t_max) for i in idx]), np.array([self.labels_multihot[i] for i in idx])
        # else:
        labs = self.labels_multihot[self.trainidx[self.currenttrainidx]]
        feat = self.features[self.trainidx[self.currenttrainidx]]
        vid_name = self.videoname[self.trainidx[self.currenttrainidx]]

        if self.currenttrainidx == len(self.trainidx)-1:
            done = True; self.currenttrainidx = 0
        else:
            done = False; self.currenttrainidx += 1
     
        return np.array(feat), np.array(labs), vid_name, done

    def load_data(self, n_similar=3, is_training=True):
        

        if is_training==True:
            features = []
            labels = []
            idx = []

            # Load similar pairs
            rand_classid = np.random.choice(len(self.classwiseidx), size=n_similar)
            for rid in rand_classid:
                rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)
                idx.append(self.classwiseidx[rid][rand_sampleid[0]])
                idx.append(self.classwiseidx[rid][rand_sampleid[1]])

            # Load rest pairs
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size-2*n_similar)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])
          
            return np.array([utils.process_feat(self.features[i], self.t_max) for i in idx]), np.array([self.labels_multihot[i] for i in idx])
        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            vid_name = self.videoname[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return np.array(feat), np.array(labs), vid_name, done

