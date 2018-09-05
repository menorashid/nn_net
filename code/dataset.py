import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from helpers import util, visualize
import models
from criterions import MultiCrossEntropy
import random

class UCF_dataset(Dataset):
    def __init__(self, text_file, feature_limit):
        self.files = util.readLinesFromFile(text_file)
        self.feature_limit = feature_limit

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        anno = train_file_curr.split(' ')
        label = anno[1:]
        train_file_curr = anno[0]

        label =np.array([int(label) for label in label]).astype(float)
        label = label/np.sum(label)

        sample = np.load(train_file_curr)
        
        if sample.shape[0]>self.feature_limit and self.feature_limit is not None:
            idx_start = sample.shape[0] - self.feature_limit
            idx_start = np.random.randint(idx_start+1)
            sample = sample[idx_start:idx_start+self.feature_limit]
            assert sample.shape[0]==self.feature_limit

        # image = Image.open(train_file_curr)
        sample = {'features': sample, 'label': label}
        # if self.transform:
        # sample['image'] = self.transform(sample['image'])

        return sample

    def collate_fn(self,batch):
        data = [torch.FloatTensor(item['features']) for item in batch]
        target = [item['label'] for item in batch]
        target = torch.FloatTensor(target)
        return {'features':data, 'label':target}


def main():
    print 'hello'

    train_file = '../data/ucf101/train_test_files/train.txt'
    limit = 500
    model_name = 'just_mill'
    network_params = dict(n_classes=20, deno = 8, init=False )

    criterion = MultiCrossEntropy().cuda()
    
    train_data = UCF_dataset(train_file, limit)
    train_dataloader = DataLoader(train_data, collate_fn = train_data.collate_fn,
                        batch_size=10,
                        shuffle=False)
    network = models.get(model_name,network_params)
    model = network.model
    model = model.cuda()
    # net = models.Network(n_classes= 20, deno = 8)
    # print net.model
    # net.model = net.model.cuda()
    # input = np.zeros((32,2048))
    # input = torch.Tensor(input).cuda()
    # input = Variable(input)
    # output, pmf = net.model(input)
    
    optimizer = torch.optim.Adam(network.get_lr_list([1e-6]),weight_decay=0)
    print len(train_dataloader)
    
    # exit = True

    for num_epoch in range(500):

        labels = []
        preds = []

        for num_iter, train_batch in enumerate(train_dataloader):
            # print num_iter
            sample = train_batch['features']
            # [0].cuda()
            label = train_batch['label'].cuda()

            print label.size()
            print len(sample)
            print sample[0].size()

            # print labels.size()
            raw_input()

            out,pmf = model.forward(sample)
            preds.append(pmf.unsqueeze(0))
            labels.append(label)


        preds = torch.cat(preds,0)
        labels = torch.cat(labels,0)
        loss = criterion(labels, preds)
        # raw_input()
            # print pmf.size()
            
        optimizer.zero_grad()

        # loss = model.multi_ce(labels, pmf)

        loss.backward()
        optimizer.step()
            

        loss_val = loss.data[0].cpu().numpy()
        
        labels = labels.data.cpu().numpy()
        preds = torch.nn.functional.softmax(preds).data.cpu().numpy()
        
        # ,np.argmax(preds,axis=1)
        accu =  np.sum(np.argmax(labels,axis=1)==np.argmax(preds,axis=1))/float(labels.shape[0])
        print num_epoch, loss_val, accu
            # print torch.nn.functional.softmax(preds).data.cpu().numpy()




if __name__=='__main__':
    main()