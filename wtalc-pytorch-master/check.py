from __future__ import print_function
import argparse
import os
import torch
from model import Model
from video_dataset import Dataset
from test import test, testTrain
from train import train, train_debug
from tensorboard_logger import Logger
import options
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim
import numpy as np

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    seed = 999
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    
    log_dir = 'temp/logs'
    out_dir = 'temp'
    # out_dir = args.out_dir
    # log_dir = os.path.join(out_dir,'logs')

    print (out_dir, log_dir, args.Lambda)

    # log_dir = os.path.join(log_dir, args.model_name)
    dataset = Dataset(args)
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    if not os.path.exists(log_dir):
       os.makedirs(log_dir)
    logger = Logger(log_dir)
    
    model = Model(dataset.feature_size, dataset.num_class).to(device)

    if args.pretrained_ckpt is not None:
       model.load_state_dict(torch.load(args.pretrained_ckpt))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
    for itr in range(args.max_iter):
        # train_debug(itr, dataset, args, model, optimizer, logger, device)
    #   if  itr % args.test_after == 0 and not itr == 0:
        test(itr, dataset, args, model, logger, device)

    #   if  itr % args.save_after == 0 and not itr == 0:
    #     torch.save(model.state_dict(), os.path.join(out_dir, args.model_name + '_'+str(itr)+'.pkl'))
    #     # test(itr, dataset, args, model, logger, device)
    
    # torch.save(model.state_dict(), os.path.join(out_dir, args.model_name + '_'+str(itr)+'.pkl'))
    # test(itr, dataset, args, model, logger, device)
