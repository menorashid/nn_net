import argparse

parser = argparse.ArgumentParser(description='WTALC')
parser.add_argument('--lr', type=float, default=0.001,help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', default=20, help='number of classes (default: 20)')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--max-seqlen', default=750, help='maximum sequence length during training (default: 750)')
parser.add_argument('--Lambda', type = float, default=0.5, help='weight on Co-Activity Loss (default: 0.5)')
parser.add_argument('--num-similar', default=3, help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--max-grad-norm', type=float, default=10, help='value loss coefficient (default: 10)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=5000, help='maximum iteration to train (default: 10000)')
parser.add_argument('--test_after', type=int, default=500, help='iterations to test after (default: 100)')
parser.add_argument('--save_after', type=int, default=500, help='iterations to test after (default: 100)')
parser.add_argument('--out_dir',type = str, help='directory to save output')
parser.add_argument('--loss_type',type = str, default = 'regular',help='loss type')


