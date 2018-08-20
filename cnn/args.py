import sys
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# args for optimizee
parser.add_argument('--model_grad_norm', type=float, default=1.0, help='max grad norm for model')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--alpha_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch', type=str, default='simple', help='which architecture to use')
parser.add_argument('--train_alpha', type=str2bool, default=True, help='whether to train alpha')
parser.add_argument('--load_alpha', type=str2bool, default=False, help='whether to load alpha')

# add by L2-ml
# args for training in general
parser.add_argument('--epoch', type=int, default=3, help='num of epoch for each training iteration and testing')
parser.add_argument('--test_beta_epoch', type=int, default=50, help='num of epoch for test trained beta')
parser.add_argument('--test_freq', type=int, default=10, help='frequency in epoch for running testing while training')
parser.add_argument('--save_dir', type=str, default='', help='folder path to save trained model & meta optim and logs')
parser.add_argument('--log_dir', type=str, default='logs', help='dir path for logs')
parser.add_argument('--log_freq', type=int, default=50, help='logging frequency')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--gpu', type=int, default=0, help='which GPU to use')
parser.add_argument('--seed', type=int, default=997, help='set the random seed')
parser.add_argument('--max_episodes', type=int, default=3, help='max episodes of training')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size for training')
parser.add_argument('--continue_checkpoint', type=str2bool, default=False, help='whether to continue from checkpoint')
parser.add_argument('--checkpoint_path', type=str, default='', help='folder path to load saved model & meta optim')

# args for meta optimizer
parser.add_argument('--beta_entropy_penalty', type=float, default=0.1, help='ratio of entropy loss / normal loss')
parser.add_argument('--beta_weight_decay', type=float, default=0.0, help='weight decay for optim arch encoding')
parser.add_argument('--beta_learning_rate', type=float, default=0.002, help='learning rate for arch encoding')
parser.add_argument('--beta_grad_norm', type=float, default=1.0, help='max grad norm for beta')
parser.add_argument('--normalize_bptt', type=bool, default=True, help='Noramlize the gradient of beta by bptt steps?')
parser.add_argument('--use_darts_arch', action='store_true', default=False, help='use darts architecture')
parser.add_argument('--bptt_step', type=int, default=1, help='steps for bptt')
parser.add_argument('--max_beta_scaling', type=float, default=100, help='min learning rate')
parser.add_argument('--min_beta_scaling', type=float, default=100, help='min learning rate')
parser.add_argument('--use_traditional_opt', action='store_true', default=False, help='use traditional optimizer')
parser.add_argument('--train_beta', type=str2bool, default=True, help='whether to train beta')
parser.add_argument('--load_beta', type=str2bool, default=False, help='whether to load beta')

args = parser.parse_args()
