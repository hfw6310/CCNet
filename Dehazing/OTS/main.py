import os
import torch
import argparse
from torch.backends import cudnn
from models.SANet import build_net
from train import _train
from eval import _eval

import time

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.base_channel)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='SANet',type=str)
    parser.add_argument('--data_dir', type=str, default='/home/hfw/data/Dehazing/outdoor/datasets/reside-outdoor')

    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    
    # data
    parser.add_argument('--RandomCropSize', type=int, default=256)

    # Train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=30) # #
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')


    # Test
    parser.add_argument('--test_model', type=str, default='results/SANet/OTS/model.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    # network
    parser.add_argument('--base_channel', type=int, default=40)
    
    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', 'SANet', 'OTS/')

    ### add for data saving
    date = time.strftime("%Y-%m-%d")
    args.model_save_dir = args.model_save_dir + date
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    args.result_dir = os.path.join('results/', args.model_name, 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/SANet.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    print(args)
    main(args)
