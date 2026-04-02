import os, logging, warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
os.environ.update({"TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0"})
import torch
import numpy as np
import random
import argparse
import json
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run

def prepare_1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')#MF,"BaseSpace"
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ratio', default='[0.8, 0.2]')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--la_lr', type=float, default=0.01)
    parser.add_argument('--diff_lr', type=float, default=0.0002)

    parser.add_argument('--root', default='./')
    parser.add_argument('--exp_part', default='None_CDR')
    parser.add_argument('--save_path', default='./model_save_default/model.pth')
    parser.add_argument('--use_cuda', default=1)
    
    parser.add_argument('--log_file', type=str, default=None, help='log file path, such as ./logs/experiment.log')
    parser.add_argument('--test_ratio', default=0.2)     

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    return args

def prepare_2(args,config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['base_model'] = args.base_model
    config['task'] = args.task
    config['ratio'] = args.ratio
    config['epoch'] = args.epoch
    config['lr'] = args.lr
    config['la_lr'] = args.la_lr
    config['diff_lr'] = args.diff_lr
    config['exp_part']   = args.exp_part
    config['save_path']  = args.save_path
    config['log_file'] = args.log_file
    config['test_ratio'] = args.test_ratio

    return config


if __name__ == '__main__':
    args = prepare_1()
    
    config_path = args.root + 'config.json'

    config = prepare_2(args,config_path)
    config['root'] = args.root + 'data/'
    config['use_cuda'] = 0 if args.use_cuda =='0' else 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if config.get('log_file'):
        log_dir = os.path.dirname(config['log_file'])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    if args.process_data_mid:
        for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
            DataPreprocessingMid(config['root'], dealing).main()
    if args.process_data_ready:
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
            for task in ['1', '2', '3']:
                DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()
    print('task:{}; model:{}; ratio:{}; epoch:{}; gpu:{}; seed:{};'.
          format(args.task, args.base_model, args.ratio, args.epoch, args.gpu, args.seed))
    print('diff_steps:{};diff_sample_steps:{};diff_dim:{};'.
          format(config['diff_steps'],config['diff_sample_steps'],config['diff_dim']))

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main(args.exp_part,args.save_path)



