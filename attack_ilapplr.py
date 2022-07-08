import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.backends import cudnn

from attacks.helper import to_np_uint8
from attacks.ilapplr import ilapplr_attack
from models import resnet
from utils import build_dataset, build_model

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=8)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--force', default=False, action="store_true")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--constraint', type=str, default="linf", choices=["linf", "l2"])
parser.add_argument('--history-dir', type=str, default=None)
parser.add_argument('--ila-layer', type=str, default='3_0')
parser.add_argument('--lr-method', type=str, default=None, choices=["RR", "SVR", "ElasticNet"])
parser.add_argument('--njobs', type=int, default = 50)
parser.add_argument('--random-start', default=False, action="store_true")
parser.add_argument('--data-dir', type=str, default=None)
parser.add_argument('--data-info-dir', type=str, default=None)
parser.add_argument('--source-model-dir', type=str, default=None)
args = parser.parse_args()


def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    if args.constraint == "linf":
        args.epsilon = args.epsilon / 255.
        args.step_size = 1 / 255.
    elif args.constraint == "l2":
        args.step_size = (args.epsilon / 5)
    else:
        raise ValueError("invalid args.constraint.")
    print(args)
    
    os.makedirs(args.save_dir, exist_ok=True if args.force else False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    data_loader = build_dataset(args)
    model = build_model(args, device)

    # ATTACK
    label_ls = []
    for ind, (ori_img, label) in enumerate(data_loader):
        label_ls.append(label)
        ori_img = ori_img.to(device)
        att_img = ilapplr_attack(args, ori_img, model, ind, device)
        np.save(os.path.join(args.save_dir, 'batch_{}.npy'.format(ind)), to_np_uint8(att_img))
        print(' batch_{}.npy saved'.format(ind))
    label_ls = torch.cat(label_ls)
    np.save(os.path.join(args.save_dir, 'labels.npy'), label_ls.numpy())
    print('images saved')
    

if __name__ == '__main__':
    main()
