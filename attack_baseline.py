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

from attacks.linbp import linbp_attack
from attacks.pgd import pgd_attack
from models import resnet
from utils import build_dataset, build_model

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=8)
parser.add_argument('--niters', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--force', default=False, action="store_true")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--constraint', type=str, default="linf", choices=["linf", "l2"])
parser.add_argument('--method', type=str, default=None, choices=["IFGSM", "PGD", "LinBP"])
parser.add_argument('--linbp-layer', type=str, default="3_3")
parser.add_argument('--data-dir', type=int, default=None)
parser.add_argument('--data-info-dir', type=int, default=None)
parser.add_argument('--source-model-dir', type=int, default=None)
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
        ori_img, label = ori_img.to(device), label.to(device)
        if args.method == "IFGSM":
            img_ls, loss_ls = pgd_attack(args, ori_img, label, model, restart=0)
        elif args.method == "PGD":
            img_ls, loss_ls = pgd_attack(args, ori_img, label, model, restart=args.restart)
        elif args.method == "LinBP":
            img_ls, loss_ls = linbp_attack(args, ori_img, label, model, restart=args.restart)
        else:
            raise ValueError("invalid args.method.")
        np.save(os.path.join(args.save_dir, 'history_img_batch_{}.npy'.format(ind)), img_ls)
        np.save(os.path.join(args.save_dir, 'loss_batch_{}.npy'.format(ind)), loss_ls)
        # For args.restart > 1, save the last adversarial examples as default.
        np.save(os.path.join(args.save_dir, 'batch_{}.npy'.format(ind)), img_ls[-1, -1])
        print(' batch_{}.npy saved'.format(ind))
    label_ls = torch.cat(label_ls)
    np.save(os.path.join(args.save_dir, 'labels.npy'), label_ls.numpy())
    print('images saved')


if __name__ == '__main__':
    main()
