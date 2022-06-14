import os

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import LinearSVR

from .helper import random_start_function, update_and_clip


def ila_forw_resnet50(model, x, ila_layer):
    layer_ind = int(ila_layer.split('_')[0])
    block_ind = int(ila_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    if layer_ind == 0 and block_ind ==0:
        return x
    x = model[1].maxpool(x)
    for block_ind_cur, mm in enumerate(model[1].layer1):
        x = mm(x)
        if layer_ind == 1 and block_ind_cur == block_ind:
            return x
    for block_ind_cur, mm in enumerate(model[1].layer2):
        x = mm(x)
        if layer_ind == 2 and block_ind_cur == block_ind:
            return x
    for block_ind_cur, mm in enumerate(model[1].layer3):
        x = mm(x)
        if layer_ind == 3 and block_ind_cur == block_ind:
            return x
    for block_ind_cur, mm in enumerate(model[1].layer4):
        x = mm(x)
        if layer_ind == 4 and block_ind_cur == block_ind:
            return x


class ProjLoss(torch.nn.Module):
    def __init__(self, w, mid_original, normalize_new_mid = False):
        super(ProjLoss, self).__init__()
        self.w = w
        self.mid_original = mid_original
        self.normalize_new_mid = normalize_new_mid
    def forward(self, mid_output):
        n = mid_output.shape[0]
        y = mid_output - self.mid_original
        if self.normalize_new_mid:
            y = y / (y.norm(p=2, dim=1, keepdim=True) + 1e-12)
        proj_loss = torch.sum(self.w * y) / n
        return proj_loss


def parallel_fit(diff, loss, i, lr_method):
    lr_method = lr_method.lower()
    if 'rr' in lr_method:
        if "-" in lr_method:
            lam2 = float(lr_method.split('-')[1])
        else:
            lam2 = 1e10
        clf = Ridge(alpha=lam2, fit_intercept=False, tol=1e-4)
    elif 'elasticnet' in lr_method:
        if "-" in lr_method:
            lam1, lam2 = list(map(float, lr_method.split('-')))
        else:
            lam1, lam2 = 0.05, 1.
        clf = ElasticNet(alpha=(lam1+2*lam2)/(2*diff.size(1)), l1_ratio=lam1/(lam1+2*lam2),
                         tol=1e-4,fit_intercept=False,selection='random')
    elif 'svr' in lr_method:
        if "-" in lr_method:
            C = float(lr_method.split('-')[1])
        else:
            C = 1e-10
        clf = LinearSVR(tol=1e-4, C=C)
    clf.fit(diff, loss)
    return torch.from_numpy(clf.coef_).float()


def get_diff(model, mid_original, img_history_ls, ila_layer, device, n_split=1):
    # Set n_split > 1 in case of "CUDA out of memory".
    baseline_nrestart, baseline_niters, baseline_batch_size = img_history_ls.shape[:3]
    diff_ls = []
    chunk_indices = torch.arange(baseline_nrestart * baseline_niters).split((baseline_nrestart * baseline_niters) // n_split)
    for ch_inds in chunk_indices:
        diff_ls.append((
            ila_forw_resnet50(
                model, 
                img_history_ls.view(-1, *img_history_ls.shape[2:])[ch_inds].view(-1, *img_history_ls.shape[3:]).to(device) / 255., 
                ila_layer).data.view(len(ch_inds), baseline_batch_size, -1) 
            - mid_original.data).cpu())
    diff_ls = torch.cat(diff_ls, dim=0).permute(1,0,2)
    return diff_ls


def ilapplr_attack(args, ori_img, model, batch_index, device):
    batch_size = ori_img.shape[0]
    img_history_ls = torch.from_numpy(np.load(os.path.join(args.history_dir, 'history_img_batch_{}.npy'.format(batch_index)))).float()
    loss_history_ls = torch.from_numpy(np.load(os.path.join(args.history_dir, 'loss_batch_{}.npy'.format(batch_index)))).float()
    baseline_nrestart, baseline_niters, baseline_batch_size = img_history_ls.shape[:3]
    assert batch_size == baseline_batch_size
    with torch.no_grad():
        mid_original = ila_forw_resnet50(model, ori_img, args.ila_layer).view(batch_size, -1)
        diff_ls = get_diff(model, mid_original, img_history_ls, args.ila_layer, device, 10)
    del img_history_ls
    loss_history_ls = loss_history_ls.view(baseline_nrestart*baseline_niters, batch_size).permute(1,0)
    w = Parallel(n_jobs=args.njobs)(delayed(parallel_fit)(diff_ls[f_i].clone(), loss_history_ls[f_i].clone(), f_i, args.lr_method) for f_i in range(batch_size))
    del diff_ls, loss_history_ls
    w = torch.stack(w)
    proj_loss = ProjLoss(w=w.to(device), mid_original = mid_original)
    if args.random_start:
        att_img = random_start_function(ori_img, args.epsilon)
    else:
        att_img = ori_img.clone()
    for i in range(args.niters):
        att_img.requires_grad_(True)
        mid_output = ila_forw_resnet50(model, att_img, args.ila_layer)
        loss = proj_loss(mid_output.view(batch_size, -1))
        model.zero_grad()
        loss.backward()
        input_grad = att_img.grad.data
        model.zero_grad()
        att_img = update_and_clip(ori_img, att_img, input_grad, args.epsilon, args.step_size, args.constraint)
        print('\r iter {}'.format(i), end='')
    del mid_output, mid_original
    return att_img
