import numpy as np
import torch
import torch.nn.functional as F

from .helper import random_start_function, to_np_uint8, update_and_clip


def pgd_attack(args, ori_img, label, model, restart):
    img_ls, loss_ls = [], []
    for restart_ind in range(max(restart, 1)):
        img_re_ls, loss_re_ls = [], []
        if restart > 0:
            att_img = random_start_function(ori_img, args.epsilon)
        elif restart == 0:
            att_img = ori_img.clone()
        else:
            raise ValueError("invalid argument restart.")
        for i in range(args.niters+1):
            att_img.requires_grad_(True)
            att_output = model(att_img)
            loss_ = F.cross_entropy(att_output, label, reduction="none")
            loss = loss_.mean()
            img_re_ls.append(to_np_uint8(att_img.data.clone()))
            loss_re_ls.append(loss_.view(-1).data.clone())
            if i == args.niters:
                break
            model.zero_grad()
            loss.backward()
            input_grad = att_img.grad.data
            model.zero_grad()
            att_img = update_and_clip(ori_img, att_img, input_grad, args.epsilon, args.step_size, args.constraint)
            print('\r restart {}, iter {}'.format(restart_ind, i), end='')
        img_ls.append(np.stack(img_re_ls))
        loss_ls.append(torch.stack(loss_re_ls).cpu().numpy())
    img_ls = np.stack(img_ls)
    loss_ls = np.stack(loss_ls)
    return img_ls, loss_ls
