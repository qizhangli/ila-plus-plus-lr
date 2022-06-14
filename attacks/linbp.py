import numpy as np
import torch
import torch.nn.functional as F

from .helper import random_start_function, to_np_uint8, update_and_clip


def linbp_forw_resnet50(model, x, do_linbp, linbp_layer):
    layer_ind = int(linbp_layer.split('_')[0])
    block_ind = int(linbp_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = model[1].maxpool(x)
    ori_mask_ls = []
    conv_out_ls = []
    relu_out_ls = []
    conv_input_ls = []
    def layer_forw(layer_ind, block_ind, layer_ind_cur, block_ind_cur, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp):
        if layer_ind < layer_ind_cur:
            x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
            ori_mask_ls.append(ori_mask)
            conv_out_ls.append(conv_out)
            relu_out_ls.append(relu_out)
            conv_input_ls.append(conv_in)
        elif layer_ind == layer_ind_cur:
            if block_ind_cur >= block_ind:
                x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            else:
                x, _, _, _, _ = block_func(mm, x, linbp=False)
        else:
            x, _, _, _, _ = block_func(mm, x, linbp=False)
        return x, ori_mask_ls
    for ind, mm in enumerate(model[1].layer1):
        x, ori_mask_ls = layer_forw(layer_ind, block_ind, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer2):
        x, ori_mask_ls = layer_forw(layer_ind, block_ind, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer3):
        x, ori_mask_ls = layer_forw(layer_ind, block_ind, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer4):
        x, ori_mask_ls = layer_forw(layer_ind, block_ind, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return x, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls


def block_func(block, x, linbp):
    identity = x
    conv_in = x+0
    out = block.conv1(conv_in)
    out = block.bn1(out)
    out_0 = out + 0
    if linbp:
        out = linbp_relu(out_0)
    else:
        out = block.relu(out_0)
    ori_mask_0 = out.data.bool().int()
    out = block.conv2(out)
    out = block.bn2(out)
    out_1 = out + 0
    if linbp:
        out = linbp_relu(out_1)
    else:
        out = block.relu(out_1)
    ori_mask_1 = out.data.bool().int()
    out = block.conv3(out)
    out = block.bn3(out)
    if block.downsample is not None:
        identity = block.downsample(identity)
    identity_out = identity + 0
    x_out = out + 0
    out = identity_out + x_out
    out = block.relu(out)
    ori_mask_2 = out.data.bool().int()
    return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in)


def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x


def linbp_backw_resnet50(img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp):
    for i in range(-1, -len(conv_out_ls)-1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad((conv_out_ls[i+1][0], conv_input_ls[i+1][1]), conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(conv_out_ls[i][1], relu_out_ls[i][1], grads[1]*ori_mask_ls[i][2],retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(relu_out_ls[i][1], relu_out_ls[i][0], normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(relu_out_ls[i][0], conv_input_ls[i][1], normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim = (1,2,3), keepdim = True) / main_grad.norm(p=2,dim = (1,2,3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad((conv_out_ls[0][0], conv_input_ls[0][1]), img, grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data


def linbp_attack(args, ori_img, label, model, restart):
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
            att_output, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, att_img, True, args.linbp_layer)
            loss_ = F.cross_entropy(att_output, label, reduction="none")
            loss = loss_.mean()
            img_re_ls.append(to_np_uint8(att_img.data.clone()))
            loss_re_ls.append(loss_.view(-1).data.clone())
            if i == args.niters:
                break
            model.zero_grad()
            input_grad = linbp_backw_resnet50(att_img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=1.0)
            model.zero_grad()
            att_img = update_and_clip(ori_img, att_img, input_grad, args.epsilon, args.step_size, args.constraint)
            print('\r restart {}, iter {}'.format(restart_ind, i), end='')
        img_ls.append(np.stack(img_re_ls))
        loss_ls.append(torch.stack(loss_re_ls).cpu().numpy())
    img_ls = np.stack(img_ls)
    loss_ls = np.stack(loss_ls)
    return img_ls, loss_ls
