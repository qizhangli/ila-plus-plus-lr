import argparse
import logging
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tvF
from joblib import Parallel, delayed
from torch.backends import cudnn

import models

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--njobs', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--test-log', type=str, default=None)
parser.add_argument('--victim-dir', type=str, default=None)
args = parser.parse_args()


def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    if args.victim_dir[-1] != "/":
        args.victim_dir += "/"

    logging.basicConfig(filename=args.test_log, level=logging.INFO)
    logging.info(args)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def load_img(img, trans):
        return trans(tvF.to_pil_image(img))

    def test(model, trans):
        target = torch.from_numpy(np.load(args.dir + '/labels.npy')).long()
        img_num = 0
        count = 0
        dir_ls = os.listdir(args.dir)
        advfile_ls = []
        for file_name in dir_ls:
            if file_name[:5] == 'batch':
                advfile_ls.append(file_name)
        for advfile_ind in range(len(advfile_ls)):
            adv_batch = torch.from_numpy(np.load(args.dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
            batch_size = adv_batch.shape[0]
            img_ls = Parallel(n_jobs=args.njobs)(delayed(load_img)(adv_batch[img_ind].clone(), trans) for img_ind in range(batch_size))
            img = torch.stack(img_ls)
            img_num += img.shape[0]
            label = target[advfile_ind * batch_size : advfile_ind * batch_size + adv_batch.shape[0]]
            label = label.to(device)
            img = img.to(device)
            with torch.no_grad():
                pred = torch.argmax(model(img), dim=1).view(1,-1)
            count += (label != pred.squeeze(0)).sum().item()
            del pred, img
            del adv_batch
        return round(100. * count / img_num, 2)

    trans_ori = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trans_pnas = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.Resize((331, 331)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    trans_se = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trans_incep = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    # WHITE-BOX
    resnet50 = torchvision.models.resnet50()
    state_dict = torch.load(args.victim_dir + 'resnet50-19c8e357.pth', map_location='cpu')
    resnet50.load_state_dict(state_dict)
    resnet50.eval()
    resnet50.to(device)
    logging.info(('resnet50:', test(model = resnet50, trans = trans_ori)))
    del resnet50

    # BLACK-BOX
    avg_black_box_success_rate = []

    VGG19 = torchvision.models.vgg19_bn()
    state_dict = torch.load(args.victim_dir + 'vgg19_bn-c79401a0.pth', map_location = 'cpu')
    VGG19.load_state_dict(state_dict)
    VGG19.to(device)
    VGG19.eval()
    vgg_fr = test(model = VGG19, trans = trans_se)
    logging.info(('VGG19:', vgg_fr))
    avg_black_box_success_rate.append(vgg_fr)
    del VGG19

    resnet152 = torchvision.models.resnet152()
    state_dict = torch.load(args.victim_dir + 'resnet152-b121ed2d.pth', map_location = 'cpu')
    resnet152.load_state_dict(state_dict)
    resnet152.to(device)
    resnet152.eval()
    resnet152_fr = test(model = resnet152, trans = trans_se)
    logging.info(('resnet152:', resnet152_fr))
    avg_black_box_success_rate.append(resnet152_fr)
    del resnet152

    inceptionv3 = models.inceptionv3.Inception3()
    inceptionv3.to(device)
    inceptionv3.load_state_dict(torch.load(args.victim_dir + 'inception_v3_google-1a9a5a14.pth', map_location = 'cpu'))
    inceptionv3.eval()
    inceptionv3_fr = test(model = inceptionv3, trans = trans_incep)
    logging.info(('inceptionv3:', inceptionv3_fr))
    avg_black_box_success_rate.append(inceptionv3_fr)
    del inceptionv3

    densenet = torchvision.models.densenet121(pretrained=False)
    densenet.to(device)
    import re
    pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = torch.load(args.victim_dir + 'densenet121-a639ec97.pth', map_location = 'cpu')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    densenet.load_state_dict(state_dict)
    densenet.eval()
    densenet_fr = test(model = densenet, trans = trans_se)
    logging.info(('densenet:', densenet_fr))
    avg_black_box_success_rate.append(densenet_fr)
    del densenet

    mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
    mobilenet.to(device)
    mobilenet.load_state_dict(torch.load(args.victim_dir + 'mobilenet_v2-b0353104.pth', map_location = 'cpu'))
    mobilenet.eval()
    mobilenet_fr = test(model = mobilenet, trans = trans_se)
    logging.info(('mobilenet:', mobilenet_fr))
    avg_black_box_success_rate.append(mobilenet_fr)
    del mobilenet

    senet = models.senet.senet154(ckpt_dir =args.victim_dir + 'senet154-c7b49a05.pth')
    senet.to(device)
    senet.eval()
    senet_fr = test(model = senet, trans = trans_se)
    logging.info(('senet:', senet_fr))
    avg_black_box_success_rate.append(senet_fr)
    del senet

    resnext = torchvision.models.resnext101_32x8d()
    state_dict = torch.load(args.victim_dir + 'resnext101_32x8d-8ba56ff5.pth', map_location = 'cpu')
    resnext.load_state_dict(state_dict)
    resnext.to(device)
    resnext.eval()
    resnext_fr = test(model = resnext, trans = trans_se)
    logging.info(('resnext:', resnext_fr))
    avg_black_box_success_rate.append(resnext_fr)
    del resnext

    WRN = torchvision.models.wide_resnet101_2()
    state_dict = torch.load(args.victim_dir + 'wide_resnet101_2-32ee1156.pth', map_location = 'cpu')
    WRN.load_state_dict(state_dict)
    WRN.to(device)
    WRN.eval()
    wrn_fr = test(model = WRN, trans = trans_se)
    logging.info(('WRN:', wrn_fr))
    avg_black_box_success_rate.append(wrn_fr)
    del WRN

    pnasnet = models.pnasnet.pnasnet5large(ckpt_dir =args.victim_dir + 'pnasnet5large-bf079911.pth', num_classes=1000, pretrained='imagenet')
    pnasnet.to(device)
    pnasnet.eval()
    pnasnet_fr = test(model = pnasnet, trans = trans_pnas)
    logging.info(('pnasnet:', pnasnet_fr))
    avg_black_box_success_rate.append(pnasnet_fr)
    del pnasnet

    mnasnet = torchvision.models.mnasnet1_0()
    state_dict = torch.load(args.victim_dir + 'mnasnet1.0_top1_73.512-f206786ef8.pth', map_location = 'cpu')
    mnasnet.load_state_dict(state_dict)
    mnasnet.to(device)
    mnasnet.eval()
    mnas_fr = test(model = mnasnet, trans = trans_se)
    logging.info(('mnasnet:', mnas_fr))
    avg_black_box_success_rate.append(mnas_fr)
    del mnasnet

    logging.info(("Black-box AVG: ", round(sum(avg_black_box_success_rate) / len(avg_black_box_success_rate), 2)))


if __name__ == "__main__":
    main()
