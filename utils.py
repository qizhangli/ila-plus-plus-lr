import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from dataset import SelectedImagenet
from models import resnet


def build_dataset(args):
    img_transform = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor()
        ])
    dataset = SelectedImagenet(imagenet_val_dir=args.data_dir,
                               selected_images_csv=args.data_info_dir,
                               transform=img_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory = True)
    return data_loader


def build_model(args, device):
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225))
    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load(os.path.join(args.source_model_dir, 'resnet50-19c8e357.pth'), map_location='cpu'))
    model.eval()
    model = nn.Sequential(normalize, model)
    model.to(device)
    return model
