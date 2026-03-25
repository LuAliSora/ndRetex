import argparse

import datetime
# import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from pathlib import Path

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init

from modules.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from modules.dataPr import ImgSet
from modules.train import uvRex_train_one_epoch

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'eval'],
        default='train',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )
    parser.add_argument(
        "--DA",
        type=bool,
        default=True,
        help="Data Augmentation"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=['vgg', 'resnet50'],
        default='vgg'
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default='weights'
    )
    parser.add_argument(
        "--Freeze_Train",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--Init_Epoch",
        type=int,
        default=0
    )
    parser.add_argument(
        "--epoch_sum",
        type=int,
        default=0
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    # print(parser.parse_args())
    return parser.parse_args()


def get_model(backbone, pretrained, model_dir:str, Init_Epoch, device):
    model = Unet(2, backbone, pretrained, model_dir)

    if pretrained:
        return model

    if Init_Epoch==0:
        weights_init(model)
        return model
    
    # model_dir=Path("weights")
    model_path=f"{model_dir}/uvRex_{backbone}_epoch{Init_Epoch}.pth"

    print(f'Load weights {model_path}.')

    model_dict      = model.state_dict()
    pretrained_dict = torch.load(str(model_path), map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    return model


def train_main(seed, backbone, pretrained, model_dir:str, Freeze_Train, batch_size, Init_Epoch, epoch_sum, device):  
    if Init_Epoch<0 or Init_Epoch>epoch_sum:
        raise Exception("Require valid epoch!")
    
    local_rank      = 0
    rank            = 0

    seed_everything(seed)

    save_dir            = 'logs'

    Init_lr         = 1e-4
    Min_lr          = Init_lr * 0.01
    nbs             = 16
    lr_limit_max    = 1e-4 
    lr_limit_min    = 1e-4 
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    momentum            = 0.9
    weight_decay        = 0

    lr_decay_type       = 'cos'

    data_dir=Path("data")
    trainData_dir=data_dir/"train"
    testData_dir=data_dir/"test"

    train_dataset   = ImgSet(trainData_dir)
    test_dataset     = ImgSet(testData_dir, False)

    train_loader=DataLoader(train_dataset, 
                            batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                            drop_last=True
                            )
    test_loader=DataLoader(train_dataset, 
                            batch_size, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                            drop_last=True
                            )
    
    train_len=len(train_dataset)
    test_len=len(test_dataset)
    test_per_epochs=train_len // test_len

    model=get_model(backbone, pretrained, model_dir, Init_Epoch, device)
    model.train()
    if Freeze_Train:
        model.freeze_backbone()

    scaler = GradScaler()

    optimizer=optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay)

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epoch_sum)

    for epoch in range(Init_Epoch, epoch_sum):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        if (epoch+1) % test_per_epochs==0:
            uvRex_train_one_epoch(model, optimizer, scaler, device, train_loader, test_loader)
        else:
            uvRex_train_one_epoch(model, optimizer, scaler, device, train_loader)


if __name__ == "__main__":
    args=get_args()

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode=='train':
        train_main(args.seed, 
                   args.backbone, 
                   args.pretrained, 
                   args.model_dir,
                   args.Freeze_Train, 
                   args.batch_size, 
                   args.Init_Epoch, 
                   args.epoch_sum, 
                   device
                   )
