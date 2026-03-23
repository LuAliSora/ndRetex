import argparse

import datetime
import os
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

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--if_train",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=['vgg', 'resnet50'],
        default='vgg'
    )
    parser.add_argument(
        "--if_pretrained",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "init_epoch",
        type=int
    )
    parser.add_argument(
        "epoch_sum",
        type=int
    )
    # print(parser.parse_args())
    return parser.parse_args()


def get_model(backbone, pretrained, init_epoch, device):
    model = Unet(2, backbone, pretrained)

    if pretrained:
        return model

    if init_epoch==0:
        weights_init(model)
        return model
    
    model_dir=Path("weights")
    model_path=model_dir/f"unet_{backbone}_epoch{init_epoch}.pth"

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


def train_main(backbone, pretrained, batch_size, init_epoch, epochSum, device):  
    local_rank      = 0
    rank            = 0

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


    model=get_model(backbone, pretrained, init_epoch, device)
    model.train()

    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    # loss_history    = LossHistory(log_dir, model, input_shape=input_shape)

    scaler = GradScaler()

    optimizer=optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay)

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochSum)



    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )


    


if __name__ == "__main__":
    args=get_args()

    if (args.epoch_sum)<=(args.init_epoch):
        raise Exception("epoch_sum should greater than init_epoch")
    
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.if_train:
        train_main(args.backbone, 
                   args.if_pretrained, 
                   args.batch_size, 
                   args.init_epoch, 
                   args.epoch_sum, 
                   device
                   )