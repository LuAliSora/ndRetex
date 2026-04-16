import argparse

# import datetime
# import os
from functools import partial

# import numpy as np
import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
# import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
import csv
import cv2

# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import ProjectConfiguration, set_seed

from modules.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn, uvRex_get_model)
from modules.dataPr import Rex_ImgSet
from modules.train import sd_train_one_epoch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'predict'],
        default='train',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="input",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
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
        default=1
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2
    )
    #uvrex_model
    parser.add_argument(
        "--uvrex_model_dir",
        type=str,
        default="weights",
    )
    parser.add_argument(
        "--uvrex_backbone",
        type=str,
        choices=['vgg', 'resnet50'],
        default='vgg'
    )
    parser.add_argument(
        "--uvrex_Epoch",
        type=int,
        default=99
    )
    #predict
    parser.add_argument(
        "--img",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--texture",
        type=str,
        default=None,
    )
    # print(parser.parse_args())
    return parser.parse_args()


def train_main(input_dir:str, uvRex_model, pretrained, Freeze_Train, batch_size, Init_Epoch, epoch_sum, device, seed): 
    if Init_Epoch<0 or Init_Epoch>epoch_sum:
        raise Exception("Require valid epoch!")
    
    seed_everything(seed)

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

    # myAcc = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.mixed_precision,
    #     log_with=args.report_to,
    #     project_config=accelerator_project_config,
    # )
    data_dir=Path(input_dir)
    train_dir=data_dir/"train"
    test_dir=data_dir/"test"

    train_dataset   = Rex_ImgSet(str(train_dir))
    test_dataset    = Rex_ImgSet(str(test_dir))

    train_loader=DataLoader(train_dataset, 
                            batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed),
                            drop_last=True,
                            # persistent_workers=True
                            )
    test_loader=DataLoader(test_dataset, 
                            batch_size, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed),
                            drop_last=True,
                            # persistent_workers=True
                            )
    
    train_len=len(train_dataset)
    test_len=len(test_dataset)
    test_per_epochs=train_len // test_len *10

    log_dir            = 'logs'
    loss_log_file=f"{log_dir}/sd_loss_{Init_Epoch}.csv"

    with open(loss_log_file, 'w', newline='') as f:
        loss_writer = csv.writer(f)
        loss_writer.writerow(['epoch', 'train_loss', 'test_loss'])

    epoch_range = range(Init_Epoch, epoch_sum)
    epoch_pbar = tqdm(epoch_range, desc='Training_Progress', unit='epoch')

    for epoch in epoch_pbar:
        if (epoch+1) % test_per_epochs==0:
            train_loss, test_loss = sd_train_one_epoch(uvRex_model, device, train_loader, test_loader)

            with open(loss_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{test_loss:.6f}"])

            # torch.save(model.state_dict(), f"{model_dir}/uvRex_{backbone}_epoch{epoch}.pth")

        else:
            train_loss, _ = sd_train_one_epoch(uvRex_model, device, train_loader)
            
            with open(loss_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", ""])


if __name__ == "__main__":
    args=get_args()

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    uvRex_model=uvRex_get_model(args.uvrex_backbone, False, args.uvrex_model_dir, args.uvrex_Epoch, device).to(device)
    uvRex_model.eval()

    if args.mode=='train':
        train_main(args.input_dir, 
                   uvRex_model, 
                   args.pretrained, 
                   args.Freeze_Train, 
                   args.batch_size, 
                   args.Init_Epoch, 
                   args.epoch_sum, 
                   device,
                   args.seed
                   )
    # else:
    #     predict_main(args.input_dir,
    #                  args.model_dir,
    #                  args.img, 
    #                  args.texture, 
    #                  args.backbone, 
    #                  args.Init_Epoch, 
    #                  device
    #                  )