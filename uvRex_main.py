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

# from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr

from modules.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn, uvRex_get_model)
from modules.dataPr import Masked_ImgSet, get_dataAug, img_masked, img2np_rgb, get_binary_mask, img2tensor_rgb

from modules.train import uvRex_train_one_epoch
from modules.predict import uvRex_predict


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
        "--model_dir",
        type=str,
        default="weights",
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


def train_main(input_dir:str, model_dir:str, backbone, pretrained, Freeze_Train, batch_size, Init_Epoch, epoch_sum, device, seed):  
    if Init_Epoch<0 or Init_Epoch>epoch_sum:
        raise Exception("Require valid epoch!")
    
    local_rank      = 0
    rank            = 0

    seed_everything(seed)

    # input_dir=Path("input")
    data_dir=Path(input_dir)
    train_dir=data_dir/"train"
    test_dir=data_dir/"test"
    img_folder="normal"
    mask_folder="mask"

    train_dataset   = Masked_ImgSet(str(train_dir/img_folder), str(train_dir/mask_folder))
    test_dataset    = Masked_ImgSet(str(test_dir/img_folder), str(test_dir/mask_folder))

    trans_size=[512, 512]  
    dataAug=get_dataAug(trans_size)
    # dataAug.to(device)

    train_loader=DataLoader(train_dataset, 
                            batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                            drop_last=True,
                            # persistent_workers=True
                            )
    test_loader=DataLoader(test_dataset, 
                            batch_size, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                            drop_last=True,
                            # persistent_workers=True
                            )
    
    train_len=len(train_dataset)
    test_len=len(test_dataset)
    test_per_epochs=train_len // test_len *10

    # model_dir="weights"
    model=uvRex_get_model(backbone, pretrained, model_dir, Init_Epoch, device).to(device)

    if Freeze_Train:
        model.freeze_backbone()

    model.train()

    #Optimizer
    scaler = GradScaler()

    Init_lr         = 1e-4
    Min_lr          = Init_lr * 0.01
    nbs             = 16
    lr_limit_max    = 1e-4 
    lr_limit_min    = 1e-4 
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    momentum            = 0.9
    weight_decay        = 1e-8

    lr_decay_type       = 'cos'

    optimizer=optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay)

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epoch_sum)

    log_dir            = 'logs'
    loss_log_file=f"{log_dir}/uvRex_{backbone}_loss_{Init_Epoch}.csv"

    with open(loss_log_file, 'w', newline='') as f:
        loss_writer = csv.writer(f)
        loss_writer.writerow(['epoch', 'train_loss', 'test_loss'])

    epoch_range = range(Init_Epoch, epoch_sum)
    epoch_pbar = tqdm(epoch_range, desc='Training_Progress', unit='epoch', position=0, leave=True)

    for epoch in epoch_pbar:
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        if (epoch+1) % test_per_epochs==0:
            train_loss, test_loss = uvRex_train_one_epoch(model, optimizer, scaler, dataAug, device, train_loader, test_loader)

            with open(loss_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{test_loss:.6f}"])

            torch.save(model.state_dict(), f"{model_dir}/uvRex_{backbone}_epoch{epoch}.pth")

        else:
            train_loss, _ = uvRex_train_one_epoch(model, optimizer, scaler, dataAug, device, train_loader)
            
            with open(loss_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", ""])


def predict_single(input_dir:str, model_dir:str, img:str, texture:str, backbone, Init_Epoch, device):
    data_dir=Path(input_dir)
    ori_path=data_dir/f"cloth/{img}"
    mask_path=data_dir/f"mask/{img}"
    normal_path=data_dir/f"normal/{img}"
    texture_path=data_dir/f"tex/{texture}"

    ori_tensor=img2tensor_rgb(ori_path).unsqueeze(0).to(device)# [1,3,H,W]
    texture_tensor=img2tensor_rgb(texture_path).unsqueeze(0).to(device)

    binary_mask= get_binary_mask(mask_path)
    normal_tensor=img2tensor_rgb(normal_path, binary_mask).unsqueeze(0).to(device)

    mask_tensor=torch.from_numpy(binary_mask).unsqueeze(0).to(device)# [1,H,W]

    model=uvRex_get_model(backbone, False, model_dir, Init_Epoch, device).to(device)
    model.eval()
    
    res_tensor=uvRex_predict(ori_tensor, mask_tensor, normal_tensor, texture_tensor, model, device)
    res_int = (res_tensor * 255.0).to(torch.uint8)

    #img_save
    output_dir=Path("output")
    output_dir.mkdir(exist_ok=True)
    res_path=output_dir/img

    res_np = res_int[0].cpu().numpy().transpose(1, 2, 0)
    res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(res_path), res_bgr)


if __name__ == "__main__":
    args=get_args()

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode=='train':
        train_main(args.input_dir, 
                   args.model_dir,
                   args.backbone, 
                   args.pretrained, 
                   args.Freeze_Train, 
                   args.batch_size, 
                   args.Init_Epoch, 
                   args.epoch_sum, 
                   device,
                   args.seed
                   )
    else:
        predict_single(args.input_dir,
                     args.model_dir,
                     args.img, 
                     args.texture, 
                     args.backbone, 
                     args.Init_Epoch, 
                     device
                     )