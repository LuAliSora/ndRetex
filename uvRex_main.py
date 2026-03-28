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
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
import csv
import cv2

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init

from modules.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from modules.dataPr import Masked_ImgSet, get_dataAug, img_masked, img2np_rgb
from modules.train import uvRex_train_one_epoch

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
        default=0
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
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


def train_main(input_dir:str, model_dir:str, backbone, pretrained, Freeze_Train, batch_size, Init_Epoch, epoch_sum, device, seed):  
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

    # input_dir=Path("input")
    data_dir=Path(input_dir)
    train_dir=data_dir/"train"
    test_dir=data_dir/"test"
    img_folder="normal"
    mask_folder="mask"

    train_dataset   = Masked_ImgSet(str(train_dir/img_folder), str(train_dir/mask_folder))
    test_dataset     = Masked_ImgSet(str(test_dir/img_folder), str(test_dir/mask_folder))

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
    test_per_epochs=train_len // test_len

    # model_dir="weights"

    model=get_model(backbone, pretrained, model_dir, Init_Epoch, device)

    if Freeze_Train:
        model.freeze_backbone()

    model.train()
    model.to(device)

    scaler = GradScaler()

    optimizer=optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay)

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epoch_sum)

    loss_log_file=f"{model_dir}/uvRex_{backbone}_loss.csv"
    loss_log_path=Path(loss_log_file)
    if loss_log_path.is_file()==False:
        with open(loss_log_file, 'w', newline='') as f:
            loss_writer = csv.writer(f)
            loss_writer.writerow(['epoch', 'train_loss', 'test_loss'])

    epoch_range = range(Init_Epoch, epoch_sum)
    epoch_pbar = tqdm(epoch_range, desc='Training_Progress', unit='epoch')

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


def predict_main(input_dir:str, model_dir:str, img:str, texture:str, backbone, Init_Epoch, device):
    data_dir=Path(input_dir)
    ori_path=data_dir/f"ori/{img}"
    normal_path=data_dir/f"normal/{img}"
    mask_path=data_dir/f"mask/{img}"
    texture_path=data_dir/f"tex/{texture}"

    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask_np > 127)

    normal_np = img2np_rgb(normal_path).astype(np.float32)/ 255.0
    normal_masked =img_masked(normal_np, binary_mask)
    normal_tensor = torch.from_numpy(normal_masked).permute(2, 0, 1).contiguous()
    normal_tensor = normal_tensor.unsqueeze(0).to(device) #[1, 3, H, W]

    # model_dir="weights"

    model=get_model(backbone, False, model_dir, Init_Epoch, device)
    model.eval()
    model.to(device)

    with torch.no_grad():
        uv_tensor=model(normal_tensor) #[1, 2, H, W]

    print(f"UV_range: [{uv_tensor.min():.3f}, {uv_tensor.max():.3f}]")

    uv_sampler = uv_tensor.clone().cpu()
    uv_min = uv_sampler.min()
    uv_max = uv_sampler.max()
    uv_sampler = (uv_sampler - uv_min) / (uv_max - uv_min) * 2 - 1
    uv_grid = uv_sampler.permute(0, 2, 3, 1) # [1, H, W, 2]

    texture_np = img2np_rgb(texture_path).astype(np.float32)/ 255.0
    texture_tensor = torch.from_numpy(texture_np).permute(2, 0, 1).contiguous() 
    texture_tensor = texture_tensor.unsqueeze(0) # [1, 3, H_tex, W_tex]

    sampled_color = F.grid_sample(
        texture_tensor,
        uv_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ) 

    retex_np = sampled_color.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    retex_int = (retex_np * 255.0).astype(np.uint8)

    ori_np = img2np_rgb(ori_path)

    mask_3ch = np.stack([binary_mask]*3, axis=2)

    result_np = np.where(mask_3ch, retex_int, ori_np)

    output_dir=Path("output")
    output_dir.mkdir(exist_ok=True)
    res_path=output_dir/img
    cv2.imwrite(str(res_path), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))


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
        predict_main(args.input_dir,
                     args.model_dir,
                     args.img, 
                     args.texture, 
                     args.backbone, 
                     args.Init_Epoch, 
                     device
                     )