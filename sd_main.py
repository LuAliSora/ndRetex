import os
import multiprocessing
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['HF_HOME'] = 'D:/BaiduNetdiskDownload/hf_cache'
# os.environ['HF_HOME'] = '/autodl-fs/data/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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


from diffusers.optimization import get_scheduler
from accelerate import Accelerator


# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import ProjectConfiguration, set_seed

from modules.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn, sd_get_model)
from modules.dataPr import SD_ImgSet, sd_collate_fn
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
        "--tex_pretrained",
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
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=2,
        help="gradient_accumulation_steps"
    )
    #uvRex_model
    parser.add_argument(
        "--uvRex_model_dir",
        type=str,
        default="weights",
    )
    parser.add_argument(
        "--uvRex_backbone",
        type=str,
        choices=['vgg', 'resnet50'],
        default='vgg'
    )
    parser.add_argument(
        "--uvRex_Epoch",
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


def train_main(input_dir:str, uvRex_model_state, tex_pretrained, Freeze_Train, batch_size, grad_acc_steps, Init_Epoch, epoch_sum, seed): 
    if Init_Epoch<0 or Init_Epoch>epoch_sum:
        raise Exception("Require valid epoch!")
    
    seed_everything(seed)

    #logs
    log_dir            = 'logs'
    loss_log_file=f"{log_dir}/sd_loss_{Init_Epoch}.csv"

    with open(loss_log_file, 'w', newline='') as f:
        loss_writer = csv.writer(f)
        loss_writer.writerow(['epoch', 'train_loss', 'test_loss'])

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_acc_steps,
        mixed_precision="fp16",
        # log_with="tensorboard",
        # project_dir=log_dir
    )
    device = accelerator.device
    
    # Load_data
    print("Load_data.")

    data_dir=Path(input_dir)
    train_dir=data_dir/"train"
    test_dir=data_dir/"test"

    train_dataset   = SD_ImgSet(str(train_dir))
    test_dataset    = SD_ImgSet(str(test_dir))

    train_loader=DataLoader(train_dataset, 
                            batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed),
                            drop_last=True,
                            collate_fn=sd_collate_fn,
                            # persistent_workers=True
                            )
    test_loader=DataLoader(test_dataset, 
                            batch_size, 
                            num_workers=4, 
                            pin_memory=True, 
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed),
                            drop_last=True,
                            collate_fn=sd_collate_fn,
                            # persistent_workers=True
                            )
    
    train_len=len(train_dataset)
    test_len=len(test_dataset)
    test_per_epochs=train_len // test_len *10

    # Load_model
    model_dict=sd_get_model(uvRex_model_state, tex_pretrained, Init_Epoch, device)
    texture_controlnet=model_dict["texture_controlnet"]
    texture_controlnet.requires_grad_(True)
    texture_controlnet.train()
    texture_controlnet.enable_gradient_checkpointing()

    #Optimizer
    Init_lr         = 1e-4
    Min_lr          = Init_lr * 0.01
    nbs             = 16
    lr_limit_max    = 1e-4 
    lr_limit_min    = 1e-4 
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    momentum            = 0.9
    weight_decay        = 1e-8

    lr_decay_type       = 'cosine'
    optimizer=optim.Adam(texture_controlnet.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay)

    lr_scheduler = get_scheduler(
        lr_decay_type,
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(train_len//batch_size * (epoch_sum-Init_Epoch)) // grad_acc_steps,
    )

    texture_controlnet, optimizer, lr_scheduler, train_loader, test_loader = accelerator.prepare(
        texture_controlnet, optimizer, lr_scheduler, train_loader, test_loader
    )

    epoch_range = range(Init_Epoch, epoch_sum)
    epoch_pbar = tqdm(epoch_range, desc='Training_Progress', unit='epoch', position=0, leave=True)

    for epoch in epoch_pbar:
        if (epoch+1) % test_per_epochs==0:
            train_loss, test_loss = sd_train_one_epoch(accelerator, model_dict, optimizer, lr_scheduler, train_loader, test_loader)

            with open(loss_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{test_loss:.6f}"])

            torch.save(texture_controlnet.state_dict(), f"{uvRex_model_state["model_dir"]}/texControl_epoch{epoch}.pth")

        else:
            train_loss, _ = sd_train_one_epoch(accelerator, model_dict, optimizer, lr_scheduler, train_loader)
            
            with open(loss_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", ""])


if __name__ == "__main__":
    args=get_args()

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    uvRex_model_state={
        "backbone":args.uvRex_backbone,
        "pretrained":False, 
        "model_dir":args.uvRex_model_dir, 
        "Init_Epoch":args.uvRex_Epoch
    }
    
    if args.mode=='train':
        train_main(args.input_dir, 
                   uvRex_model_state, 
                   args.tex_pretrained, 
                   args.Freeze_Train, 
                   args.batch_size, 
                   args.grad_acc_steps,
                   args.Init_Epoch, 
                   args.epoch_sum, 
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