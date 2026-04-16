import os
import multiprocessing
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['HF_HOME'] = 'D:/BaiduNetdiskDownload/hf_cache'
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

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    UniPCMultistepScheduler
)
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from transformers import CLIPTokenizer, CLIPTextModel

# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import ProjectConfiguration, set_seed

from modules.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn, uvRex_get_model)
from modules.dataPr import Rex_ImgSet
from modules.train import sd_train_one_epoch

MODEL_DICT = {
    "sd": "runwayml/stable-diffusion-v1-5",
    "normal_controlnet": "lllyasviel/sd-controlnet-normal",
    "texture_controlnet": "lllyasviel/sd-controlnet-canny",  # 用canny作为初始模板
}


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


def train_main(input_dir:str, uvRex_model, tex_pretrained, Freeze_Train, batch_size, Init_Epoch, epoch_sum, device, seed): 
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
    
    # Load_data
    print("Load_data.")

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

    # Load_model
    print("Load_model.")

    vae = AutoencoderKL.from_pretrained(
        MODEL_DICT["sd"], 
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_DICT["sd"], 
        subfolder="unet",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    # Text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_DICT["sd"], 
        subfolder="tokenizer",
        local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_DICT["sd"], 
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    # Controlnet
    normal_controlnet = ControlNetModel.from_pretrained(
        MODEL_DICT["normal_controlnet"],
        torch_dtype=torch.float16,
        local_files_only=True
    )

    if tex_pretrained:
        texture_controlnet = ControlNetModel.from_pretrained(
            MODEL_DICT["texture_controlnet"],  
            torch_dtype=torch.float16,
            local_files_only=True
        )

    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_DICT["sd_path"], 
        subfolder="scheduler",
        local_files_only=True
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # tokenizer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    normal_controlnet.requires_grad_(False)

    vae.eval()
    unet.eval()
    # tokenizer.eval()
    text_encoder.eval()
    normal_controlnet.eval()

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
    optimizer=optim.Adam(tex_pretrained.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay)

    lr_scheduler = get_scheduler(
        lr_decay_type,
        optimizer=optimizer,
        # num_warmup_steps=500,
        num_training_steps=(train_len//batch_size * (epoch_sum-Init_Epoch)) // grad_acc_steps,
    )

    texture_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        texture_controlnet, optimizer, train_dataloader, lr_scheduler
    )
    uvRex_model = uvRex_model.to(accelerator.device)
    vae = vae.to(accelerator.device)
    unet = unet.to(accelerator.device)
    normal_controlnet = normal_controlnet.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)


    epoch_range = range(Init_Epoch, epoch_sum)
    epoch_pbar = tqdm(epoch_range, desc='Training_Progress', unit='epoch', position=0, leave=True)

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
    
    uvRex_model=uvRex_get_model(args.uvrex_backbone, False, args.uvrex_model_dir, args.uvrex_Epoch, device)
    uvRex_model.eval()

    if args.mode=='train':
        train_main(args.input_dir, 
                   uvRex_model, 
                   args.tex_pretrained, 
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