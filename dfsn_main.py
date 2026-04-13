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