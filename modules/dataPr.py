import torch
from torch.utils import data
# import torch.nn.functional as F
# import torchvision.transforms as transforms

import numpy as np

from pathlib import Path
# from PIL import Image
import cv2

import kornia.augmentation as K


def img2np_rgb(img_path:str):
    img_np = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_rgb


def img2tensor_rgb(img_path:str, binary_mask=None):
    img_np = img2np_rgb(img_path).astype(np.float32)/ 255.0
    if binary_mask is not None:
        img_np =img_masked(img_np, binary_mask)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    img_tensor = img_tensor.unsqueeze(0) #[1, 3, H, W]
    return img_tensor


def get_binary_mask(mask_path:str):
    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask_np > 127)# [H,W]
    return  binary_mask


def img_masked(ori_np, binary_mask):  
    res=ori_np.copy()
    res[~binary_mask] = 0
    return res


def get_dataAug(trans_size_2D=[512, 512]):
    transfm = K.AugmentationSequential(
        K.Resize(size=trans_size_2D),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.3),
        K.RandomRotation(degrees=30, p=0.5),
        # K.RandomResizedCrop(size=trans_size, scale=(0.8, 1.0), p=0.5),
        K.RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.5),
        same_on_batch=False,
        data_keys=["input"]
    )
    return transfm


class Masked_ImgSet(data.Dataset):

    def __init__(self, data_dir:str, mask_dir:str):
        self.img_dir=Path(data_dir)
        self.mask_dir=Path(mask_dir)
        
        self.img_list = sorted([str(img.name) for img in (self.img_dir).glob('*.jpg')])
        # print(self.imgList)

        # trans_size=[512, 512]
        # self.transfm=get_dataAug(trans_size)

    @torch.no_grad()
    def __getitem__(self, index):
        img_name=self.img_list[index]

        img_path=self.img_dir/img_name
        mask_path=self.mask_dir/img_name

        img_np = img2np_rgb(str(img_path)).astype(np.float32)/ 255.0

        binary_mask = get_binary_mask(str(mask_path))

        res_np=img_masked(img_np, binary_mask)
        res_tensor = torch.from_numpy(res_np).permute(2, 0, 1).contiguous()

        # transfm=get_dataAug()
        # res_tensor = res_tensor.unsqueeze(0)  # (1, C, H, W)
        # res_tensor = transfm(res_tensor)
        # res_tensor = res_tensor.squeeze(0)  # (C, H, W)

        # tensor2img(res_tensor, f"output/{img_path.name}")
        return res_tensor
    
    def __len__(self):
        return len(self.img_list)
    

class Masked_ImgSet(data.Dataset):

    def __init__(self, data_dir:str,  model_dir:str, backbone, device):
        img_dir=Path(data_dir)
        self.ori_dir=img_dir/"ori"
        self.mask_dir=img_dir/"mask"
        self.normal_dir=img_dir/"normal"
        self.texture_dir=img_dir/"tex"

        self.img_list = sorted([str(img.name) for img in (self.ori_dir).glob('*.jpg')])
        self.tex_list = sorted([str(tex.name) for tex in (self.texture_dir).glob('*.jpg')])

        self.tex_num=len(self.tex_list)

    @torch.no_grad()
    def __getitem__(self, index):
        img_name=self.img_list[index]
        tex_name=self.tex_list[index%(self.tex_num)]

