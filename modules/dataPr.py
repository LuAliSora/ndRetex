import torch
from torch.utils import data
# import torch.nn.functional as F
# import torchvision.transforms as transforms
from torch.utils.data import default_collate

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
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()# [3,H,W]
    # img_tensor = img_tensor.unsqueeze(0) #[1, 3, H, W]
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

        binary_mask = get_binary_mask(str(mask_path))
        res_tensor=img2tensor_rgb(str(img_path),binary_mask)# [3,H,W]

        # transfm=get_dataAug()
        # res_tensor = res_tensor.unsqueeze(0)  # (1, C, H, W)
        # res_tensor = transfm(res_tensor)
        # res_tensor = res_tensor.squeeze(0)  # (C, H, W)

        # tensor2img(res_tensor, f"output/{img_path.name}")
        return res_tensor
    
    def __len__(self):
        return len(self.img_list)
    

class SD_ImgSet(data.Dataset):

    def __init__(self, data_dir:str):
        img_dir=Path(data_dir)
        self.ori_dir=img_dir/"cloth"
        self.mask_dir=img_dir/"mask"
        self.normal_dir=img_dir/"normal"
        self.tex_dir=img_dir/"tex"

        self.img_list = sorted([str(img.name) for img in (self.ori_dir).glob('*.jpg')])
        self.tex_list = sorted([str(tex.name) for tex in (self.tex_dir).glob('*.jpg')])

        self.tex_num=len(self.tex_list)

        self.prompt="clothes_prompt"

    @torch.no_grad()
    def __getitem__(self, index):
        img_name=self.img_list[index]
        tex_name=self.tex_list[index%(self.tex_num)]

        ori_path=self.ori_dir/img_name
        mask_path=self.mask_dir/img_name
        normal_path=self.normal_dir/img_name
        tex_path=self.tex_dir/tex_name

        ori_tensor=img2tensor_rgb(ori_path)# [3,H,W]
        tex_tensor=img2tensor_rgb(tex_path)# [3,H,W]

        binary_mask = get_binary_mask(str(mask_path))
        normal_tensor=img2tensor_rgb(str(normal_path),binary_mask)# [3,H,W]

        mask_tensor=torch.from_numpy(binary_mask)# [H,W]

        return ori_tensor, mask_tensor, normal_tensor, tex_tensor, self.prompt
    
    def __len__(self):
        return len(self.img_list)
    

#collate_fn
def sd_collate_fn(batch):
    """
    自定义 collate_fn 用于处理数据集返回的混合类型数据
    
    Args:
        batch: list of tuples, 每个tuple包含 (ori_tensor, mask_tensor, normal_tensor, tex_tensor, prompt)
    
    Returns:
        dict: 包含批量化后的数据
    """
    # 解包batch中的各个元素
    ori_list = [item[0] for item in batch]
    mask_list = [item[1] for item in batch]
    normal_list = [item[2] for item in batch]
    tex_list = [item[3] for item in batch]
    prompts = [item[4] for item in batch]
    
    # 对于tensor数据，使用default_collate进行批量化
    ori_batch = default_collate(ori_list)      # [B, 3, H, W]
    mask_batch = default_collate(mask_list)    # [B, H, W]
    normal_batch = default_collate(normal_list) # [B, 3, H, W]
    tex_batch = default_collate(tex_list)      # [B, 3, H, W]
    
    return ori_batch, mask_batch, normal_batch, tex_batch, prompts






