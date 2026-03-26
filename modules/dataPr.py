import torch
from torch.utils import data
# import torch.nn.functional as F
# import torchvision.transforms as transforms

import numpy as np

from pathlib import Path
# from PIL import Image
import cv2

import kornia.augmentation as K


@torch.no_grad()
def tensor2img(imgTensor, save_path:str):
    tensor_copy=imgTensor.detach().cpu()
    img_np= tensor_copy.permute(1, 2, 0).numpy().clip(0, 1)
    img_np= (img_np * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


@torch.no_grad()
def get_masked(ori_path:str, mask_path:str):
    ori_np = cv2.imread(ori_path)
    ori_np = cv2.cvtColor(ori_np, cv2.COLOR_BGR2RGB)
    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    ori_float = ori_np.astype(np.float32)/ 255.0
    # mask_np = mask_img.astype(np.float32)
    
    binary_mask = (mask_np > 127)
    ori_float[~binary_mask] = 0

    res_tensor = torch.from_numpy(ori_float).permute(2, 0, 1).contiguous()
    return res_tensor



class Masked_ImgSet(data.Dataset):

    def __init__(self, img_dir:str, mask_dir:str, if_dataAug=True):
        self.img_dir=Path(img_dir)
        self.mask_dir=Path(mask_dir)
        
        self.img_list = sorted([str(img.name) for img in (self.img_dir).glob('*.jpg')])
        # print(self.imgList)

        trans_size=[512, 512]
        if if_dataAug:
            self.transfm = K.AugmentationSequential(
                K.Resize(size=trans_size),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.3),
                K.RandomRotation(degrees=30, p=0.5),
                # K.RandomResizedCrop(size=trans_size, scale=(0.8, 1.0), p=0.5),
                K.RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.5),
                same_on_batch=False,
                data_keys=["input"]
            )
        else:
            self.transfm = K.AugmentationSequential(
                K.Resize(size=trans_size),
                data_keys=["input"]
            )

    @torch.no_grad()
    def __getitem__(self, index):
        img_name=self.img_list[index]

        img_path=self.img_dir/img_name
        mask_path=self.mask_dir/img_name

        img_tensor=get_masked(str(img_path), str(mask_path))

        # if self.transfm!=None:
        #     normal_tensor = normal_tensor.unsqueeze(0)  # (1, C, H, W)
        #     normal_tensor = self.transfm(normal_tensor)
        #     normal_tensor = normal_tensor.squeeze(0)  # (C, H, W)

        # tensor2img(normal_tensor, f"output/{normal_path.name}")
        return img_tensor
    
    def __len__(self):
        return len(self.img_list)
    
