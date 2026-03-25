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



class ImgSet(data.Dataset):

    def __init__(self, img_dir:Path, DA=True):
        normal_dir=img_dir/"normal"
        mask_dir=img_dir/"mask"
        
        self.normal_list = sorted(normal_dir.glob('*.jpg'))
        self.mask_list = sorted(mask_dir.glob('*.jpg'))
        # print(self.imgList)

        trans_size=[512, 512]
        if DA:
            self.transfm = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.3),
                K.RandomRotation(degrees=30, p=0.5),
                K.RandomResizedCrop(size=trans_size, scale=(0.8, 1.0), p=0.5),
                K.RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.5),
                same_on_batch=False,
                data_keys=["input"]
            )
        else:
            self.transfm=None

    @torch.no_grad()
    def __getitem__(self, index):
        normal_path=self.normal_list[index]
        mask_path=self.mask_list[index]

        normal_tensor=get_masked(normal_path, mask_path)

        # if self.transfm!=None:
        #     normal_tensor = normal_tensor.unsqueeze(0)  # (1, C, H, W)
        #     normal_tensor = self.transfm(normal_tensor)
        #     normal_tensor = normal_tensor.squeeze(0)  # (C, H, W)

        return normal_tensor
    
    def __len__(self):
        return len(self.normal_list)
    
