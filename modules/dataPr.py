import torch
from torch.utils import data
# import torch.nn.functional as F
# import torchvision.transforms as transforms

import numpy as np

from pathlib import Path
from PIL import Image

import kornia.augmentation as K

class ImgSet(data.Dataset):

    def __init__(self, img_dir, if_imgTrans=True):
        normal_dir=img_dir/"normal"
        mask_dir=img_dir/"mask"
        
        self.normal_list=[img for img in normal_dir.glob('*.jpg')]
        self.mask_list=[img for img in mask_dir.glob('*.jpg')]
        # print(self.imgList)

        trans_size=[512, 512]
        if if_imgTrans:
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

        normal_img = Image.open(str(normal_path)).convert('RGB')
        mask_img = Image.open(str(mask_path)).convert('L')

        normal_np = np.array(normal_img).astype(np.float32)  # (H, W, C)
        mask_np = np.array(mask_img).astype(np.float32)      # (H, W)
        binary_mask = (mask_np > 127)
        normal_np[~binary_mask] = [0, 0, 0]# Masked normal
        normal_tensor = torch.from_numpy(normal_np).permute(2, 0, 1) / 255.0

        if self.transfm!=None:
            normal_tensor = normal_tensor.unsqueeze(0)  # (1, C, H, W)
            normal_tensor = self.transfm(normal_tensor)
            normal_tensor = normal_tensor.squeeze(0)  # (C, H, W)

        return normal_tensor
    
    def __len__(self):
        return len(self.normal_list)
    
