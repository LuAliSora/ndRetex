import random

import numpy as np
import torch
# from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
# def resize_image(image, size):
#     iw, ih  = image.size
#     w, h    = size

#     scale   = min(w/iw, h/ih)
#     nw      = int(iw*scale)
#     nh      = int(ih*scale)

#     image   = image.resize((nw,nh), Image.BICUBIC)
#     new_image = Image.new('RGB', size, (128,128,128))
#     new_image.paste(image, ((w-nw)//2, (h-nh)//2))

#     return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50'  : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

#---------------------------------------------------#
def compute_d_x(ori):
    """计算x方向梯度，支持batch维度"""
    temp = ori[:, :, 1:, :] - ori[:, :, :-1, :]
    d_x = torch.zeros_like(ori)
    d_x[:, :, 1:, :] = temp
    return d_x

def compute_d_y(ori):
    """计算y方向梯度，支持batch维度"""
    temp = ori[:, :, :, 1:] - ori[:, :, :, :-1]
    d_y = torch.zeros_like(ori)
    d_y[:, :, :, 1:] = temp
    return d_y

def uvRex_loss(uv, normal):
    """
    计算UV和Normal的损失函数
    
    Args:
        uv: [B, 2, H, W] - UV坐标
        normal: [B, 3, H, W] - 法线贴图 (x,y,z)
    
    Returns:
        loss: 标量损失值
    """
    # 计算UV的梯度
    du_x = compute_d_x(uv[:, 0:1])  # [B, 1, H, W]
    dv_x = compute_d_x(uv[:, 1:2])  # [B, 1, H, W]
    du_y = compute_d_y(uv[:, 0:1])  # [B, 1, H, W]
    dv_y = compute_d_y(uv[:, 1:2])  # [B, 1, H, W]
    
    # 获取法线分量
    nx = normal[:, 0:1]  # [B, 1, H, W]
    ny = normal[:, 1:2]  # [B, 1, H, W]
    nz = normal[:, 2:3]  # [B, 1, H, W]
    
    epsilon = 1e-8
    
    # 几何损失
    loss_geo = (du_x**2 + dv_x**2 - 1 - nx**2 / (nz**2 + epsilon))**2 + \
               (du_y**2 + dv_y**2 - 1 - ny**2 / (nz**2 + epsilon))**2 + \
               (du_x * du_y + dv_x * dv_y - 1 - nx * ny / (nz**2 + epsilon))**2
    
    # Z方向约束
    loss_z = torch.clamp(du_x * dv_y - du_y * dv_x, min=0)
    
    # 计算平均损失（而不是总和）
    loss_fin = loss_geo.mean() + 0.01 * loss_z.mean()
    
    return loss_fin