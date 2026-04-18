import random

import numpy as np
import torch
# from PIL import Image

from nets.unet_training import weights_init
from nets.unet import Unet


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
# get_model
#---------------------------------------------------#
def uvRex_get_model(backbone, pretrained, model_dir:str, Init_Epoch, device):
    model = Unet(2, backbone, pretrained, model_dir)

    if pretrained:
        return model

    if Init_Epoch==0:
        weights_init(model)
        return model
    
    # model_dir=Path("weights")
    model_path=f"{model_dir}/uvRex_{backbone}_epoch{Init_Epoch}.pth"

    print(f'Load uvRex_weights {model_path}.')

    model_dict      = model.state_dict()
    pretrained_dict = torch.load(str(model_path), map_location = device, weights_only=True)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    # print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    return model


def sd_get_model(uvRex_model_state, tex_pretrained, Init_Epoch, device)->dict:
    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
        UniPCMultistepScheduler
    )
    from transformers import CLIPTokenizer, CLIPTextModel

    pre_sd="runwayml/stable-diffusion-v1-5"
    pre_normal="lllyasviel/sd-controlnet-normal"
    pre_canny="lllyasviel/sd-controlnet-canny"
    
    print("Load_model.")
    model_dict={}
    model_dict["uvRex_model"]=uvRex_get_model(uvRex_model_state["backbone"], 
                                uvRex_model_state["pretrained"], 
                                uvRex_model_state["model_dir"], 
                                uvRex_model_state["Init_Epoch"], 
                                device
                                ).to(device)

    model_dict["vae"] = AutoencoderKL.from_pretrained(
        pre_sd, 
        subfolder="vae",
        # torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    model_dict["unet"] = UNet2DConditionModel.from_pretrained(
        pre_sd, 
        subfolder="unet",
        # torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    # Text_encoder
    model_dict["tokenizer"] = CLIPTokenizer.from_pretrained(
        pre_sd, 
        subfolder="tokenizer",
        local_files_only=True
    )
    model_dict["text_encoder"] = CLIPTextModel.from_pretrained(
        pre_sd, 
        subfolder="text_encoder",
        # torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    # Controlnet
    model_dict["normal_controlnet"] = ControlNetModel.from_pretrained(
        pre_normal,
        # torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)


    texture_controlnet = ControlNetModel.from_pretrained(
        pre_canny,  
        # torch_dtype=torch.float16,
        local_files_only=True
    )
    if tex_pretrained==False:
        tex_model_path=f'{uvRex_model_state["model_dir"]}/texControl_epoch{Init_Epoch}.pth'
        tex_state_dict=torch.load(tex_model_path, map_location = device, weights_only=True)
        texture_controlnet.load_state_dict(tex_state_dict)
    model_dict["texture_controlnet"] = texture_controlnet.to(device)

    model_dict["noise_scheduler"] = DDPMScheduler.from_pretrained(
        pre_sd, 
        subfolder="scheduler",
        local_files_only=True
    )

    for model_name, model in model_dict.items():
    # 检查是否是 PyTorch 模型
        if hasattr(model, 'parameters'):
            model.requires_grad_(False)
            model.eval()

    return model_dict
#---------------------------------------------------#
# loss_fn
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

def uvRex_loss(normal, uv):
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
    
    epsilon = 1e-6

    norm_n = torch.sqrt(nx**2 + ny**2 + nz**2 + epsilon)
    nx = nx / norm_n
    ny = ny / norm_n
    nz = nz / norm_n
    
    # 几何损失
    nz_square = nz**2 + epsilon
    geo1= du_x**2 + dv_x**2 - 1 - nx**2 / nz_square
    geo2= du_y**2 + dv_y**2 - 1 - ny**2 / nz_square
    geo3= du_x * du_y + dv_x * dv_y - (nx * ny) / nz_square
    loss_geo= geo1**2 + geo2**2 + geo3**2
    
    # Z方向约束
    jacobian = du_x * dv_y - du_y * dv_x
    loss_flip = torch.relu(-jacobian)
    # loss_area = (jacobian - 1.0)**2

    # 平滑性约束
    loss_smooth = (du_x**2 + du_y**2 + dv_x**2 + dv_y**2).mean()
    
    # 计算平均损失（而不是总和）
    loss_fin = loss_geo.mean() + 0.1 * loss_flip.mean()+ 0.01 * loss_smooth.mean()
    
    return loss_fin