import torch
import numpy as np

def uvRex_get_model(backbone, pretrained, model_dir:str, Init_Epoch, device):
    from nets.unet_training import weights_init
    from nets.unet import Unet
    
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
        # DDPMScheduler,
        DDIMScheduler,
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

    model_dict["noise_scheduler"] = DDIMScheduler.from_pretrained(
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