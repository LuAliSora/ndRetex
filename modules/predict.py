import torch
import torch.nn.functional as F
from torch.amp import autocast

from modules.dataPr import tensor_combine


@torch.no_grad()
def uvRex_predict(normal_tensor, texture_tensor, mask_tensor, model, device):

    with torch.no_grad():
        with autocast(device.type):
            x = normal_tensor
            y = model(x) #[B, 2, H, W]

    # print(f"UV_range: [{y.min():.3f}, {y.max():.3f}]")

    uv_tensor = y.clone()
    B, _, H, W = uv_tensor.shape

    # uv_normalized
    uv_flat = uv_tensor.view(B, 2, -1)
    uv_min = uv_flat.min(dim=-1, keepdim=True)[0].min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    uv_max = uv_flat.max(dim=-1, keepdim=True)[0].max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    uv_dist = uv_max - uv_min
    uv_range = torch.where(uv_dist == 0, torch.ones_like(uv_dist), uv_dist)

    uv_norm = (uv_tensor - uv_min) / (uv_range) * 2 - 1
    uv_grid = uv_norm.permute(0, 2, 3, 1) # [B, H, W, 2]

    if texture_tensor.dtype != uv_grid.dtype:
        uv_grid = uv_grid.to(dtype=texture_tensor.dtype)

    # uv_sampled
    sampled_color = F.grid_sample(
        texture_tensor,
        uv_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    bg=torch.zeros_like(sampled_color)
    retex=tensor_combine(bg, sampled_color, mask_tensor)

    return retex


@torch.no_grad()
def sd_predict(data, model_dict, device, contro_scale=[1.0, 0.8], infer_steps=20, guidance_scale=7.5,  strength=0.8):
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

    uvRex_model = model_dict["uvRex_model"]
    vae = model_dict["vae"]
    unet = model_dict["unet"]
    tokenizer = model_dict["tokenizer"]
    text_encoder = model_dict["text_encoder"]
    normal_controlnet = model_dict["normal_controlnet"]
    texture_controlnet = model_dict["texture_controlnet"]
    noise_scheduler = model_dict["noise_scheduler"]

    mask, normal, tex, prompt = data

    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=[normal_controlnet, texture_controlnet],
        scheduler=UniPCMultistepScheduler.from_config(noise_scheduler.config, use_karras_sigmas=True),
        safety_checker=None,      
        feature_extractor=None  
    ).to(device)
    pipe.enable_model_cpu_offload()

    with torch.no_grad():
        rough=uvRex_predict(normal, tex, mask, uvRex_model, device)

        retex = pipe(
            prompt=prompt,
            image=rough,
            control_image=[normal, tex],
            controlnet_conditioning_scale=contro_scale,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            output_type="pt"  
        )

    return retex