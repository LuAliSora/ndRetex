import torch
import torch.nn.functional as F
from torch.amp import autocast

from modules.utils import  uvRex_get_model


def uvRex_predict(ori_tensor, binary_mask, normal_tensor, texture_tensor, model, device):
    # model_dir="weights"
    # model=uvRex_get_model(backbone, False, model_dir, Init_Epoch, device)
    # model.eval()
    # model.to(device)

    with torch.no_grad():
        with autocast(device.type):
            x = normal_tensor.to(device)
            y = model(x) #[B, 2, H, W]

    # print(f"UV_range: [{y.min():.3f}, {y.max():.3f}]")

    uv_tensor = y.clone().cpu().float()
    B, _, H, W = uv_tensor.shape

    # uv_normalized
    uv_flat = uv_tensor.view(B, 2, -1)
    uv_min = uv_flat.min(dim=-1, keepdim=True)[0].min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    uv_max = uv_flat.max(dim=-1, keepdim=True)[0].max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    uv_dist = uv_max - uv_min
    uv_range = torch.where(uv_dist == 0, torch.ones_like(uv_dist), uv_dist)

    uv_norm = (uv_tensor - uv_min) / (uv_range) * 2 - 1
    uv_grid = uv_norm.permute(0, 2, 3, 1) # [B, H, W, 2]

    # uv_sampled
    sampled_color = F.grid_sample(
        texture_tensor,
        uv_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    retex_int = (sampled_color * 255.0).to(torch.uint8)
    
    # combine
    ori_int = (ori_tensor[:] * 255.0).to(torch.uint8)
    mask_3ch = binary_mask.unsqueeze(1).expand(-1, 3, -1, -1)
    results = torch.where(mask_3ch, retex_int, ori_int)#[B,3,H,W]

    return results