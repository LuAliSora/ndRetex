from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from modules.dataPr import img_masked, img2np_rgb
from modules.utils import  uvRex_get_model


def uvRex_predict_main(input_dir:str, model_dir:str, img:str, texture:str, backbone, Init_Epoch, device):
    data_dir=Path(input_dir)
    ori_path=data_dir/f"ori/{img}"
    normal_path=data_dir/f"normal/{img}"
    mask_path=data_dir/f"mask/{img}"
    texture_path=data_dir/f"tex/{texture}"

    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask_np > 127)

    normal_np = img2np_rgb(normal_path).astype(np.float32)/ 255.0
    normal_masked =img_masked(normal_np, binary_mask)
    normal_tensor = torch.from_numpy(normal_masked).permute(2, 0, 1).contiguous()
    normal_tensor = normal_tensor.unsqueeze(0).to(device) #[1, 3, H, W]

    # model_dir="weights"

    model=uvRex_get_model(backbone, False, model_dir, Init_Epoch, device)
    model.eval()
    model.to(device)

    with torch.no_grad():
        uv_tensor=model(normal_tensor) #[1, 2, H, W]

    print(f"UV_range: [{uv_tensor.min():.3f}, {uv_tensor.max():.3f}]")

    uv_sampler = uv_tensor.clone().cpu()
    uv_min = uv_sampler.min()
    uv_max = uv_sampler.max()
    uv_sampler = (uv_sampler - uv_min) / (uv_max - uv_min) * 2 - 1
    uv_grid = uv_sampler.permute(0, 2, 3, 1) # [1, H, W, 2]

    texture_np = img2np_rgb(texture_path).astype(np.float32)/ 255.0
    texture_tensor = torch.from_numpy(texture_np).permute(2, 0, 1).contiguous() 
    texture_tensor = texture_tensor.unsqueeze(0) # [1, 3, H_tex, W_tex]

    sampled_color = F.grid_sample(
        texture_tensor,
        uv_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ) 

    retex_np = sampled_color.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    retex_int = (retex_np * 255.0).astype(np.uint8)

    ori_np = img2np_rgb(ori_path)

    mask_3ch = np.stack([binary_mask]*3, axis=2)

    result_np = np.where(mask_3ch, retex_int, ori_np)

    output_dir=Path("output")
    output_dir.mkdir(exist_ok=True)
    res_path=output_dir/img
    cv2.imwrite(str(res_path), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))