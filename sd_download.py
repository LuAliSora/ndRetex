# export HF_HOME="/autodl-fs/data/hf_cache"
# export HF_ENDPOINT=https://hf-mirror.com

import os
import multiprocessing
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['HF_HOME'] = 'D:/BaiduNetdiskDownload/hf_cache'
# os.environ['HF_HOME'] = '/autodl-fs/data/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from diffusers import StableDiffusionPipeline, ControlNetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 下载 Stable Diffusion v1-5
print("正在下载 Stable Diffusion v1-5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    # local_files_only=True
).to(device)
print("✅ Stable Diffusion v1-5 下载完成！")

# 2. 下载法线 ControlNet
print("正在下载法线 ControlNet...")
normal_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-normal",
    torch_dtype=torch.float16,
    # local_files_only=True
).to(device)
print("✅ 法线 ControlNet 下载完成！")

texture_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",  # 用任意一个 ControlNet 作为结构模板
    torch_dtype=torch.float16,
    # local_files_only=True
)
print("✅ canny ControlNet 下载完成！")

# 检查所有必要组件
print("✅ 模型组件检查:")
print(f"  UNet: {pipe.unet is not None}")
print(f"  VAE: {pipe.vae is not None}")
print(f"  Text Encoder: {pipe.text_encoder is not None}")
print(f"  Tokenizer: {pipe.tokenizer is not None}")
print(f"  Scheduler: {pipe.scheduler is not None}")


# # 检查是否需要下载额外文件
# try:
#     # 尝试一次小推理（不实际运行）
#     # 方法1：统一使用 float16
#     dummy_latents = torch.randn(1, 4, 64, 64, dtype=torch.float16).to(device)  # 改为 half
#     dummy_timestep = torch.tensor([500], dtype=torch.long).to(device)
#     dummy_encoder_hidden_states = torch.randn(1, 77, 768, dtype=torch.float16).to(device)  # 改为 half
    
#     with torch.no_grad():
#         noise_pred = pipe.unet(
#             dummy_latents, 
#             dummy_timestep, 
#             encoder_hidden_states=dummy_encoder_hidden_states
#         )
#     print("✅ UNet 前向传播正常")
# except Exception as e:
#     print(f"❌ 测试失败: {e}")


# # 完整的训练检查脚本
# # 1. 加载核心组件
# print("\n加载模型核心组件...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16
# ).to(device)

# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-normal",
#     torch_dtype=torch.float16
# ).to(device)

# print(f"模型位置:")
# print(f"  UNET: {pipe.unet.device}")
# print(f"  VAE: {pipe.vae.device}")
# print(f"  Text Encoder: {pipe.text_encoder.device}")
# print(f"  ControlNet: {controlnet.device}")

# # 2. 冻结所有组件（准备训练）
# pipe.vae.requires_grad_(False)
# pipe.unet.requires_grad_(False)
# pipe.text_encoder.requires_grad_(False)

# # 3. 创建新的纹理 ControlNet（可训练）
# from diffusers import ControlNetModel
# texture_controlnet = ControlNetModel.from_unet(pipe.unet, conditioning_channels=3)
# texture_controlnet = texture_controlnet.to(dtype=torch.float16)  # ← 添加这行
# texture_controlnet = texture_controlnet.to(device)  # ← 添加这行
# print(f"纹理 ControlNet 已创建并移到 {texture_controlnet.device}")

# # 4. 模拟一次训练步骤
# print("\n模拟训练步骤...")

# # 创建假数据（确保在正确的设备和 dtype）
# batch_size = 1
# image = torch.randn(batch_size, 3, 512, 512, dtype=torch.float16, device=device)
# texture_cond = torch.randn(batch_size, 3, 512, 512, dtype=torch.float16, device=device)
# normal_cond = torch.randn(batch_size, 3, 512, 512, dtype=torch.float16, device=device)
# prompt = ["a photo"]

# print(f"数据设备:")
# print(f"  image: {image.device}")
# print(f"  texture_cond: {texture_cond.device}")
# print(f"  normal_cond: {normal_cond.device}")

# # 编码图像
# with torch.no_grad():
#     latents = pipe.vae.encode(image).latent_dist.sample()
#     latents = latents * pipe.vae.config.scaling_factor
#     print(f"latents shape: {latents.shape}, device: {latents.device}")
    
#     # 编码文本
#     text_inputs = pipe.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#     text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
#     text_embeddings = pipe.text_encoder(text_inputs["input_ids"])[0]
#     print(f"text_embeddings shape: {text_embeddings.shape}, device: {text_embeddings.device}")

# # 添加噪声
# noise = torch.randn_like(latents)
# timesteps = torch.randint(0, 1000, (batch_size,), device=device)
# noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
# print(f"noisy_latents shape: {noisy_latents.shape}, device: {noisy_latents.device}")

# # 双 ControlNet 前向
# print("\n执行双 ControlNet 前向传播...")
# with torch.no_grad():
#     # 法线 ControlNet 前向
#     normal_output = controlnet(
#         noisy_latents, timesteps, 
#         encoder_hidden_states=text_embeddings,
#         controlnet_cond=normal_cond,
#         return_dict=False
#     )
    
#     # 检查返回值的类型和结构
#     print(f"法线 ControlNet 返回类型: {type(normal_output)}")
#     print(f"法线 ControlNet 返回长度: {len(normal_output) if hasattr(normal_output, '__len__') else 'N/A'}")
    
#     # 根据返回类型处理
#     if isinstance(normal_output, tuple):
#         if len(normal_output) == 2:
#             normal_down, normal_mid = normal_output
#         elif len(normal_output) == 1:
#             normal_down = normal_output[0]
#             normal_mid = None
#         else:
#             normal_down = normal_output
#             normal_mid = None
#     else:
#         normal_down = normal_output
#         normal_mid = None
    
#     print(f"法线 ControlNet 输出: {len(normal_down) if hasattr(normal_down, '__len__') else 'N/A'} 个 down samples")
#     if normal_mid is not None:
#         print(f"mid shape: {normal_mid.shape if hasattr(normal_mid, 'shape') else 'N/A'}")
    
#     # 纹理 ControlNet 前向
#     texture_output = texture_controlnet(
#         noisy_latents, timesteps,
#         encoder_hidden_states=text_embeddings,
#         controlnet_cond=texture_cond,
#         return_dict=False
#     )
    
#     print(f"纹理 ControlNet 返回类型: {type(texture_output)}")
    
#     if isinstance(texture_output, tuple):
#         if len(texture_output) == 2:
#             texture_down, texture_mid = texture_output
#         elif len(texture_output) == 1:
#             texture_down = texture_output[0]
#             texture_mid = None
#         else:
#             texture_down = texture_output
#             texture_mid = None
#     else:
#         texture_down = texture_output
#         texture_mid = None
    
#     print(f"纹理 ControlNet 输出: {len(texture_down) if hasattr(texture_down, '__len__') else 'N/A'} 个 down samples")
    
#     # 合并
#     if normal_mid is not None and texture_mid is not None:
#         down_samples = [n + t for n, t in zip(normal_down, texture_down)]
#         mid_sample = normal_mid + texture_mid
#     else:
#         # 如果只有一个输出，直接使用
#         down_samples = normal_down if texture_down is None else texture_down
#         mid_sample = None
    
#     print(f"合并后的输出: {len(down_samples) if hasattr(down_samples, '__len__') else 'N/A'} 个 down samples")

# # UNet 预测
# print("\n执行 UNET 预测...")
# with torch.no_grad():
#     unet_kwargs = {
#         "sample": noisy_latents,
#         "timestep": timesteps,
#         "encoder_hidden_states": text_embeddings,
#         "down_block_additional_residuals": down_samples if down_samples is not None else None,
#     }
    
#     if mid_sample is not None:
#         unet_kwargs["mid_block_additional_residual"] = mid_sample
    
#     noise_pred = pipe.unet(**unet_kwargs)[0]
    
# print(f"noise_pred shape: {noise_pred.shape}, device: {noise_pred.device}")

# # 计算损失
# loss = torch.nn.functional.mse_loss(noise_pred, noise)
# print(f"\n✅ 训练步骤成功！Loss: {loss.item():.6f}")

# # 检查梯度
# print("\n梯度检查:")
# for name, param in texture_controlnet.named_parameters():
#     if param.requires_grad:
#         print(f"  {name}: requires_grad={param.requires_grad}")
#     break  # 只显示第一个

# print(f"  UNet 需要梯度: {pipe.unet.conv_in.weight.requires_grad}")
# print(f"  VAE 需要梯度: {pipe.vae.encoder.conv_in.weight.requires_grad}")

# # 显存使用情况
# if torch.cuda.is_available():
#     print(f"\n显存使用:")
#     print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
#     print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# print("\n🎉 所有组件正常工作，可以开始训练！")