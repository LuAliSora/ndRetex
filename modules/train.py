import torch
from torch.amp import autocast
import torch.nn.functional as F

# from accelerate import Accelerator

from modules.utils import uvRex_loss
from modules.predict import uvRex_predict


def uvRex_train_one_epoch(model, optimizer, scaler, dataAug, device, train_loader, test_loader=None):

    train_loss = 0.0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast(device.type):
            x_aug = dataAug(data).to(device)
            y = model(x_aug)
            # print(x_aug.shape, y.shape)
            loss = uvRex_loss(x_aug, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    test_loss = 0.0
    if test_loader is not None:
        model.eval()
        with torch.no_grad():  # 禁用梯度计算，节省内存和计算
            for i, data in enumerate(test_loader):
                with autocast(device.type):
                    x_aug = dataAug(data).to(device)
                    y = model(x_aug)
                    loss = uvRex_loss(x_aug, y)

                test_loss += loss.item()

        model.train()
    # print(train_loss, test_loss)
    return train_loss, test_loss


def sd_cal_loss(data, model_dict, device, eval_flag=False):
    uvRex_model = model_dict["uvRex_model"]
    vae = model_dict["vae"]
    unet = model_dict["unet"]
    tokenizer = model_dict["tokenizer"]
    text_encoder = model_dict["text_encoder"]
    normal_controlnet = model_dict["normal_controlnet"]
    texture_controlnet = model_dict["texture_controlnet"]
    noise_scheduler = model_dict["noise_scheduler"]

    # ori, mask, normal, tex, prompt = data
    mask, normal, tex, prompt = data

    with torch.no_grad():
        rough=uvRex_predict(normal, tex, mask, uvRex_model, device)
    
    with torch.no_grad():
    # VAE编码
        latents = vae.encode(rough).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    # 添加噪声（img2img的核心）
    # 随机选择时间步
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],), device=latents.device
    ).long()
    
    # 添加噪声
    with torch.no_grad():
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 2. 编码prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]

    # 3. 两个ControlNet的前向传播
    # 法线ControlNet（冻结，不计算梯度）
    with torch.no_grad():
        normal_down_block_res_samples, normal_mid_block_res_sample = normal_controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=normal,
            conditioning_scale=1.0,
            return_dict=False,
        )

    with torch.no_grad() if eval_flag else torch.enable_grad():
        # 纹理ControlNet
        texture_down_block_res_samples, texture_mid_block_res_sample = texture_controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=tex,
            conditioning_scale=1.0,
            return_dict=False,
        )

        # 4. 合并两个ControlNet的输出
        # 简单相加或加权融合
        down_block_res_samples = [
            normal_sample + texture_sample 
            for normal_sample, texture_sample in zip(
                normal_down_block_res_samples, texture_down_block_res_samples
            )
        ]
        mid_block_res_sample = normal_mid_block_res_sample + texture_mid_block_res_sample
        # 5. UNet前向传播（预测噪声）
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
    # 6. 计算损失
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    loss = F.mse_loss(model_pred, target, reduction="mean")
    return loss
            
            
def sd_train_one_epoch(accelerator, model_dict, optimizer, lr_scheduler, train_loader, test_loader=None):
    device = accelerator.device
    texture_controlnet = model_dict["texture_controlnet"]

    train_loss = 0.0
    for i, data in enumerate(train_loader):
        with accelerator.accumulate(texture_controlnet):          
            optimizer.zero_grad()
            loss=sd_cal_loss(data, model_dict, device)
            # 7. 反向传播
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(texture_controlnet.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()

    test_loss = 0.0
    if test_loader is not None:
        for i, data in enumerate(test_loader):
            texture_controlnet.eval()
            loss=sd_cal_loss(data, model_dict, device, True)
            test_loss += loss.item()

    return train_loss, test_loss

                 