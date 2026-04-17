import torch
from torch.amp import autocast

from accelerate import Accelerator

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
    if test_loader!=None:
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


def sd_train_one_epoch(accelerator, uvRex_model, vae, noise_scheduler, tokenizer, text_encoder, normal_controlnet, texture_controlnet, unet, train_loader, test_loader=None):
    device = accelerator.device
    for i, data in enumerate(train_loader):
        with accelerator.accumulate(texture_controlnet):          
            ori, mask, normal, tex, prompt = data
            rough=uvRex_predict(ori, mask, normal, tex, uvRex_model, device)
            print(prompt)
            
            # with torch.no_grad():
            #     # VAE编码
            #     latents = vae.encode(rough).latent_dist.sample()
            #     latents = latents * vae.config.scaling_factor
                
            #     # 添加噪声（img2img的核心）
            #     # 随机选择时间步
            #     timesteps = torch.randint(
            #         0, noise_scheduler.config.num_train_timesteps,
            #         (latents.shape[0],), device=latents.device
            #     ).long()
                
            #     # 添加噪声
            #     noise = torch.randn_like(latents)
            #     noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            #     # 2. 编码prompt
            #     text_inputs = tokenizer(
            #         prompt,
            #         padding="max_length",
            #         max_length=tokenizer.model_max_length,
            #         truncation=True,
            #         return_tensors="pt"
            #     )
            #     encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

            #     # 法线ControlNet（冻结，不计算梯度）
            #     with torch.no_grad():
            #         normal_down_block_res_samples, normal_mid_block_res_sample = normal_controlnet(
            #             noisy_latents,
            #             timesteps,
            #             encoder_hidden_states=encoder_hidden_states,
            #             controlnet_cond=normal,
            #             conditioning_scale=1.0,
            #             return_dict=False,
            #         )
                    
            #     # 纹理ControlNet（训练）
            #     texture_down_block_res_samples, texture_mid_block_res_sample = texture_controlnet(
            #         noisy_latents,
            #         timesteps,
            #         encoder_hidden_states=encoder_hidden_states,
            #         controlnet_cond=tex,
            #         conditioning_scale=1.0,
            #         return_dict=False,
            #     )

            #     mid_block_res_sample = normal_mid_block_res_sample + texture_mid_block_res_sample
            #     # 5. UNet前向传播（预测噪声）
            #     with torch.no_grad():
            #         model_pred = unet(
            #             noisy_latents,
            #             timesteps,
            #             encoder_hidden_states=encoder_hidden_states,
            #             down_block_additional_residuals=down_block_res_samples,
            #             mid_block_additional_residual=mid_block_res_sample,
            #         ).sample
        

                 