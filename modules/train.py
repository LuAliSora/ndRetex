import torch
from torch.amp import autocast

from modules.dataPr import tensor2img

def uvRex_train_one_epoch(model, optimizer, scaler, dataAug, device, train_loader, test_loader=None):

    train_loss = 0.0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast(device.type):
            x_aug=dataAug(data).to(device)
            y=model(x_aug)
            loss=loss_fn(x, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    if test_loader!=None:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():  # 禁用梯度计算，节省内存和计算
            for i, data in enumerate(test_loader):
                with autocast(device.type):
                    x_aug=dataAug(data).to(device)
                    # tensor2img(x_aug[0].squeeze(0), f"output/res{i}.jpg")
                    y=model(x_aug)
                    loss=loss_fn(x, y)

                test_loss += loss.item()

    return model