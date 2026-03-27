import torch
from torch.amp import autocast

from modules.dataPr import tensor2img
from modules.utils import uvRex_loss

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
    return train_loss/len(train_loader), test_loss/len(test_loader)