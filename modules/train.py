import torch
from torch.amp import autocast

from modules.dataPr import tensor2img

def uvRex_train_one_epoch(model, optimizer, scaler, dataAug, device, train_loader, test_loader=None):
    model.train()
    model.to(device)

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast(device.type):
            x_aug=dataAug(data).to(device)
            y=model(x_aug)
            loss=loss_fn(x, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    model.cpu()
    return model