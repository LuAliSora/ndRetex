import torch
from torch.amp import autocast

from modules.dataPr import tensor2img

def uvRex_train_one_epoch(model, optimizer, scaler, dataAug, device, train_loader, test_loader=None):


    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast(device.type):
            x_aug=dataAug(data).to(device)
            # tensor2img(x_aug[0].squeeze(0), f"output/res{i}.jpg")
            y=model(x_aug)
        #     loss=loss_fn(x, y)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
    model.cpu()
    return model