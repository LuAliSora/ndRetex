import torch
from torch.amp import autocast

def uvRex_train_one_epoch(model, optimizer, scaler, device, train_loader, test_loader=None):

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast():
            x=data.to(device)
            y=model(data)
            loss=loss_fn(x, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



    return model