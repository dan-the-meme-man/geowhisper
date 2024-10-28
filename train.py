from time import time

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_

from model import get_optimizer, get_scheduler, GeoWhisper
from dataloader import get_dataloader

def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    max_updates,
    max_grad_norm
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    criterion = CrossEntropyLoss()
    total_loss = 0.0
    start = time()
    epoch = 0
    updates_count = 0
    while updates_count < max_updates:
        print(f'Epoch {epoch + 1}')
        for i, (src, tgt) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            updates_count += 1
            total_loss += loss.item()
            total_time = time() - start
            if i % 100 == 0:
                msg = f'Iteration {i+1:04}/{len(train_loader)}'
                msg += f' - Avg. Loss: {total_loss/(i+1):.4f}'
                msg += f' - Avg. Time: {total_time/(i+1):.4f}'
                print(msg)
            if updates_count == max_updates:
                break
            
def main():
    
    batch_size = 1 # 256
    max_updates = 1_048_576
    
    lr = 1e-3
    betas = (0.9, 0.98)
    eps = 1e-6
    weight_decay = 0.1
    max_grad_norm = 1.0
    warmup_updates = 2048

    d_model = 512
    nhead = 8
    num_layers = 6
    max_length = 1024
    
    model = GeoWhisper(
        d_model,
        nhead,
        num_layers,
        max_length
    )
    
    optimizer = get_optimizer(
        model,
        lr,
        betas,
        eps,
        weight_decay
    )
    scheduler = get_scheduler(
        optimizer,
        warmup_updates
    )
    
    train(
        model,
        get_dataloader(batch_size),
        optimizer,
        scheduler,
        max_updates,
        max_grad_norm
    )
    
if __name__ == '__main__':
    main()