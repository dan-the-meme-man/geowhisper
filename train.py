from time import time

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from transformers import GPT2Tokenizer

from model import get_optimizer, get_scheduler, GeoWhisper
from dataloader import get_dataloader

def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    max_updates,
    max_grad_norm,
    log_interval=100
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)
    model.train()
    
    ignore_index = model.tokenizer.pad_token_id
    criterion = CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    start = time()
    updates_count = 0
    
    while updates_count < max_updates:
        for batch in train_loader:
            
            src = batch['inputs'].to(device)
            tgt = model.get_targets(batch['supervisions']).to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.permute(0, 2, 1), tgt['input_ids'])
            
            loss.backward()
            
            clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            )
            
            optimizer.step()
            
            scheduler.step()
            
            updates_count += 1
            total_loss += loss.item()
            
            if updates_count % log_interval == 0:
                total_time = time() - start
                msg = f'Iteration {updates_count:06}/{max_updates}'
                msg += f' - Avg. Loss: {total_loss/updates_count:.4f}'
                msg += f' - Avg. Time: {total_time/updates_count:.4f}'
                print(msg)

            if updates_count == max_updates:
                break
            
def main():
    
    overfit = True
    
    max_duration = 30 # probably way too big, just see what fits
    num_buckets = 50
    num_mel_bins = 80
    max_updates = 1_048_576
    audio_length = max_duration * 100 # 10ms frames
    
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
        max_length,
        audio_length,
        num_mel_bins,
        GPT2Tokenizer.from_pretrained('gpt2')
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
        warmup_updates,
        max_updates
    )
    
    train(
        model,
        get_dataloader(max_duration, num_buckets, num_mel_bins, overfit),
        optimizer,
        scheduler,
        max_updates,
        max_grad_norm
    )
    
if __name__ == '__main__':
    main()