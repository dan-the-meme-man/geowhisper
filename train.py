import os
from time import time

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from model import get_optimizer, get_scheduler, GeoWhisper
from dataloader import get_dataloader
from evaluate import evaluate

CKPTS_DIR = '/exp/ddegenaro/geowhisper_ckpts'

def train(
    model: GeoWhisper,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    max_updates: int,
    max_grad_norm: float,
    device: torch.device,
    langs: list,
    max_duration: float,
    num_buckets: int,
    num_mel_bins: int,
    overfit: bool,
    log_interval: int = 100,
    ckpt_interval: int = 10_000
):

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
                print(msg, flush=True)
                
            if updates_count % ckpt_interval == 0:
                print('Saving model...')
                save_path = os.path.join(CKPTS_DIR, f'model_{updates_count}.pt')
                torch.save(
                    model.state_dict(),
                    save_path
                )
                print(f'Saved to {save_path}')
                print('Evaluating model...')
                evaluate(
                    model,
                    'dev',
                    langs,
                    max_duration,
                    num_buckets,
                    num_mel_bins,
                    overfit,
                    device,
                    log_interval
                )

            if updates_count == max_updates:
                break
            
def main():
    
    langs = [
        'ar_eg', 'en_us', 'es_419', 'fr_fr', 'pt_br', 'ru_ru'
    ]
    
    overfit = True
    
    max_duration = 30 # probably way too big, just see what fits
    num_buckets = 50
    num_mel_bins = 80
    max_updates = 1_000 if overfit else 1_048_576
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    
    model = GeoWhisper(
        d_model,
        nhead,
        num_layers,
        max_length,
        audio_length,
        num_mel_bins,
        AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
    )
    model.to(device)
    
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
    
    train_loader = get_dataloader(max_duration, num_buckets, num_mel_bins, overfit)
    
    train(
        model,
        train_loader,
        optimizer,
        scheduler,
        max_updates,
        max_grad_norm,
        device,
        langs,
        max_duration,
        num_buckets,
        num_mel_bins,
        overfit,
        log_interval = 1 if overfit else 100,
        ckpt_interval = 100 if overfit else 50_000
    )
    
    print('Saving final model...', flush=True)
    torch.save(model.state_dict(), 'models/model_final.pt')
    print('Done!', flush=True)
    print('Evaluation...', flush=True)
    evaluate(
        model,
        'dev',
        langs,
        train_loader,
        max_updates,
        device
    )
    
    for i, batch in enumerate(train_loader):
            
        src = batch['inputs'][0].unsqueeze(0).to(device)
        tgt = batch['supervisions']['text'][0]
        
        print('\n\n\n', flush=True)
        print('decoded:', model.greedy_decode(src), flush=True)
        print('target:', tgt, flush=True)
        
        if i == 1:
            break
    
if __name__ == '__main__':
    main()