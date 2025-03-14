import os
from time import time
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.nn import CrossEntropyLoss, CTCLoss
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from model import get_optimizer, get_scheduler, GeoWhisper
from dataloader import get_dataloader
from evaluate import evaluate

CKPTS_DIR = '/exp/ddegenaro/geowhisper_ckpts'

def get_lengths(model: GeoWhisper, attn_mask: torch.Tensor):
    
    lengths = []
    
    for row in attn_mask:
        if row[-1] == 1:
            lengths.append(len(row))
        else:
            lengths.append(torch.where(row == model.tokenizer.pad_token_id)[0][0].item())
        
    return torch.tensor(lengths)

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
    do_eval: bool = False
):

    model.train()
    
    ignore_index = model.tokenizer.pad_token_id
    ce = CrossEntropyLoss(ignore_index=ignore_index)
    ctc = CTCLoss(blank=ignore_index)
    total_loss = 0.0
    between_log_running_loss = 0.0
    start = time()
    updates_count = 0
    
    while updates_count < max_updates:
        for batch in train_loader:
            
            # breakpoint()
            
            src = batch['inputs'].to(device)
            tgt = model.get_targets(batch['supervisions']).to(device)
            
            optimizer.zero_grad()
            
            cuts = batch['supervisions']['cut']
            
            src_attn_mask = torch.tensor([
                cuts[i].supervisions_audio_mask() for i in range(len(cuts))
            ]).bool().to(device)[:, ::320] # downsampled correctly???
            
            # shift teacher forcing by 1
            tgt_inputs = {
                'input_ids': tgt['input_ids'][:, :-1],
                'attention_mask': tgt['attention_mask'][:, :-1]
            }
            output = model(src, tgt_inputs, src_attn_mask=src_attn_mask)
            # print(output.shape[1].unsqueeze())
            # print(tgt_inputs['attention_mask'])
            # exit()
            
            # shift labels by 1
            ce_loss = ce(output.permute(0, 2, 1), tgt['input_ids'][:, 1:])
            
            # input_lengths = (torch.ones(output.shape[0]) * output.shape[1]).int()
            # target_lengths = get_lengths(model, tgt['attention_mask'][:, 1:])
            # ctc_loss = ctc(
            #     output.permute(1, 0, 2),
            #     tgt['input_ids'][:, 1:],
            #     input_lengths = input_lengths,
            #     target_lengths = target_lengths
            # )
            
            # loss = 0.5*ce_loss + 0.5*ctc_loss
            loss = ce_loss
            
            loss.backward()
            
            clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            
            updates_count += 1
            loss_val = loss.item()
            total_loss += loss_val
            between_log_running_loss += loss_val
            
            if updates_count % log_interval == 0:
                total_time = time() - start
                avg_loss = total_loss / updates_count
                msg = f'Iteration {updates_count:06}/{max_updates}'
                msg += f' - Loss: {loss_val:.4f}'
                mean_loss = between_log_running_loss / log_interval
                msg += f' - Running Avg. Loss: {mean_loss:.4f}'
                msg += f' - Avg. Loss: {avg_loss:.4f}'
                msg += f' - Avg. Time: {total_time/updates_count:.4f}'
                print(msg, flush=True)
                # print('gr decode:', model.greedy_decode(src, device), flush=True)
                print('tf decode:', model.teacher_forced_decode(src, tgt), flush=True)
                print('   target:', model.tokenizer.decode(tgt['input_ids'][0].tolist(), skip_special_tokens=True), flush=True)
                # print('predicted ids:', torch.argmax(output, dim=-1)[0].tolist()[:20], flush=True)
                # print('   target ids:', tgt['input_ids'][:, 1:][0].tolist()[:20], flush=True)
                print()
                between_log_running_loss = 0.0
                
                if not overfit and (updates_count + 1) % 10_000 == 0:
                    print('Saving model...')
                    save_path = os.path.join(CKPTS_DIR, f'model_{updates_count}.pt')
                    torch.save(
                        model.state_dict(),
                        save_path
                    )
                    print(f'Saved to {save_path}', flush=True)
                    if do_eval:
                        print('Evaluating model...', flush=True)
                        evaluate(
                            model,
                            'dev',
                            langs,
                            max_duration,
                            num_buckets,
                            num_mel_bins,
                            device,
                            log_interval,
                            train_steps = str(updates_count)
                        )
                        model.train()

            if updates_count == max_updates:
                break
        
        if not overfit:
            print('Saving model...')
            save_path = os.path.join(CKPTS_DIR, f'model_{updates_count}.pt')
            torch.save(
                model.state_dict(),
                save_path
            )
            print(f'Saved to {save_path}', flush=True)
            if do_eval:
                print('Evaluating model...', flush=True)
                evaluate(
                    model,
                    'dev',
                    langs,
                    max_duration,
                    num_buckets,
                    num_mel_bins,
                    device,
                    log_interval,
                    train_steps = updates_count
                )
                model.train()
            
def main():
    
    langs = [
        'ar_eg', 'en_us', 'es_419', 'fr_fr', 'pt_br', 'ru_ru'
    ]
    
    overfit = False
    print(f'Overfitting: {overfit}', flush=True)
    
    max_duration = 15 # probably way too big, just see what fits
    num_buckets = 30 # 50
    num_mel_bins = 80
    max_updates = 20_000 if overfit else 1_048_576
    audio_length = max_duration * 100 # 10ms frames
    
    lr = 1.5e-4
    betas = (0.9, 0.98)
    eps = 1e-6
    weight_decay = 0.1
    max_grad_norm = 1.0
    warmup_updates = 2048 # 2048
    print(f'Learning rate: {lr}', flush=True)
    print(f'Warmup updates: {warmup_updates}', flush=True)

    d_model = 384 # 512
    nhead = 6 # 8
    num_layers = 4 # 6
    max_length = 128 # 1024
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    
    model = GeoWhisper(
        d_model,
        nhead,
        num_layers,
        max_length,
        audio_length,
        num_mel_bins,
        AutoTokenizer.from_pretrained('openai/whisper-large-v3')
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
    
    train_loader = get_dataloader(
        max_duration,
        num_buckets,
        num_mel_bins,
        overfit,
        lang = 'en_us' if overfit else 'all'
    )
    
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
        log_interval = 20
    )
    
    if not overfit:
        print('Saving final model...', flush=True)
        torch.save(
            model.state_dict(),
            os.path.join(CKPTS_DIR, f'model_{max_updates}.pt')
        )
        print('Done!', flush=True)
        print('Evaluation...', flush=True)
        evaluate(
            model,
            'dev',
            langs,
            max_duration,
            num_buckets,
            num_mel_bins,
            device,
            log_interval = 100,
            train_steps = 'last'
        )
    
    for i, batch in enumerate(train_loader):
            
        src = batch['inputs'][0].unsqueeze(0).to(device)
        tgt = batch['supervisions']['text'][0]
        
        print('\n\n\n', flush=True)
        print('decoded:', model.greedy_decode(src, device), flush=True)
        print('target:', tgt, flush=True)
        
        if i == 1:
            break
    
if __name__ == '__main__':
    main()