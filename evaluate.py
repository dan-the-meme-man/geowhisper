# CER, MER

import subprocess
import unicodedata
import argparse
import os
from time import time

import torch
from transformers import AutoTokenizer

from model import GeoWhisper
from dataloader import get_dataloader

cmn_range = (0x4E00, 0x9FFF)

# https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/scripts/utils/evaluate_asr.sh
# https://github.com/espnet/espnet/blob/master/egs2/fleurs/asr1/local/score_lang_id.py

# TRN FORMAT:
# https://troylee2008.blogspot.com/2010/03/using-sclite.html
# https://sources.debian.org/data/main/s/sctk/2.4.10-20151007-1312Z%2Bdfsg2-3.1~deb10u1/doc/infmts.htm#trn_fmt_name_0

def cer(split: str = 'dev', lang: str = 'en_us'):
    """Calculate Character Error Rate (CER)"""
    
    write_dir = f'results/{lang}_{split}'
    os.makedirs(write_dir, exist_ok=True)
    
    subprocess.run([
        '/home/hltcoe/ddegenaro/SCTK/bin/sclite',
        '-r', 'ref.trn',
        'trn',
        '-h', 'hyp.trn',
        'trn',
        '-i', 'rm',
        '-o', 'all'
    ], cwd=write_dir)
    
    print(f'Wrote CER results to {write_dir}', flush=True)

def evaluate(
    model: GeoWhisper,
    split: str,
    langs: list,
    max_duration: int,
    num_buckets: int,
    num_mel_bins: int,
    overfit: bool,
    device: torch.device,
    log_interval: int = 100
):

    model.eval()
    
    start = time()
    
    for lang in langs:
        
        loader = get_dataloader(
            max_duration,
            num_buckets,
            num_mel_bins,
            overfit,
            split=split,
            lang=lang
        )
    
        ref_path = f'results/{lang}_{split}/ref.trn'
        if not os.path.exists(ref_path):
            print(f'Creating reference file for {lang} {split}...', flush=True)
            os.makedirs(f'results/{lang}_{split}', exist_ok=True)
            with open(
                f'results/{lang}_{split}/ref.trn', 'w+', encoding='utf-8'
            ) as ref_file:
                for i, batch in enumerate(loader):
                    tgt = batch['supervisions']
                    id_str = batch['supervisions']['cut'][0].supervisions[0].id
                    ref_file.write(tgt['text'][0].strip() + f' ({id_str})\n')
        else:
            print(f'Reference file {ref_path} already exists. Skipping...', flush=True)
        
        print(f'Creating hypothesis file for {lang} {split}...', flush=True)
        hyp_path = f'results/{lang}_{split}/hyp.trn'
        with open(
            hyp_path, 'w+', encoding='utf-8'
        ) as hyp_file:
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    
                    src = batch['inputs'].to(device)
                    output = model.greedy_decode(src)
                    id_str = batch['supervisions']['cut'][0].supervisions[0].id
                    hyp_file.write(output.strip() + f' ({id_str})\n')
                    
                    if (i+1) % log_interval == 0:
                        total_time = time() - start
                        msg = f'Iteration {i:06}/{len(loader)}'
                        msg += f' - Avg. Time: {total_time/(i+1):.4f}'
                        print(msg, flush=True)
        print(f'Wrote hypothesis file to {hyp_path}', flush=True)
        cer(f'results/{split}_{lang}_hyp.trn', f'results/{split}_{lang}_ref.trn', split, lang)
                
def main(split: str, lang: str, updates_count: int):
    
    overfit = False
    
    max_duration = 30 # probably way too big, just see what fits
    num_buckets = 50
    num_mel_bins = 80
    max_updates = 100 if overfit else 1_048_576
    audio_length = max_duration * 100 # 10ms frames

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
    model.load_state_dict(torch.load(f'models/model_{updates_count}.pt'))
    
    loader = get_dataloader(
        max_duration,
        num_buckets,
        num_mel_bins,
        overfit,
        split=split,
        lang=lang
    )
    
    evaluate(
        model,
        split,
        lang,
        loader,
        device,
        log_interval = 1 if overfit else 100
    )
    
if __name__ == '__main__':
    
    langs = ['ar_eg', 'en_us', 'es_419', 'fr_fr', 'pt_br', 'ru_ru']
    splits = ['dev', 'test']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='dev', choices=splits)
    parser.add_argument('--lang', type=str, default='en_us', choices=langs)
    parser.add_argument('--updates_count', type=int, default=50_000)
    
    args = parser.parse_args()
    split = args.split
    lang = args.lang
    updates_count = args.updates_count
    
    main(split, lang, updates_count)