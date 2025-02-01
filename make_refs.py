import os

from dataloader import get_dataloader

max_duration = 30
num_buckets = 50
num_mel_bins = 80
overfit = False
split = 'test'

langs = ['ar_eg', 'en_us', 'es_419', 'fr_fr', 'pt_br', 'ru_ru']

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