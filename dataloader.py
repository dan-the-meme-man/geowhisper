TEST_PATH = '/expscratch/mwiesner/scale23/scale2023/icefall/tools/icefall/egs/scale24/ASR/data/manifests/english/cuts_fleurs_dev.jsonl.gz'

# /expscratch/mwiesner/geolocation/icefall/egs/radio/geolocation/w2v2_angular_distance/geolocation_datamodule.py

# geolocation repo: https://github.com/geolocation-from-speech

"""
/export/common/data/corpora/ASR/openslr/SLR12/LibriSpeech
PUT THIS IN prepare.sh as path/to/LibriSpeech
icefall/blob/geolocation/egs/librispeech/ASR/prepare.sh
Run zipformer/train.py:
icefall/blob/geolocation/egs/librispeech/ASR/zipformer/train.py
"""

import os

from torch.utils.data import DataLoader

from lhotse import Fbank, FbankConfig, load_manifest_lazy, CutSet
from lhotse.manipulation import combine
from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset, OnTheFlyFeatures

FLEURS_PATH = '/exp/ddegenaro/fleurs'

def get_dataloader(
    max_duration: float,
    num_buckets: int,
    num_mel_bins: int,
    overfit: bool = False,
    split: str = 'train',
    lang: str = 'all'
):
    
    assert split in ['train', 'dev', 'test']
    
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(
            Fbank(FbankConfig(num_mel_bins=num_mel_bins))
        ),
        cut_transforms=[],
        input_transforms=[],
        return_cuts=True
    )
    if lang == 'all':
        langs = ['ar_eg', 'en_us', 'es_419', 'fr_fr', 'pt_br', 'ru_ru']
    else:
        langs = [lang]
    fnames = [
        f'fleurs-{lang}_recordings_{split}.jsonl.gz'
        for lang in langs
    ]
    paths = [
        os.path.join(FLEURS_PATH, langs[i], fnames[i])
        for i in range(len(langs))
    ]
    recordings = [
        load_manifest_lazy(path)
        for path in paths
    ]
    recordings = combine(recordings)
    supervisions = [
        load_manifest_lazy(path.replace('recordings', 'supervisions'))
        for path in paths
    ]
    supervisions = combine(supervisions)
    assert len(recordings) == len(supervisions)
        
    manifest = CutSet.from_manifests(recordings, supervisions)
    
    if overfit:
        manifest = manifest.subset(first=500)
        
    # trim silences
    manifest = manifest.trim_to_supervisions()
    # stats = manifest.compute_global_feature_stats()
    # means = stats['norm_means']
    # stds = stats['norm_stds']
    
    sampler = DynamicBucketingSampler(
        manifest.filter(lambda c: c.duration <= max_duration).pad(duration=max_duration),
        max_duration=max_duration, # per GPU, in seconds
        shuffle=True,
        num_buckets=num_buckets, # make small if small data - can cause errors
        # buffer_size=self.args.num_buckets * 2000,
        # shuffle_buffer_size=self.args.num_buckets * 5000,
        # drop_last=self.args.drop_last,
    )

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=0 # 4
    )
    
if __name__ == "__main__":
    dl = get_dataloader(
        30,
        50,
        80,
        True,
        'dev',
        'all'
    )
    for batch in dl:
        #print(batch)
        print(batch['inputs'].shape)
        #break