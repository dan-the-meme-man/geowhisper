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

from torch.utils.data import DataLoader

from lhotse import Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset, OnTheFlyFeatures

def get_dataloader(max_duration, num_buckets, num_mel_bins, overfit=False):
    
    train_dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(
            Fbank(FbankConfig(num_mel_bins=num_mel_bins))
        ),
        cut_transforms=[],
        input_transforms=[],
        return_cuts=False
    )

    if overfit:
        manifest = load_manifest_lazy(TEST_PATH).subset(first=100)
    else:
        manifest = load_manifest_lazy(TEST_PATH)
    
    train_sampler = DynamicBucketingSampler(
        manifest.filter(lambda c: c.duration <= 30).pad(duration=max_duration),
        max_duration=max_duration, # per GPU, in seconds
        shuffle=True,
        num_buckets=num_buckets, # make small if small data - can cause errors
        # buffer_size=self.args.num_buckets * 2000,
        # shuffle_buffer_size=self.args.num_buckets * 5000,
        # drop_last=self.args.drop_last,
    )

    return DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=None,
        num_workers=0 # 4
    )
    
if __name__ == "__main__":
    dl = get_dataloader(10, 100, 80)
    for batch in dl:
        #print(batch)
        print(batch['inputs'].shape)
        #break