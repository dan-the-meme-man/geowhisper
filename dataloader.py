import torch
from torch.utils.data import DataLoader

class LhotseDataset:
    def __init__(self):
        pass
    
    def __len__(self):
        return 1_000_000
    
    def __getitem__(self, idx):
        
        return (
            torch.randn(16000),
            torch.randint(0, 10, (100,))
        )

def get_dataloader(batch_size):
    return DataLoader(
        LhotseDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )