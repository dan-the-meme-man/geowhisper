import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(model, lr, betas, eps, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

def get_scheduler(optimizer, warmup_updates):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda u: 1 - u / warmup_updates
    )

class GeoWhisper(nn.Module):
    
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        max_length
    ):
        super(GeoWhisper, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 3, 1, 2)
        self.conv2 = nn.Conv1d(1, 1, 3, 2, 2)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        
        self.register_buffer(
            'sinusoidal_positional_encoding',
            self._generate_sinusoidal_positional_encoding(max_length, d_model)
        )
        
        self.learned_positional_encoding = nn.Embedding(max_length, d_model)
        
        # gaussian fan-in initialization
        nn.init.xavier_normal_(self.conv1.weight, gain=1.0)
        nn.init.xavier_normal_(self.conv2.weight, gain=1.0)
        nn.init.xavier_normal_(self.learned_positional_encoding.weight, gain=1.0)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)
        
    def _generate_sinusoidal_positional_encoding(self, max_length, d_model):
        encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, src, tgt):
        print('input and target', src.shape, tgt.shape)
        
        # convs
        src = F.gelu(self.conv1(src))
        print('1st conv', src.shape)
        src = F.gelu(self.conv2(src))
        print('2nd conv', src.shape)
        
        # reshape for transformer
        src = src.permute(2, 0, 1)
        seq_len, batch_size, _ = src.shape
        src += self.sinusoidal_positional_encoding[:seq_len, :].unsqueeze(1).expand(-1, batch_size, -1)
        print('src plus SPE', src.shape)
        
        # transformer encoder
        memory = self.transformer_encoder(src)
        print('enc out', memory.shape)
        
        # learned positional encoding
        tgt_seq_len, tgt_batch_size, _ = tgt.shape
        assert tgt_batch_size == batch_size, "Source and target batch sizes must match"
        tgt_pos = torch.arange(tgt_seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        tgt = tgt + self.learned_positional_encoding(tgt_pos).permute(1, 0, 2)
        print('tgt plus LPE', tgt.shape)
        
        # transformer decoder
        output = self.transformer_decoder(tgt, memory)
        print('dec out', output.shape)
        return output
    
    def preprocess(audio_file):
        pass