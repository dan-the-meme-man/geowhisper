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

def get_scheduler(optimizer, warmup_updates, max_updates):
    def lr_lambda(current_update):
        if current_update < warmup_updates:
            return current_update / warmup_updates
        else:
            return max(0.0, 1 - (current_update - warmup_updates) / (max_updates - warmup_updates))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

class GeoWhisper(nn.Module):
    
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        max_length,
        audio_length,
        num_mel_bins,
        tokenizer
    ):
        super(GeoWhisper, self).__init__()
        
        # convs and projection
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, 3, 1, 1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, 2, 1)
        
        # transformer        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        
        # text embedding and output projection, shared weights
        self.input_embedding = nn.Embedding(tokenizer.vocab_size, d_model)
        self.output_projection = nn.Linear(d_model, tokenizer.vocab_size)
        self.input_embedding.weight = self.output_projection.weight
        
        self.register_buffer(
            'sinusoidal_positional_encoding',
            self._generate_sinusoidal_positional_encoding(int(audio_length / 2), d_model).T.unsqueeze(0)
        )
        
        self.learned_positional_encoding = nn.Embedding(max_length, d_model)
        
        # gaussian fan-in initialization
        nn.init.xavier_normal_(self.conv1.weight, gain=1.0)
        nn.init.xavier_normal_(self.conv2.weight, gain=1.0)
        nn.init.xavier_normal_(self.learned_positional_encoding.weight, gain=1.0)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)
        
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.max_length = max_length
        
    def _generate_sinusoidal_positional_encoding(self, max_length, d_model):
        encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, src, tgt):
        
        # batch, channels, seq_len
        src = src.permute(0, 2, 1)
        # print('input', src.shape) # batch, num_mel_bins, audio frames
        # print('target', tgt['input_ids'].shape) # batch, text length
        
        # convs
        src = F.gelu(self.conv1(src))
        # print('1st conv', src.shape) # batch, d_model/4, audio length + 2
        src = F.gelu(self.conv2(src))
        # print('2nd conv', src.shape) # batch, d_model, (audio length / 2) + 2
        
        # add sinusoidal positional encoding
        src_embedded = src + self.sinusoidal_positional_encoding
        src_embedded = src_embedded.permute(0, 2, 1)
        # print('src plus SPE', src_embedded.shape) # batch, max_length, d_model
        
        text = tgt['input_ids']
        attn_mask = tgt['attention_mask']
        
        # learned positional encoding
        tgt_embedded = self.input_embedding(text)
        tgt_embedded = self.input_embedding(
            text
        ) + self.learned_positional_encoding(
            torch.arange(text.shape[1]).to(text.device)
        )
        # print('tgt plus LPE', tgt_embedded.shape) # batch, max_length, d_model
        
        # target key padding mask
        tgt_key_padding_mask = (attn_mask == 0)
        # print('tgt key padding mask', tgt_key_padding_mask.shape) # batch, max_length
        
        # causal mask
        causal_mask = torch.triu(
            torch.ones(text.shape[1], text.shape[1]),
            diagonal=1
        ).bool().to(text.device)
        
        # transformer
        trf_out = self.transformer(
            src_embedded,
            tgt_embedded,
            src_mask=None,
            tgt_mask=causal_mask,
            memory_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )
        # print('transformer', trf_out.shape) # batch, max_length, d_model
        
        # output projection
        output = self.output_projection(trf_out)
        # print('output', output.shape) # batch, max_length, vocab_size
        
        return output
    
    def get_targets(self, supervisions):
        """TODO"""
        # sequence_idx = supervisions['sequence_idx']
        # start_frame = supervisions['start_frame']
        # num_frames = supervisions['num_frames']
        
        return self.tokenizer(
            supervisions['text'],
            return_tensors='pt',
            padding='longest',
            max_length=self.max_length,
            truncation=True
        )
        
    def save(self):
        torch.save(self.state_dict(), 'models/plain.pt')
        
    def greedy_decode(self, src, device='cuda'):

        self.eval()

        tgt_input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(device)

        for _ in range(model.max_length):
            tgt = {
                'input_ids': tgt_input_ids.to(device),
                'attention_mask': torch.ones_like(tgt_input_ids).to(device)
            }

            # forward pass
            with torch.no_grad():
                logits = model(src, tgt)

            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            tgt_input_ids = torch.cat(
                [tgt_input_ids, next_token_id.unsqueeze(1)],
                dim=1
            )

            if next_token_id.item() == model.end_id:
                break

        output_text = model.tokenizer.decode(
            tgt_input_ids[0].tolist(),
            skip_special_tokens=True
        )

        return output_text
    
    def teacher_forced_decode(self, src, tgt):
        output = self(src, tgt)
        return self.tokenizer.batch_decode(output.argmax(dim=-1))

if __name__ == "__main__":
    model = GeoWhisper(512, 8, 6, 1024, 2048)
    src = torch.randn(10, 1, 80)
    tgt = torch.randn(10, 1, 80)
    model(src, tgt)