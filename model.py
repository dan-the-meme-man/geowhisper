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

def get_scheduler(optimizer, warmup_steps, total_steps):
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps # linear warmup
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps)) # linear decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class ConvBlock(nn.Module):
    
    def __init__(self, num_mel_bins, d_model):
        super(ConvBlock, self).__init__()
        
        # convs and projection
        self.conv1 = nn.Conv1d(
            num_mel_bins,
            d_model,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.conv2 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size = 3,
            stride = 2,
            padding = 1
        )
        
    def forward(self, x):
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        
        return x
    
class SinusoidalPositionalEncoding(nn.Module):
    
    def __init__(self, max_length, d_model):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        self.register_buffer(
            'positional_encoding',
            self._generate_positional_encoding(max_length, d_model)
        )
        
    def _generate_positional_encoding(self, max_length, d_model):
        encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]
    
class LearnedPositionalEncoding(nn.Module):
    
    def __init__(self, max_length, d_model):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.positional_encoding = nn.Parameter(
            torch.randn(max_length, d_model).unsqueeze(0)
        )
        
    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]

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
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.max_length = max_length
        
        self.convs = ConvBlock(num_mel_bins, d_model)
        
        # input embedding
        self.input_embedding = nn.Embedding(len(self.tokenizer), d_model)
        
        self.spe = SinusoidalPositionalEncoding(audio_length, d_model)
        self.lpe = LearnedPositionalEncoding(max_length, d_model)
        
        # transformer        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # output projection layer
        self.output_projection = nn.Linear(d_model, len(self.tokenizer))
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Gaussian fan-in for linear layers
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):  # If learned positional encoding uses an embedding
                nn.init.xavier_normal_(m.weight)
        
        # init LPE with small normal noise
        nn.init.normal_(self.lpe.positional_encoding, mean=0, std=0.02)

    def forward(self, src, tgt, src_attn_mask=None):
        
        # batch, channels, seq_len
        src = src.permute(0, 2, 1)
        # print('input', src.shape) # batch, num_mel_bins, audio length
        # print('target', tgt['input_ids'].shape) # batch, text length
        
        # convs
        src = self.convs(src).permute(0, 2, 1)
        # print('convs', src.shape) # batch, d_model, audio length / 2
        
        # add sinusoidal positional encoding
        src_embedded = self.spe(src)
        # print('src plus PE', src_embedded.shape) # batch, d_model, audio length / 2
        
        text = tgt['input_ids']
        tgt_attn_mask = tgt['attention_mask']
        
        # learned positional encoding
        tgt_embedded = self.lpe(self.input_embedding(text))
        # print('tgt plus LPE', tgt_embedded.shape) # batch, max_length, d_model
        
        # source key padding mask
        if src_attn_mask is not None:
            src_key_padding_mask = (src_attn_mask == 0)
            # print('src key padding mask', src_key_padding_mask.shape) # batch, audio length
            # print(src_key_padding_mask)
        else:
            src_key_padding_mask = None
        
        # target key padding mask
        tgt_key_padding_mask = (tgt_attn_mask == 0)
        # print('tgt key padding mask', tgt_key_padding_mask.shape) # batch, max_length
        # print(tgt_key_padding_mask)
        
        # causal mask
        causal_mask = self.transformer.generate_square_subsequent_mask(
            tgt_embedded.shape[1]
        ).unsqueeze(0).expand(tgt_embedded.shape[0], -1, -1).to(src.device)
        # print('causal mask', causal_mask.shape) # text length, text length
        
        # transformer
        trf_out = self.transformer(
            src_embedded,
            tgt_embedded,
            src_mask=None,
            tgt_mask=causal_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )
        # print('transformer', trf_out.shape) # batch, max_length, d_model
        
        # output projection
        output = self.output_projection(trf_out)
        # print('output', output.shape) # batch, max_length, vocab_size
        
        return output
    
    # def forward(self, src, tgt):
        
    #     # batch, channels, seq_len
    #     src = src.permute(0, 2, 1)
    #     # print('input', src.shape) # batch, num_mel_bins, audio frames
    #     # print('target', tgt['input_ids'].shape) # batch, text length
        
    #     # convs
    #     src = F.gelu(self.conv1(src))
    #     # print('1st conv', src.shape) # batch, d_model/4, audio length + 2
    #     src = F.gelu(self.conv2(src))
    #     # print('2nd conv', src.shape) # batch, d_model, (audio length / 2) + 2
        
    #     proj = self.proj_layer(src.T)
        
    #     # output projection
    #     output = self.output_projection(proj)
    #     # print('output', output.shape) # batch, max_length, vocab_size
        
    #     return output
    
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
    
    def greedy_decode(self, src, device):

        self.eval()

        tgt_input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(device)
        
        with torch.no_grad():

            for _ in range(self.max_length):
                tgt = {
                    'input_ids': tgt_input_ids.to(device),
                    'attention_mask': torch.ones_like(tgt_input_ids).to(device)
                }

                # forward pass
                with torch.no_grad():
                    logits = self(src, tgt)
                    # print(logits.shape)

                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                tgt_input_ids = torch.cat(
                    [tgt_input_ids, next_token_id.unsqueeze(1)],
                    dim=1
                )

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        
        #print(logits[0, :16, :5])
        
        output_text = self.tokenizer.decode(
            tgt_input_ids[0].tolist(),
            skip_special_tokens=True
        )
        
        self.train()

        return output_text
    
    def batched_greedy_decode(self, src, device, max_length):
        
        self.eval()
        
        batch_size = src.shape[0]
        tgt = torch.fill(self.tokenizer.bos_token_id, (batch_size, 1)).to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self(src, {'input_ids': tgt})
                tgt = torch.cat((tgt, output.argmax(dim=-1)[:, -1].unsqueeze(1)), dim=1)
            
        self.train()
        
        return tgt[:, 1:]
    
    def teacher_forced_decode(self, src, tgt):
        
        self.eval()
        
        output = None
        
        with torch.no_grad():
            output = self(src, tgt)
        
        self.train()
        
        return self.tokenizer.batch_decode(output.argmax(dim=-1), skip_special_tokens=True)[0]

if __name__ == "__main__":
    model = GeoWhisper(512, 8, 6, 1024, 2048)
    src = torch.randn(10, 1, 80)
    tgt = torch.randn(10, 1, 80)
    model(src, tgt)