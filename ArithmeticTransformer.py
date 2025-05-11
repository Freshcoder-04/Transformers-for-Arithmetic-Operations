import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        self_attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class ArithmeticTransformer(nn.Module):
    def __init__(self, 
                 input_vocab_size,
                 output_vocab_size,
                 d_model=256,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=1024,
                 dropout=0.1,
                 max_seq_length=20,
                 pad_token_id=0):
        super().__init__()
        
        # Embeddings and Positional Encoding
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
    def create_causal_mask(self, size):
        # Create a causal mask for the decoder
        # Shape: [size, size]
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)
        
    def create_padding_mask(self, x):
        # Create a padding mask
        # Shape: [batch_size, seq_len]
        return (x == self.pad_token_id)
        
    def forward(self, src, tgt):
        # Create masks
        src_key_padding_mask = self.create_padding_mask(src)  # [batch_size, src_len]
        tgt_key_padding_mask = self.create_padding_mask(tgt)  # [batch_size, tgt_len]
        tgt_mask = self.create_causal_mask(tgt.size(1))      # [tgt_len, tgt_len]
        
        # Encoder
        src_embedded = self.dropout(self.pos_encoding(self.input_embedding(src) * math.sqrt(self.d_model)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, key_padding_mask=src_key_padding_mask)
            
        # Decoder
        tgt_embedded = self.dropout(self.pos_encoding(self.output_embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(
                dec_output, 
                enc_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
        # Output layer
        output = self.output_layer(dec_output)
        return output

    def generate(self, src, max_length, start_token_id, end_token_id):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # Encoder
            src_kpm = self.create_padding_mask(src)
            enc_out = self.pos_encoding(self.input_embedding(src) * math.sqrt(self.d_model))
            for enc in self.encoder_layers:
                enc_out = enc(enc_out, key_padding_mask=src_kpm)

            # Decoder init
            dec_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
            finished  = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_length):
                tgt_mask = self.create_causal_mask(dec_input.size(1))
                tgt_kpm  = self.create_padding_mask(dec_input)

                dec_out = self.pos_encoding(self.output_embedding(dec_input) * math.sqrt(self.d_model))
                for dec in self.decoder_layers:
                    dec_out = dec(dec_out, enc_out,
                                tgt_mask=tgt_mask,
                                memory_key_padding_mask=src_kpm,
                                tgt_key_padding_mask=tgt_kpm)

                logits      = self.output_layer(dec_out[:, -1:])        # [B, 1, vocab]
                next_token  = torch.argmax(logits, dim=-1)              # [B, 1]
                dec_input   = torch.cat([dec_input, next_token], 1)

                finished |= (next_token.squeeze(1) == end_token_id)
                if finished.all():                                      # wait for *all* rows
                    break

            return dec_input