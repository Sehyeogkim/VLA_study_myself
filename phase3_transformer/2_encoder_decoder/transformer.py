'''
I think we had better put the 
- transformer block
    ->Multihead attention 

in here pleases.    
'''

from configuration import Config
import math
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, cfg):
        '''
        d_model = hidden states dimension
        nhead = number of heads
        dropout = dropout rate
        '''
        super().__init__()
        assert cfg.d_model % cfg.nhead == 0
        self.H = cfg.nhead
        self.Dh = cfg.d_model // cfg.nhead # d_model = Dimenison of hidden states
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
    
    def _split(self, t):
        '''split for the multi head attention.'''
        # [B, T, D] -> [B, H, T, Dh]
        B, T, _ = t.shape
        return t.view(B, T, self.H, self.Dh).transpose(1, 2)
    
    def _merge(self, t):
        '''merge for the multi head attention. -> single head tensor'''
        # [B, H, T, Dh] -> [B, T, D]
        B, _, T, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, self.H * self.Dh)
    
    def forward(self, x, attn_mask, key_padding_mask):
        
        q,k,v = self.W_q(x), self.W_k(x), self.W_v(x)
        q,k,v = self._split(q), self._split(k), self._split(v)
        #all of them [B, H, T, Dh]

        # per head: [B, H, T, T]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.Dh) # [B, H, T, T]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))  # [T,T] broadcasts
        
        # Padding + Attention Mask.
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        
        # We put drop out in here. 
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = self._merge(weights @ v) # [B, H, T, Dh] -> [B, T, D]
        return self.W_o(out) # [B, T, D] -> [B, T, D]

class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.LN1 = nn.LayerNorm(cfg.d_model)
        self.LN2 = nn.LayerNorm(cfg.d_model)
        self.FFN = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model))
        
        # call the instnace for the attention module will be diefnie further classes.
        self.self_attention = MultiHeadSelfAttention(cfg)

    
    def forward(self, x, attn_mask, key_padding_mask):
        '''
        input
        - x: hidden states
        - attn_mask: casual mask
        - key_padding_mask: pad mask
        output:
        - hidden states
        '''

        x_pre = x
        x = self.LN1(x)
        x = self.self_attention(x, attn_mask, key_padding_mask)
        x = x + x_pre

        x_pre = x
        x = self.LN2(x)
        x = self.FFN(x)
        x = x + x_pre
        return x




# this is additionall for th encdoer - decoder blck.


class MultiHeadCrossAttention(nn.Module):
    '''
    reference by attention is all you need figure1.
    '''
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.nhead == 0
        self.H = cfg.nhead
        self.Dh = cfg.d_model // cfg.nhead
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
    
    def _split(self, t):
        '''split for the multi head attention.'''
        # [B, T, D] -> [B, H, T, Dh]
        B, T, _ = t.shape
        return t.view(B, T, self.H, self.Dh).transpose(1, 2)
    
    def _merge(self, t):
        '''merge for the multi head attention. -> single head tensor'''
        # [B, H, T, Dh] -> [B, T, D]
        B, _, T, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, self.H * self.Dh)
    
    def forward(self, x, memory, enc_key_padding_mask):
        '''
        input
        - x: decoder hidden states [B, T_dec, D]
        - memory: encoder hidden states [B, T_enc, D]
        - enc_key_padding_mask: [B, T_enc], True = pad token on encoder keys
        output:
        - hidden states [B, T_dec, D]
        '''

        q, k, v = self.W_q(x), self.W_k(memory), self.W_v(memory)
        q, k, v = self._split(q), self._split(k), self._split(v)
        # q: [B, H, T_dec, Dh], k/v: [B, H, T_enc, Dh]

        # [B, H, T_dec, T_enc] — no causal mask; each decoder position may attend all encoder keys
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.Dh)

        if enc_key_padding_mask is not None:
            scores = scores.masked_fill(enc_key_padding_mask[:, None, None, :], float("-inf"))

        # We put drop out in here. 
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = self._merge(weights @ v) # [B, H, T, Dh] -> [B, T, D]
        return self.W_o(out) # [B, T, D] -> [B, T, D]

class DecoderTransformerBlock(nn.Module):
    '''
    referece by attention is all you need figure1.
    '''

    def __init__(self, cfg):
        super().__init__()
        self.LN1 = nn.LayerNorm(cfg.d_model)
        self.LN2 = nn.LayerNorm(cfg.d_model)
        self.LN3 = nn.LayerNorm(cfg.d_model) # one more.
        self.FFN = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model))
        
        # call the instnace for the attention module will be diefnie further classes.
        self.self_attention = MultiHeadSelfAttention(cfg)
        self.cross_attention = MultiHeadCrossAttention(cfg)
    
    def forward(self, x, memory, causal_attn_mask, dec_key_padding_mask, enc_key_padding_mask):
        '''
        Pre-LN (same pattern as TransformerBlock / encoder):
        LN → sublayer → residual, for masked SA, cross-attn, FFN.
        '''
        x_pre = x
        x = self.LN1(x)
        x = self.self_attention(x, causal_attn_mask, dec_key_padding_mask)
        x = x + x_pre

        x_pre = x
        x = self.LN2(x)
        x = self.cross_attention(x, memory, enc_key_padding_mask)
        x = x + x_pre

        x_pre = x
        x = self.LN3(x)
        x = self.FFN(x)
        x = x + x_pre
        return x