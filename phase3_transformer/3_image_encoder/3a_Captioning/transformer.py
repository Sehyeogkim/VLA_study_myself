import math

import torch

from configuration import BlockConfig, Config

# Single head attention
class SinlgeHeadAttention(torch.nn.Module):
    def __init__(self, block: BlockConfig):
        super().__init__()
        self.block = block
        d = block.d_model

        self.W_q = torch.nn.Linear(d, d)
        self.W_k = torch.nn.Linear(d, d)
        self.W_v = torch.nn.Linear(d, d)
        self.W_o = torch.nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        attn_weights = q @ k.transpose(-2, -1) / math.sqrt(self.block.d_model)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        x = attn_weights @ v
        return self.W_o(x)

# Self attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, block: BlockConfig):
        """
        [B, L, D] -> linear -> head_split -> [B, H, L, Dh] -> attn -> head_merge -> [B, L, D]
        """
        super().__init__()
        self.block = block
        self.Dh = block.d_model // block.nhead
        d = block.d_model

        self.W_q = torch.nn.Linear(d, d)
        self.W_k = torch.nn.Linear(d, d)
        self.W_v = torch.nn.Linear(d, d)
        self.W_o = torch.nn.Linear(d, d)

    def head_split(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, H, L, Dh]"""
        B, L, _ = x.shape
        H, Dh = self.block.nhead, self.Dh
        return x.view(B, L, H, Dh).transpose(1, 2).contiguous()

    def head_merge(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, L, Dh] -> [B, L, D]"""
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.head_split(self.W_q(x))
        k = self.head_split(self.W_k(x))
        v = self.head_split(self.W_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.Dh)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        x = self.head_merge(weights @ v)
        return self.W_o(x)

# Cross attention
class MultiHeadCrossAttention(torch.nn.Module):
    def __init__(self, decoder: BlockConfig, encoder: BlockConfig):
        super().__init__()


        self.d_enc = encoder.d_model
        self.d_dec = decoder.d_model

        self.nhead = decoder.nhead
        self.Dh = self.d_dec // self.nhead

        assert self.d_dec % self.nhead == 0

        self.W_q = torch.nn.Linear(self.d_dec, self.d_dec)
        self.W_k = torch.nn.Linear(self.d_enc, self.d_dec)
        self.W_v = torch.nn.Linear(self.d_enc, self.d_dec)
        self.W_o = torch.nn.Linear(self.d_dec, self.d_dec)
    
    def head_split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, Dh = self.nhead, self.Dh
        return x.view(B, L, H, Dh).transpose(1, 2).contiguous()

    def head_merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        x: [B, L, d_decoder] <- decoder 
        y: [B, T, d_encoder] <- encoder

        Grouped Head attention let's try.
        '''
        q = self.head_split(self.W_q(x))
        k = self.head_split(self.W_k(y))
        v = self.head_split(self.W_v(y))
        attn_weights = q @ k.transpose(-2, -1) / math.sqrt(self.Dh)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        x = attn_weights @ v
        x = self.head_merge(x)

        return self.W_o(x)

# Grouped-query cross-attention (GQA): many Q heads share fewer K/V heads
class GroupedHeadCrossAttention(torch.nn.Module):
    """
    x: decoder [B, L, d_dec], y: encoder memory [B, T, d_enc]
    n_group = nhead // num_kv_heads (e.g. 15 Q heads, 3 KV heads -> 5 queries per KV)
    """

    def __init__(self, decoder: BlockConfig, encoder: BlockConfig):
        super().__init__()
        self.d_enc = encoder.d_model
        self.d_dec = decoder.d_model
        self.nhead = decoder.nhead
        self.num_kv_heads = decoder.num_kv_heads
        self.n_group = self.nhead // self.num_kv_heads
        self.Dh = self.d_dec // self.nhead

        kv_dim = self.num_kv_heads * self.Dh

        self.W_q = torch.nn.Linear(self.d_dec, self.nhead * self.Dh)
        self.W_k = torch.nn.Linear(self.d_enc, kv_dim)
        self.W_v = torch.nn.Linear(self.d_enc, kv_dim)
        self.W_o = torch.nn.Linear(self.d_dec, self.d_dec)

    def _split_q(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.Dh).transpose(1, 2).contiguous()

    def _split_kv(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_kv_heads, self.Dh).transpose(1, 2).contiguous()

    def _repeat_kv(self, t: torch.Tensor) -> torch.Tensor:
        """[B, H_kv, T, Dh] -> [B, H_q, T, Dh]"""
        if self.n_group == 1:
            return t
        return t.repeat_interleave(self.n_group, dim=1)

    def _merge_q(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        enc_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_q(self.W_q(x))
        k = self._repeat_kv(self._split_kv(self.W_k(y)))
        v = self._repeat_kv(self._split_kv(self.W_v(y)))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.Dh)
        if enc_key_padding_mask is not None:
            scores = scores.masked_fill(
                enc_key_padding_mask[:, None, None, :], float("-inf")
            )
        weights = torch.softmax(scores, dim=-1)
        out = self._merge_q(weights @ v)
        return self.W_o(out)


class EncoderTransformer(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        enc = config.encoder

        self.LN1 = torch.nn.LayerNorm(enc.d_model)
        self.LN2 = torch.nn.LayerNorm(enc.d_model)
        self.MHA = MultiHeadAttention(enc)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(enc.d_model, enc.d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(enc.d_ff, enc.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA(self.LN1(x))
        x = x + self.FFN(self.LN2(x))
        return x


class DecoderTransformer(torch.nn.Module):
    '''
    Basic is just simple Self-Attention, and then FFN.
    '''
    def __init__(self, config: Config):
        super().__init__()
        dec = config.decoder
        enc = config.encoder

        self.LN1 = torch.nn.LayerNorm(dec.d_model)
        self.LN2 = torch.nn.LayerNorm(dec.d_model)
        self.LN3 = torch.nn.LayerNorm(dec.d_model)
        self.MHA = MultiHeadAttention(dec)
        self.cross_attn = GroupedHeadCrossAttention(dec, enc)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(dec.d_model, dec.d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(dec.d_ff, dec.d_model),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        causal = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        x = x + self.MHA(self.LN1(x), attn_mask=causal)
        x = x + self.cross_attn(self.LN2(x), memory)
        x = x + self.FFN(self.LN3(x))
        return x


class DecoderTransformer_SAONLY(torch.nn.Module):
    '''
    decoder transformer SA only.
    '''
    
    def __init__(self, config: Config):
        super().__init__()
        dec = config.decoder
        enc = config.encoder

        self.LN1 = torch.nn.LayerNorm(dec.d_model)
        self.LN2 = torch.nn.LayerNorm(dec.d_model)
        self.LN3 = torch.nn.LayerNorm(dec.d_model)
        self.MHA = MultiHeadAttention(dec)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(dec.d_model, dec.d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(dec.d_ff, dec.d_model),
        )

    def forward(self, x: torch.Tensor, atten_mask: torch.Tensor | None = None) -> torch.Tensor:
        L = x.size(1)
        x = x + self.MHA(self.LN1(x), atten_mask)
        x = x + self.FFN(self.LN3(x))
        return x