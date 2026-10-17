'''
- Gropued Multi Head attention
- self attention.
- Cross attention.
'''
from configuration import Config
import torch.nn as nn
import torch
import math

# Self attention layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config                      # Group_size 를 config 에서 씀

        self.suffix_dim = config.expert_hidden
        self.prefix_dim = config.lm_hidden

        self.head_dim = config.head_dim # head dimension 64 fixed by LLM.

        self.d_q_head = config.head_dim * config.num_q_heads
        self.d_kv_head = config.head_dim * config.num_kv_heads
        self.q_heads = config.num_q_heads
        self.kv_heads = config.num_kv_heads

        self.W_q = torch.nn.Linear(self.suffix_dim, self.d_q_head)
        self.W_k = torch.nn.Linear(self.suffix_dim, self.d_kv_head)
        self.W_v = torch.nn.Linear(self.suffix_dim, self.d_kv_head)
        self.W_o = torch.nn.Linear(self.d_q_head, self.suffix_dim)
        self.dropout = torch.nn.Dropout(0.0)
    
    def split(self, t):
        '''
            split [B,T,D] -> [B,H,T,Dh]
            Dh = self.head_dim is fixed over q, k and v.
        '''
        B,T,_ = t.shape
        return t.view(B,T,-1,self.head_dim).transpose(1,2)
    
    def merge(self, t):
        '''
            merge [B,H,T,Dh] -> [B,T, H * Dh]
        '''
        B, _, T, _ = t.shape
        return t.transpose(1,2).contiguous().view(B,T,-1)

    def forward(self, x, attn_mask, KV_cache):

        # x = [B, T_suffix, d_suffix]
        q = self.W_q(x) # [B, T_suffix, d_q_head]
        k_suffix, v_suffix = self.W_k(x), self.W_v(x) # [B, T_suffix, d_kv_head]
        k_suffix, v_suffix = self.split(k_suffix), self.split(v_suffix)

        # KV cahce is already split with it's # of heads = 3 we are 
        k_prefix, v_prefix = KV_cache

        # Now concat the KV cache(prefix) + KV from suffix
        k = torch.cat([k_prefix, k_suffix], dim = 2) # [B, H_kv, T_prefix + T_suffix, Dh]
        v = torch.cat([v_prefix, v_suffix], dim = 2) # [B, H_kv, T_prefix + T_suffix, Dh]

        # Now we can perform the multi head attention.
        q = self.split(q) # [B, H_q, T_suffix, Dh]

        # k,v 를 expands.
        k = k.repeat_interleave(self.config.Group_size, dim = 1) # [B, H', T', Dh] -> [B, H, T', Dh]
        v = v.repeat_interleave(self.config.Group_size, dim = 1) # [B, H', T', Dh] -> [B, H, T', Dh]

        # Now we can perform the multi head attention.
        
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) # [B, H, T_suffix, T_prefix + T_suffix]
        attn_mask = attn_mask[:, None, -scores.shape[2]:, : scores.shape[3]] # [B, 1, T_suffix, T_prefix + T_suffix]
        scores = scores.masked_fill(attn_mask == False, -1e9)

        attn = self.dropout(torch.softmax(scores, dim = -1)) # [B, H, T_suffix, T_prefix + T_suffix]
        out = attn @ v  # [B, H, T_suffix, T_prefix + T_suffix] @ [B, H, T_prefix + T_suffix, D_h] -> [B, H, T_suffix, D_h]
        out = self.merge(out) # [B, T_suffix, H * D_h] -> [B, T_suffix, d_q_head]

        return self.W_o(out) # [B, T_suffix, d_q_head] -> [B, T_suffix, d_suffix]
    
# Cross attention layer
class MultiHeadCrossAttention(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config                      # Group_size 를 config 에서 씀

        self.suffix_dim = config.expert_hidden
        self.prefix_dim = config.lm_hidden

        self.head_dim = config.head_dim # head dimension 64 fixed by LLM.

        self.d_q_head = config.head_dim * config.num_q_heads
        self.d_kv_head = config.head_dim * config.num_kv_heads
        self.q_heads = config.num_q_heads
        self.kv_heads = config.num_kv_heads

        self.W_q = torch.nn.Linear(self.suffix_dim, self.d_q_head)
        self.W_k = torch.nn.Linear(self.d_kv_head, self.d_kv_head)
        self.W_v = torch.nn.Linear(self.d_kv_head, self.d_kv_head)
        self.W_o = torch.nn.Linear(self.d_q_head, self.suffix_dim)
        self.dropout = torch.nn.Dropout(0.0)      # 지금은 dropout 안 씀 (자리만)

    def split(self, t):
        B,T,_ = t.shape
        # [B, T, D] -> [B, T, H, Dh] each q and (k,v) has different T but Dh is same.
        return t.view(B, T, -1, self.head_dim).transpose(1, 2)

    def merge(self, t):
        '''merge for the multi head attention. -> single head tensor'''
        # [B, H, T, Dh] -> [B, T, D]
        B, _, T, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, -1)
    
    def forward(self, x, attn_mask, KV_cache):
        '''
        KV Cache is alreay divided with it's # of heads = 3 so 
        '''
        k,v = KV_cache
        B, _, T_prefix, _ = k.shape
        # [ B, 3, T_prefix, D] -> [B, T_prefix, 3 * D]
        k = k.transpose(1,2).contiguous().view(B, T_prefix, self.d_kv_head)
        v = v.transpose(1,2).contiguous().view(B, T_prefix, self.d_kv_head)

        q = self.W_q(x) # [B, T_suffix, d_q_head]
        k = self.W_k(k) # [B, T_prefix, d_kv_head]
        v = self.W_v(v) # [B, T_prefix, d_kv_head]

        #Split into the heads.
        _, T_suffix, _ = q.shape
        q = self.split(q)   # [B, 9, T_suffix, 64]
        k = self.split(k)  # [B, 3, T_prefix, 64]
        v = self.split(v)  # [B, 3, T_prefix, 64]
        
        # k,v 를 expands.
        k = k.repeat_interleave(self.config.Group_size, dim = 1) # [B, H', T', Dh] -> [B, H, T', Dh]
        v = v.repeat_interleave(self.config.Group_size, dim = 1) # [B, H', T', Dh] -> [B, H, T', Dh]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) # [B, H, T, T']

        #attentio mask should be cropped
        attn_mask = attn_mask[:,None,-T_suffix:, : T_prefix] # [B, 1, T_suffix, T_prefix]

        scores = scores.masked_fill(attn_mask == False, -1e9)
        attn = self.dropout(torch.softmax(scores, dim = -1)) # [B, H, T, T']

        out = attn @ v  # [B, H, T, T'] @ [B, H, T', Dh] -> [B, H, T, Dh]
        # out = self._merge(out) # [B, H, T, Dh] -> [B, T, D]
        out = self.merge(out)    # [B, 9, T, 64] -> [B, T, 576]

        return self.W_o(out) # [B, T, D] -> [B, T, D]

    


# Transformer block
class SingleTransformerBlock(torch.nn.Module):
    
    def __init__(self, config: Config, is_cross: bool = False):
        super().__init__()
        self.LN1 = nn.LayerNorm(config.expert_hidden)
        self.LN2 = nn.LayerNorm(config.expert_hidden)

        self.FFN = nn.Sequential(
            nn.Linear(config.expert_hidden, config.expert_ff_hidden),
            nn.GELU(),
            nn.Linear(config.expert_ff_hidden, config.expert_hidden))  

        if is_cross:
            self.attention = MultiHeadCrossAttention(config)
        else:
            self.attention = MultiHeadSelfAttention(config)


    def forward(self, x, attn_mask, KV_cache):
        x = x + self.attention(self.LN1(x), attn_mask, KV_cache)
        x = x + self.FFN(self.LN2(x))
        return x


# Expert trnasformer block has bunch of layers
class ExpertTransformer(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer_blocks = nn.ModuleList()
        for i in range(config.expert_layers):
            self.transformer_blocks.append(SingleTransformerBlock(config, is_cross = (i%2 ==1)))

    def forward(self, x, attn_mask, KV_cache):
        for i, layer in enumerate(self.transformer_blocks): # i strats from zero right?
            kv = (KV_cache.layers[i].keys, KV_cache.layers[i].values)
            x = layer(x, attn_mask, kv)
        return x