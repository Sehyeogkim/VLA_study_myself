import torch
import torch.nn as nn

"""
Decoder-only LM (GPT-style): tokenizer + your own stack (embedding → causal blocks → lm_head).

Tokenizer: distilgpt2 (same BPE vocab as gpt2, 50257). Use cfg.tokenizer_name = "gpt2" if you prefer.
No VLM — nn.Embedding lives in this module.
"""

from configuration import Config

class TransformerBlock(nn.Module):
    '''
    하나의 transformer layer block을 정의 이 레이어가 여러개 호출되어서 연속적인 연산이 일어날 예정임.
    '''

    def __init__(self, cfg):
        super().__init__()
        self.LN1 = nn.LayerNorm(cfg.d_model)
        self.LN2 = nn.LayerNorm(cfg.d_model)
        self.FFN = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.ReLU(),
            nn.Linear(cfg.dim_feedforward, cfg.d_model))

        # Multi Head Attention block. should be done by my self.
        self.SAttention = nn.MultiheadAttention(cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True)  # 입력 [B, L, D]


    def forward(self, x, attn_mask, key_padding_mask):
        x_pre = x
        x = self.LN1(x)
        # q = k = v = x (self-attention). 반환 (출력, attention 가중치) 중 출력만
        x = self.SAttention(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + x_pre

        x_pre = x
        x = self.LN2(x)
        x = self.FFN(x)
        x = x + x_pre
        return x

class DecoderOnlyTransformer(nn.Module):
    """Skeleton — stack causal blocks + lm_head yourself."""

    def __init__(self):
        super().__init__()
        self.cfg = Config()
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        
        # what is eos token? it is the end of the sentence token.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 임베딩 전용을 따로 학습을 해야한다는 거임. 아 Linear이랑 다르게 그러니끼ㅏ 
        # #list index마다 벡터가 있따고 생각하면 되미 그 리스트의 길이와 벡터가 저기 저장되어있는거임.
        self.cfg.vocab_size = len(self.tokenizer)
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos_embedding = nn.Embedding(self.cfg.max_seq_len, self.cfg.d_model)

        # 한번에 모든 Transformer layer을 뽑아야함. 이게 이제 학습용 레이어들임.
        self.layers = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.num_layers)])
        self.ln_f = nn.LayerNorm(self.cfg.d_model)                          # 마지막 LN (모델에 1개)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size)     # D -> vocab (모델에 1개)
    

    def tokenizer_preprocessing(self, prompt):
        '''
        text -> (tokenizer) -> input_ids -> (embed_tokens, this is dictonary) -> lang_tokens
        '''

        # text -> token
        lang = self.tokenizer(prompt, 
                            return_tensors = "pt", 
                            padding = "max_length", 
                            truncation = True, 
                            max_length = self.cfg.tokenizer_max_length)

        input_ids = lang.input_ids
        lang_pad = lang.attention_mask

        return input_ids, lang_pad

    def forward(self, input_ids, lang_pad):
        '''
        Input: input_ids [B, L] - list of tokens, lang_pad [B, L] - list of padding 0 or 1.
        Transformer layer * N -> Layernorm -> lm_head linear D -> vocab
        '''

        # token -> embedding
        lang_tokens = self.token_embedding(input_ids) # [B, L, d_model]

        # position -> embedding
        # pad 제외하고 0 부터 번호 (pad 자리는 앞 번호 반복 -> 어차피 mask 로 가려짐)
        position_ids = (torch.cumsum(lang_pad, dim=1) - 1).clamp(min=0)   # [B, L]
        pos_tokens = self.pos_embedding(position_ids)                     # [B, L, d_model]

        # Add (not concat) the token and position embeddings
        x = lang_tokens + pos_tokens                                      # [B, L, d_model]

        L = x.shape[1]

        # causal mask [L, L]: True = 못 봄 (대각선 위 = 미래 토큰)
        attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        
        # pad mask [B, L]: True = pad (key 로 안 봄)
        key_padding_mask = lang_pad.bool()

        # Transformer layer * N
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask) # [B, L, d_model]

        # Layernorm
        x = self.ln_f(x)

        # lm_head linear D -> vocab
        x = self.lm_head(x)                       # [B, L, vocab] = logits
        
        return x

def main():

    prompt = "Hello, how are you?"
    model = DecoderOnlyTransformer()
    input_ids, lang_pad = model.tokenizer_preprocessing(prompt)
    x = model.forward(input_ids, lang_pad)
    print("x :", x.shape)

if __name__ == "__main__":
    main()
