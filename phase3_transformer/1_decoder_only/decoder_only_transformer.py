import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Decoder-only LM (GPT-style): tokenizer + your own stack (embedding → causal blocks → lm_head).

Tokenizer: distilgpt2 (same BPE vocab as gpt2, 50257). Use cfg.tokenizer_name = "gpt2" if you prefer.
No VLM — nn.Embedding lives in this module.


Multihead Attention cases.
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
        key_padding_mask = (lang_pad == 0)                 # pad(0) 만 True. (lang_pad.bool() 이면 반대로 진짜 토큰을 가림 -> NaN)

        # Transformer layer * N
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask) # [B, L, d_model]

        # Layernorm
        x = self.ln_f(x)

        # lm_head linear D -> vocab
        x = self.lm_head(x)                       # [B, L, vocab] = logits
        
        return x
    
    # This is for the infernece.
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> str:
        
        self.eval()
        device = next(self.parameters()).device

        # prompt -> token.
        ids = torch.tensor([self.tokenizer.encode(prompt)], device=device)
        eos = self.tokenizer.eos_token_id
        
        # 계속해서 반복해서 토큰을 예측하는 거임. (한 바퀴 = forward 1 번 = 토큰 1 개)
        for _ in range(max_new_tokens):

            # 1. 최근 L(128) 개 토큰만 사용 (context window)
            ctx = ids[:, -self.cfg.L:]                       # [1, T]  정수 토큰 ID (T <= 128)

            # 2. forward: 임베딩 -> 블록 6 개 -> lm_head   (self(...) = self.forward(...))
            logits_all = self(ctx, torch.ones_like(ctx))     # [1, T, vocab]  위치마다 "다음 토큰" 점수

            # 3. 마지막 위치 점수만 사용
            logits = logits_all[:, -1, :]                    # [1, vocab]

            # 4. 다음 토큰 1 개 고르기
            if temperature <= 0:
                # 가장 점수 높은 토큰 (항상 같은 결과)
                next_id = logits.argmax(dim=-1, keepdim=True)            # [1, 1]
            else:
                # 4-1. temperature: 작으면 무난하게, 크면 다양하게
                logits = logits / temperature

                # 4-2. top_k: 상위 k 개 후보만 남기고 나머지는 -inf (확률 0)
                if top_k is not None:
                    topk_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))   # 상위 k 개 점수
                    kth_value = topk_values[:, -1:]                                     # k 번째 점수 (커트라인)
                    logits[logits < kth_value] = float("-inf")

                # 4-3. 점수 -> 확률 -> 확률에 따라 1 개 뽑기 (주사위)
                probs = F.softmax(logits, dim=-1)                        # [1, vocab]  합 = 1
                next_id = torch.multinomial(probs, num_samples=1)        # [1, 1]

            # 5. 뽑은 토큰을 뒤에 붙임 -> 다음 반복의 입력이 됨
            ids = torch.cat([ids, next_id], dim=1)           # [1, T] -> [1, T+1]

            # 6. "글 끝" 토큰이면 중단
            if next_id.item() == eos:
                break

        # 7. 토큰 ID -> 글자
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

def main():

    #Inference test.
    prompt = "Hello, how are you?"
    model = DecoderOnlyTransformer()
    input_ids, lang_pad = model.tokenizer_preprocessing(prompt)
    x = model.forward(input_ids, lang_pad)
    print("x :", x.shape)

if __name__ == "__main__":
    main()
