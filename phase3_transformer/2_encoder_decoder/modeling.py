'''
Let us start modeling encoder and decoder in here.

Input: Text
Output: Text

Reference: Attention is all you need.


text1 → embed → Encoder (self-attn×N) → x1 (memory)

text2 → embed+pos → Decoder layer (반복):
         ① masked self-attn (Q,K,V = decoder)
         ② cross-attn (Q=decoder, K,V=x1)
         ③ FFN
       → y → lm_head → logits → (decode) text3

'''

from configuration import Config
import torch.nn as nn
from transformers import AutoTokenizer
import torch
MODEL_ID = "distilgpt2"
from transformer import TransformerBlock
from transformer import DecoderTransformerBlock

class EncoderTransformer(nn.Module):
    '''
    ########### Encoder Transformer ###########

    Process:
    1. text -> tokenizer
    2. Embedding + Positional ids
    3. Self-attention
    4. Feed Forward Network
    5. Return Hidden states
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cfg.vocab_size = len(self.tokenizer)
        enc = cfg.encoder                                                   # encoder 쪽 크기 (d_model 1024)
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, enc.d_model)
        self.pos_embedding = nn.Embedding(self.cfg.max_seq_len, enc.d_model)

        # Define the Multi Head Attention block.
        self.layers = nn.ModuleList()
        for _ in range(enc.num_layers):
            self.layers.append(TransformerBlock(enc))                       # 블록 안의 cfg.d_model = 1024
        # Final LN after encoder stack (paper Figure 1; stabilizes memory for cross-attention).
        self.ln_f = nn.LayerNorm(enc.d_model)

    def tokenizer_preprocessing(self, text):
        '''
        Input: text
        Output: input_ids, lang_pad
        '''
        # text -> token
        lang = self.tokenizer(text, 
                            return_tensors = "pt", 
                            padding = "max_length", 
                            truncation = True, 
                            max_length = self.cfg.max_seq_len)

        input_ids = lang.input_ids
        lang_pad = lang.attention_mask
        return input_ids, lang_pad

    def forward(self, input_ids, lang_pad):
        '''
        Input: input_ids(tokens), lang_pad
        Output: hidden_states
        '''
        # token -> embedding
        lang_tokens = self.token_embedding(input_ids) # [B, L, d_encoder]
        
        # create position -> embedding 
        position_ids = (torch.cumsum(lang_pad, dim=1) - 1).clamp(min=0)   # [B, L]
        pos_tokens = self.pos_embedding(position_ids)                     # [B, L, d_encoder]
        
        # Just add in here
        x = lang_tokens + pos_tokens                                      # [B, L, d_encoder]
        
        #attn_mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        attn_mask = None

        # Pad mask.
        key_padding_mask = (lang_pad == 0)                 # pad(0) 만 True

        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)

        x = self.ln_f(x)
        return x


class DecoderTransformer(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cfg.vocab_size = len(self.tokenizer)
        dec = cfg.decoder                                                   # decoder 쪽 크기 (d_model 512)
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, dec.d_model)
        self.pos_embedding = nn.Embedding(self.cfg.max_seq_len, dec.d_model)

        

        self.decoder_layers = nn.ModuleList()
        for _ in range(dec.num_layers):
            # cross-attn 의 K, V 입력 = encoder 출력 차원
            self.decoder_layers.append(DecoderTransformerBlock(dec, cfg.encoder.d_model))
        # Final LN before vocab projection (same role as ln_f in decoder-only GPT stack).
        self.ln_f = nn.LayerNorm(dec.d_model)
        self.lm_head = nn.Linear(dec.d_model, self.cfg.vocab_size)

    def tokenizer_preprocessing(self, text):
        '''
        Input: text
        Output: input_ids, lang_pad
        '''
        # text -> token
        lang = self.tokenizer(text, 
                            return_tensors = "pt", 
                            padding = "max_length", 
                            truncation = True, 
                            max_length = self.cfg.max_seq_len)

        input_ids = lang.input_ids
        lang_pad = lang.attention_mask
        return input_ids, lang_pad
    
    def forward(self, input_ids, lang_pad, memory, enc_lang_pad):
        '''
        Input: decoder tokens/pad, encoder memory, encoder attention_mask (for CA key padding).
        Output: logits [B, L_dec, vocab] (raw; use CrossEntropyLoss in train, argmax in infer).
        '''
        # token -> embedding
        lang_tokens = self.token_embedding(input_ids) # [B, L, d_model]

        # create position -> embedding 
        position_ids = (torch.cumsum(lang_pad, dim=1) - 1).clamp(min=0)   # [B, L]
        pos_tokens = self.pos_embedding(position_ids)                     # [B, L, d_model]
        
        # Just add in here
        x = lang_tokens + pos_tokens                                      # [B, L, d_model]

        L = x.shape[1]
        causal_attn_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
        )
        dec_key_padding_mask = (lang_pad == 0)
        enc_key_padding_mask = (enc_lang_pad == 0)

        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                causal_attn_mask,
                dec_key_padding_mask,
                enc_key_padding_mask,
            )

        x = self.ln_f(x)
        return self.lm_head(x)
    
def main_debugging():

    text1 = "Hello, how are you?"
    text2 = "안녕하세요"

    cfg = Config()

    # Step1. text1 -> encoder -> hidden states
    encoder = EncoderTransformer(cfg)
    src_ids, src_pad = encoder.tokenizer_preprocessing(text1)
    memory = encoder(src_ids, src_pad)
    print("memory", memory.shape)

    decoder = DecoderTransformer(cfg)
    tgt_ids, tgt_pad = decoder.tokenizer_preprocessing(text2)
    logits = decoder(tgt_ids, tgt_pad, memory, src_pad)
    print("logits", logits.shape)

    decoded_tokens = torch.argmax(logits, dim=-1)
    decoded_text = decoder.tokenizer.decode(decoded_tokens[0])
    print(decoded_text)


    return

if __name__ == "__main__":
    main_debugging()