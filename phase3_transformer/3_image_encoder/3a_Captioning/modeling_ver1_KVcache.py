import torch
from configuration import Config
from transformer_KVcache import DecoderTransformer, EncoderTransformer

'''
build three classes

1. Config (configuration.py)
2. Encoder
3. Decoder

Transformer blocks at ./transformer.py

input: image
output: text
'''

class Encoder(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        enc = config.encoder

        patch_dim = config.C * config.P * config.P
        self.patch_proj = torch.nn.Linear(patch_dim, enc.d_model)

        grid = config.H // config.P
        n_patches = grid * grid
        n_tokens = n_patches + (1 if config.use_cls else 0)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, n_tokens, enc.d_model))
        if config.use_cls:
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, enc.d_model))
        else:
            self.cls_token = None

        self.layers = torch.nn.ModuleList(
            [EncoderTransformer(config) for _ in range(enc.num_layers)]
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.config.C
        assert H % self.config.P == 0 and W % self.config.P == 0

        gh, gw = H // self.config.P, W // self.config.P
        patches = x.unfold(2, self.config.P, self.config.P).unfold(
            3, self.config.P, self.config.P
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, gh * gw, C * self.config.P * self.config.P)

        patch_embeds = self.patch_proj(patches)

        image_embeds = patch_embeds
        if self.cls_token is not None:
            cls_embed = self.cls_token.expand(B, -1, -1)
            image_embeds = torch.cat([cls_embed, patch_embeds], dim=1)

        image_embeds = image_embeds + self.pos_embed[:, : image_embeds.size(1), :]
        return image_embeds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dec = config.decoder

        self.token_embedding = torch.nn.Embedding(config.vocab_size, dec.d_model)
        self.pos_embedding = torch.nn.Embedding(config.max_seq_len, dec.d_model)

        self.decode_layers = torch.nn.ModuleList(
            [DecoderTransformer(config) for _ in range(dec.num_layers)]
        )
        self.lm_head = torch.nn.Linear(dec.d_model, config.vocab_size)

    def embed_tokens(self, input_ids: torch.Tensor, past_len: int = 0) -> torch.Tensor:
        '''
        input_ids: [B, L] (학습 / 첫 step) 또는 [B, 1] (cache 있을 때 new token만 들어옴.)
        past_len : cache 에 이미 들어있는 토큰 수 -> 위치를 past_len 부터 매김
        '''
        B, L = input_ids.shape
        pos = torch.arange(past_len, past_len + L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        return self.token_embedding(input_ids) + self.pos_embedding(pos)
    
    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor, past_kvs: list | None = None):
        '''
        past_kvs: layer 수 만큼의 [(K, V), (K, V), ...] , K/V: [B, H, T, Dh]
        return  : logits [B, L, V], new_past_kvs (layer 수 만큼)
        '''
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)   # 첫 layer K 의 T
        x = self.embed_tokens(input_ids, past_len)

        new_past_kvs = []
        for i, layer in enumerate(self.decode_layers):
            layer_past = None if past_kvs is None else past_kvs[i]       # i 번째 layer 의 cache
            x, kv = layer(x, memory, past_kv=layer_past)
            new_past_kvs.append(kv)
        return self.lm_head(x), new_past_kvs


#Summate encoder and decoder for the training simplicity.
class CaptioningModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, pixel_values: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(pixel_values)
        logits, _ = self.decoder(dec_in, memory)   # 학습: cache 안 씀
        return logits