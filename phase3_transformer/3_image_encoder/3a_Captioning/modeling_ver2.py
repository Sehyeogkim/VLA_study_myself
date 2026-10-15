import torch
from configuration import Config
from transformer import DecoderTransformer_SAONLY, EncoderTransformer

'''
이 버전에서는 Cross attention 을 안하고 그냥 deocder - SA only 로 들어간다.
'''

# Encoder is same.
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


# Decoder is different.
class Decoder(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dec = config.decoder

        self.token_embedding = torch.nn.Embedding(config.vocab_size, dec.d_model)
        self.pos_embedding = torch.nn.Embedding(config.max_seq_len, dec.d_model)

        self.decode_layers = torch.nn.ModuleList(
            [DecoderTransformer_SAONLY(config) for _ in range(dec.num_layers)]
        )
        self.lm_head = torch.nn.Linear(dec.d_model, config.vocab_size)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        return self.token_embedding(input_ids) + self.pos_embedding(pos)
    
    def forward(self, seq: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """seq: [B, T_img + L_txt, d_model] already concatenated embeddings."""
        x = seq
        for layer in self.decode_layers:
            x = layer(x, attn_mask)
        return self.lm_head(x)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """
    SmolVLA / OpenPI style (see phase2_policy_myself/utils.py).
    att_masks: 1 = new block starts at this token, 0 = same block as previous.
    Returns bool [B, L, L] with True = query i may attend to key j.
    """
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d & pad_2d


def prefix_image_text_1d_masks(
    T_img: int, L_txt: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [image | text]: image tokens att=0 (one bidirectional block),
    text tokens att=1,1,... (causal block, can see all image).
    """
    S = T_img + L_txt
    pad_masks = torch.ones(batch_size, S, dtype=torch.bool, device=device)
    att_masks = torch.zeros(batch_size, S, dtype=torch.int64, device=device)
    if L_txt > 0:
        att_masks[:, T_img:] = 1
    return pad_masks, att_masks


def mha_block_mask(T_img: int, L_txt: int, device: torch.device) -> torch.Tensor:
    """bool [S, S] with True = masked out (for MultiHeadAttention)."""
    pad_masks, att_masks = prefix_image_text_1d_masks(T_img, L_txt, 1, device)
    can_attend = make_att_2d_masks(pad_masks, att_masks)
    return ~can_attend[0]


def debugging_main():
    cfg = Config()
    device = torch.device("cpu")

    image = torch.randn(1, cfg.C, cfg.H, cfg.W, device=device)
    encoder = Encoder(cfg).to(device)
    decoder = Decoder(cfg).to(device)

    with torch.no_grad():
        memory = encoder(image)
    text_tokens = torch.tensor([[cfg.bos_token_id]], device=device)
    
    #Wanna get output token min 16.
    max_steps = min(16, cfg.max_seq_len - 1)
    for step in range(max_steps):
        with torch.no_grad():

            #근데 이게 image는 양방향그리고 text는 casual로 만들어야하기는 함. 
            text_emb = decoder.embed_tokens(text_tokens)
            T_img, L_txt = memory.size(1), text_tokens.size(1)
            attn_mask = mha_block_mask(T_img, L_txt, device)
            seq = torch.cat([memory, text_emb], dim=1)
            logits = decoder(seq, attn_mask)

            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() == cfg.eos_token_id:
                break
            text_tokens = torch.cat([text_tokens, next_id], dim=1)
    

if __name__ == "__main__":
    debugging_main()
