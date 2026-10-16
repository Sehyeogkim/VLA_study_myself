import torch
from configuration import Config
from transformer import DecoderTransformer, EncoderTransformer

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

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        return self.token_embedding(input_ids) + self.pos_embedding(pos)
    
    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.decode_layers:
            x = layer(x, memory)
        return self.lm_head(x)


#Summate encoder and decoder for the training simplicity.
class CaptioningModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, pixel_values: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(pixel_values)
        logits = self.decoder(dec_in, memory)
        return logits



def debugging_main():
    cfg = Config()
    device = torch.device("cpu")

    image = torch.randn(1, cfg.C, cfg.H, cfg.W, device=device)
    encoder = Encoder(cfg).to(device)
    decoder = Decoder(cfg).to(device)

    with torch.no_grad():
        memory = encoder(image)
    print(f"memory: {tuple(memory.shape)}")

    input_ids = torch.tensor([[cfg.bos_token_id]], device=device)
    max_steps = min(16, cfg.max_seq_len - 1)

    for step in range(max_steps):
        with torch.no_grad():
            logits = decoder(input_ids, memory)
        print(f"step {step}: input_ids {input_ids.shape} -> logits {tuple(logits.shape)}")

        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_id.item() == cfg.eos_token_id:
            input_ids = torch.cat([input_ids, next_id], dim=1)
            print("stopped at EOS")
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)
    else:
        print(f"stopped at max_steps={max_steps}")

    print("generated token ids:", input_ids.tolist())
    print("=== encoder + decoder debug OK ===")


if __name__ == "__main__":
    debugging_main()
