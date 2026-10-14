"""
modeling.py — 모델 본체 (smolvla: modeling_smolvla.py)

흐름:
    image -> vision encoder (frozen) -> image tokens
    state -> state_proj             -> state token
    (x_t, t) -> action_in_proj + time emb -> action tokens
    [prefix | suffix] -> transformer (attention mask) -> action_out_proj -> v_t

학습:  x_t = t*noise + (1-t)*a,  target u_t = noise - a,  loss = MSE(v_t, u_t)
추론:  x_1 = noise -> Euler (t: 1 -> 0) -> x_0 = action chunk
"""
import torch
from torch import nn

from configuration import MyPolicyConfig


class MyFlowMatchingModel(nn.Module):
    def __init__(self, config: MyPolicyConfig):
        super().__init__()
        # TODO: vision encoder, state_proj, action_in/out_proj, time embedding, transformer
        pass

    def embed_prefix(self, images, state):
        # TODO: image + state tokens, pad/att masks
        pass

    def embed_suffix(self, x_t, t):
        # TODO: noisy action + time tokens
        pass

    def forward(self, images, state, actions):
        # TODO: noise, t 샘플 -> x_t, u_t -> v_t 예측 -> loss
        pass

    @torch.no_grad()
    def sample_actions(self, images, state):
        # TODO: noise -> Euler num_steps -> action chunk
        pass


class MyPolicy(nn.Module):
    """lerobot Policy 처럼 바깥 껍데기: batch dict 받기 + action queue"""

    def __init__(self, config: MyPolicyConfig):
        super().__init__()
        # TODO: self.model, action queue
        pass

    def forward(self, batch):
        # TODO: 학습용 -> loss 반환
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        # TODO: model.sample_actions -> [B, chunk_size, action_dim]
        pass

    @torch.no_grad()
    def select_action(self, batch):
        # TODO: queue 비면 sample_actions 로 chunk 채우고 하나씩 pop
        pass
