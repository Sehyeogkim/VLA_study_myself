'''
utils fort the VLA model.
'''
import torch

class UtilsForVLA:

    def sinusoidal_time_emb(t, dim, min_period=4e-3, max_period=4.0):
        """
        (phase2 utils.py 와 같음, SmolVLA create_sinusoidal_pos_embedding 과 같은 식)
        t     : [B]  flow matching time (0~1)
        return: [B, dim]  앞 절반 sin, 뒤 절반 cos (주기 min_period ~ max_period)
        """
        import math
        fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device)
        period = min_period * (max_period / min_period) ** fraction     # [dim/2]  빠른 주기 ~ 느린 주기
        x = t[:, None] * (2 * math.pi / period)                         # [B, dim/2]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)          # [B, dim]

    def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:  # see openpi (exact copy)
        """Copied from big_vision.

        Tokens can attend to valid inputs tokens which have a cumulative mask_ar
        smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
        setup several types of attention, for example:

        [[1 1 1 1 1 1]]: pure causal attention.

        Args:
        input_mask: bool[B, N] true if its part of the input, false if padding.
        mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
            it and 0 where it shares the same attention mask as the previous token.
        """
        if att_masks.ndim != 2:
            raise ValueError(att_masks.ndim)
        if pad_masks.ndim != 2:
            raise ValueError(pad_masks.ndim)

        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks