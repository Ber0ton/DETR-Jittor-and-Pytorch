import math
import jittor as jt
import jittor.nn as nn
from typing import Optional

from util.misc import NestedTensor 

# ---------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    """
    Sine‑cosine positional embedding (image‑adapted “Attention‑is‑All‑You‑Need” style).
    Produces a B×C×H×W tensor that can be added to backbone feature maps.
    """

    def __init__(self, num_pos_feats: int = 64,
                 temperature: float = 10_000.,
                 normalize: bool = False,
                 scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        if scale is not None and not normalize:
            raise ValueError("If `scale` is given, `normalize` must be True.")
        self.scale = scale if scale is not None else 2 * math.pi

    # ---------------------------------------------------------------------
    def execute(self, tensor_list: NestedTensor):      # Jittor’s forward
        x    = tensor_list.tensors                    # B,C,H,W
        mask = tensor_list.mask                       # B,H,W  (bool)
        assert mask is not None

        not_mask = jt.logical_not(mask)
        y_embed  = jt.cumsum(not_mask, dim=1).float()   # B,H,W
        x_embed  = jt.cumsum(not_mask, dim=2).float()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = jt.arange(self.num_pos_feats, dtype=jt.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # (B,H,W,#feat) – broadcast divide
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        # interleave sin / cos, then flatten last two dims
        pos_x = jt.stack((jt.sin(pos_x[..., 0::2]),
                          jt.cos(pos_x[..., 1::2])), dim=4).reshape(pos_x.shape[0],
                                                                     pos_x.shape[1],
                                                                     pos_x.shape[2], -1)
        pos_y = jt.stack((jt.sin(pos_y[..., 0::2]),
                          jt.cos(pos_y[..., 1::2])), dim=4).reshape(pos_y.shape[0],
                                                                     pos_y.shape[1],
                                                                     pos_y.shape[2], -1)

        pos = jt.concat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)   # B,2F,H,W
        return pos

    # PyTorch‑style alias so external code can still call .forward(...)
    forward = execute

# ---------------------------------------------------------------------------
class PositionEmbeddingLearned(nn.Module):
    """
    Learned absolute positional embedding (50×50 max resolution by default).
    """

    def __init__(self, num_pos_feats: int = 256, h_max: int = 50, w_max: int = 50):
        super().__init__()
        self.row_embed = nn.Embedding(h_max, num_pos_feats)
        self.col_embed = nn.Embedding(w_max, num_pos_feats)
        self.reset_parameters()

    # ---------------------------------------------------------------------
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    # ---------------------------------------------------------------------
    def execute(self, tensor_list: NestedTensor):
        x = tensor_list.tensors                        # B,C,H,W
        h, w = x.shape[-2:]
        i = jt.arange(w, device=x.device)
        j = jt.arange(h, device=x.device)

        x_emb = self.col_embed(i)                      # W,D
        y_emb = self.row_embed(j)                      # H,D

        pos = jt.concat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),        # H,W,D
            y_emb.unsqueeze(1).repeat(1, w, 1)         # H,W,D
        ], dim=-1).permute(2, 0, 1)                    # 2D,H,W
        pos = pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)   # B,2D,H,W
        return pos

    forward = execute

# ---------------------------------------------------------------------------
def build_position_encoding(args):
    """
    Factory identical to the original DETR helper.

    args.hidden_dim           – transformer hidden size (C); we halve it per sin/cos branch
    args.position_embedding   – 'sine'/'v2' or 'learned'/'v3'
    """
    n_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # sinu‑cosine (normalized to [0,2π])
        return PositionEmbeddingSine(n_steps, normalize=True)
    if args.position_embedding in ('v3', 'learned'):
        return PositionEmbeddingLearned(n_steps)
    raise ValueError(f"Unsupported position_embedding: {args.position_embedding}")
