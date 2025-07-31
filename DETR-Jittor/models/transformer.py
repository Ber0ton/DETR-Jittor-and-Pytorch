import copy
from typing import Optional, List

import jittor as jt
import jittor.nn as nn
import jittor.nn as F

# ---------------------------------------------------------------------
# Minimal Multi‑Head Attention for Jittor (PyTorch‑like API)
# ---------------------------------------------------------------------
import math, jittor as jt
from jittor import nn

# ---------------------------------------------------------------------
# PyTorch‑compatible Multi‑Head Attention for Jittor
# ---------------------------------------------------------------------
import math, jittor as jt
from jittor import nn

class TorchLikeMHA(nn.Module):
    """Exact signature of torch.nn.MultiheadAttention (forward=execute)."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scaling    = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout  = nn.Dropout(dropout)

    # -----------------------------------------------------------------
    def execute(self, query, key, value,
                attn_mask=None,          # (L,L) or broadcast‑able
                key_padding_mask=None,   # (B,L) bool, True=PAD
                need_weights=False, average_attn_weights=True):
        """
        Inputs follow PyTorch: (L, B, E). Returns (L,B,E), attn_weights|None.
        """
        tgt_len, B, _ = query.shape
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        # ---- reshape to (B*num_heads, L, head_dim) -------------------
        def _shape(x):
            L, B, E = x.shape
            x = x.reshape(L, B, self.num_heads, self.head_dim)
            return x.permute(1, 2, 0, 3).reshape(B * self.num_heads, L, self.head_dim)
        q, k, v = map(_shape, (q, k, v))

        # ---- scaled dot‑product attention ----------------------------
        attn = jt.matmul(q, k.transpose(1, 2))                # (B*H, L, L)

        if attn_mask is not None:                             # additive mask
            attn += attn_mask

        # --- inside TorchLikeMHA.execute()  ------------------------------------------
        if key_padding_mask is not None:
            # key_padding_mask: (B, L_k)
            key_len = key_padding_mask.shape[1]          # usually == tgt_len
            # B,1,1,L_k  →  broadcast to  B,H,L_q,L_k  →  reshape to (B*H, L_q, L_k)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)          # B,1,1,L_k
            mask = mask.broadcast([B, self.num_heads, tgt_len, key_len])
            mask = mask.reshape(B * self.num_heads, tgt_len, key_len)
            attn  = jt.where(mask, -1e9, attn)            # fill PAD positions


        attn = nn.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out  = jt.matmul(attn, v)                             # (B*H, L, head_dim)

        # ---- back to (L, B, E) --------------------------------------
        out = out.reshape(B, self.num_heads, tgt_len, self.head_dim)
        out = out.permute(2, 0, 1, 3).reshape(tgt_len, B, self.embed_dim)
        out = self.out_proj(out)

        if need_weights:
            # average over heads like PyTorch when requested
            attn = attn.reshape(B, self.num_heads, tgt_len, tgt_len)
            if average_attn_weights:
                attn = attn.mean(dim=1)
            return out, attn
        return out, None

# --- monkey‑patch the module symbol unconditionally --------------------------
nn.MultiHeadAttention = TorchLikeMHA



# -----------------------------------------------------------------------------
class Transformer(nn.Module):
    """
    Encoder–decoder transformer with positional‑encoding support in attention,
    mirroring the original DETR architecture.
    """

    def __init__(self, d_model: int = 512,
                       nhead: int = 8,
                       num_encoder_layers: int = 6,
                       num_decoder_layers: int = 6,
                       dim_feedforward: int = 2048,
                       dropout: float = 0.1,
                       activation: str = "relu",
                       normalize_before: bool = False,
                       return_intermediate_dec: bool = False):
        super().__init__()

        enc_layer  = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                             dropout, activation, normalize_before)
        enc_norm   = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers, enc_norm)

        dec_layer  = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                             dropout, activation, normalize_before)
        dec_norm   = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers, dec_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead   = nhead

    # -------------------------------------------------------------------------
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -------------------------------------------------------------------------
    def execute(self, src, mask, query_embed, pos_embed):
        """
        Args
        ----
        src         : [B,C,H,W] – backbone feature map
        mask        : [B,H,W]   – bool (True = pad) for those features
        query_embed : [Q,C]     – object queries (learned embeddings)
        pos_embed   : [B,C,H,W] – positional encoding for each pixel
        Returns
        -------
        hs      : [L,B,Q,C] – decoder layer outputs (L = num_decoder_layers)
        memory  : [B,C,H,W] – encoded feature map
        """
        bs, c, h, w = src.shape

        src       = src.flatten(2).permute(2, 0, 1)          # → HW,B,C
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)    # → HW,B,C
        mask      = mask.flatten(1)                          # B,HW

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # Q,B,C
        tgt = jt.zeros_like(query_embed)                         # Q,B,C

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs     = self.decoder(tgt, memory,
                              memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), memory.permute(1, 2, 0).reshape(bs, c, h, w)

    # PyTorch‑style alias
    forward = execute

# =============================================================================
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm   = norm

    def execute(self, src,
                      mask: Optional[jt.Var] = None,
                      src_key_padding_mask: Optional[jt.Var] = None,
                      pos: Optional[jt.Var] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output

    forward = execute

# -----------------------------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm   = norm
        self.return_intermediate = return_intermediate

    # -------------------------------------------------------------------------
    def execute(self, tgt, memory,
                      tgt_mask: Optional[jt.Var] = None,
                      memory_mask: Optional[jt.Var] = None,
                      tgt_key_padding_mask: Optional[jt.Var] = None,
                      memory_key_padding_mask: Optional[jt.Var] = None,
                      pos: Optional[jt.Var] = None,
                      query_pos: Optional[jt.Var] = None):
        output       = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return jt.stack(intermediate)          # [L,B,Q,C]
        return output.unsqueeze(0)                 # [1,B,Q,C]

    forward = execute

# =============================================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation        = _get_activation_fn(activation)
        self.normalize_before  = normalize_before

    # -------------------------------------------------------------------------
    def with_pos_embed(self, tensor, pos: Optional[jt.Var]):
        return tensor if pos is None else tensor + pos

    # -- post‑norm version -----------------------------------------------------
    def _forward_post(self, src, src_mask, src_kpad_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, src,
                              attn_mask=src_mask,
                              key_padding_mask=src_kpad_mask)[0]
        src  = src + self.dropout1(src2)
        src  = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src  = src + self.dropout2(src2)
        src  = self.norm2(src)
        return src

    # -- pre‑norm version ------------------------------------------------------
    def _forward_pre(self, src, src_mask, src_kpad_mask, pos):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2,
                              attn_mask=src_mask,
                              key_padding_mask=src_kpad_mask)[0]
        src  = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src  = src + self.dropout2(src2)
        return src

    # -------------------------------------------------------------------------
    def execute(self, src,
                      src_mask: Optional[jt.Var] = None,
                      src_key_padding_mask: Optional[jt.Var] = None,
                      pos: Optional[jt.Var] = None):
        if self.normalize_before:
            return self._forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self._forward_post(src, src_mask, src_key_padding_mask, pos)

    forward = execute

# =============================================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn      = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation        = _get_activation_fn(activation)
        self.normalize_before  = normalize_before

    # -------------------------------------------------------------------------
    def with_pos_embed(self, tensor, pos: Optional[jt.Var]):
        return tensor if pos is None else tensor + pos

    # -- post‑norm -------------------------------------------------------------
    def _forward_post(self, tgt, memory,
                      tgt_mask, memory_mask,
                      tgt_kpad_mask, mem_kpad_mask,
                      pos, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_kpad_mask)[0]
        tgt  = tgt + self.dropout1(tgt2)
        tgt  = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key  =self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=mem_kpad_mask)[0]
        tgt  = tgt + self.dropout2(tgt2)
        tgt  = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt  = tgt + self.dropout3(tgt2)
        tgt  = self.norm3(tgt)
        return tgt

    # -- pre‑norm --------------------------------------------------------------
    def _forward_pre(self, tgt, memory,
                     tgt_mask, memory_mask,
                     tgt_kpad_mask, mem_kpad_mask,
                     pos, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_kpad_mask)[0]
        tgt  = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key  =self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=mem_kpad_mask)[0]
        tgt  = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt  = tgt + self.dropout3(tgt2)
        return tgt

    # -------------------------------------------------------------------------
    def execute(self, tgt, memory,
                      tgt_mask: Optional[jt.Var] = None,
                      memory_mask: Optional[jt.Var] = None,
                      tgt_key_padding_mask: Optional[jt.Var] = None,
                      memory_key_padding_mask: Optional[jt.Var] = None,
                      pos: Optional[jt.Var] = None,
                      query_pos: Optional[jt.Var] = None):
        if self.normalize_before:
            return self._forward_pre(tgt, memory,
                                     tgt_mask, memory_mask,
                                     tgt_key_padding_mask, memory_key_padding_mask,
                                     pos, query_pos)
        return self._forward_post(tgt, memory,
                                  tgt_mask, memory_mask,
                                  tgt_key_padding_mask, memory_key_padding_mask,
                                  pos, query_pos)

    forward = execute

# -----------------------------------------------------------------------------
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# -----------------------------------------------------------------------------
def build_transformer(args):
    return Transformer(
        d_model             = args.hidden_dim,
        dropout             = args.dropout,
        nhead               = args.nheads,
        dim_feedforward     = args.dim_feedforward,
        num_encoder_layers  = args.enc_layers,
        num_decoder_layers  = args.dec_layers,
        normalize_before    = args.pre_norm,
        return_intermediate_dec=True,
    )

# -----------------------------------------------------------------------------
def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Unsupported activation: {activation}")
