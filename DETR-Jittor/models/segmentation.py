import io
from collections import defaultdict
from typing import List, Optional

import jittor as jt
import jittor.nn as nn
import jittor.nn as F
from PIL import Image

import util.box_ops as box_ops                       # ← Jittor port
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id        # optional (panoptic evaluation)
except ImportError:
    pass

# ===========================================================================
class DETRsegm(nn.Module):
    """
    Wrapper that adds a mask head on top of a DETR model.
    """

    def __init__(self, detr, freeze_detr: bool = False):
        super().__init__()
        self.detr = detr
        if freeze_detr:
            for p in self.detr.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head      = MaskHeadSmallConv(hidden_dim + nheads,
                                                [1024, 512, 256],
                                                hidden_dim)

    # ---------------------------------------------------------------------
    def execute(self, samples: NestedTensor):
        if isinstance(samples, (list, jt.Var)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()            # C5 (smallest spatial)

        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask,
                                           self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)       # [L,B,Q,#cls]
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes" : outputs_coord[-1],
        }
        if self.detr.aux_loss:
            out["aux_outputs"] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # bbox‑conditioned attention maps over the encoder memory
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)  # [B,Q,H,W]

        seg_masks = self.mask_head(
            src_proj,
            bbox_mask,
            [feat.tensors for feat in features[2::-1]]   # P3,P2,P1 for FPN up‑sampling
        )
        outputs_seg_masks = seg_masks.reshape(bs, self.detr.num_queries,
                                              seg_masks.shape[-2], seg_masks.shape[-1])
        out["pred_masks"] = outputs_seg_masks
        return out

    forward = execute

# ===========================================================================
def _expand(tensor: jt.Var, length: int):
    """Repeat along batch dimension and flatten B×Q."""
    return tensor.unsqueeze(1).repeat(1, length, 1, 1, 1).reshape(-1, *tensor.shape[1:])

# ----------------------------------------------------------------------------
class MaskHeadSmallConv(nn.Module):
    """
    Simple 5‑layer Conv+GN FPN‑style mask head (identical to DETR original).
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        inter_dims = [dim,
                      context_dim // 2,
                      context_dim // 4,
                      context_dim // 8,
                      context_dim // 16]

        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1  = nn.GroupNorm(8, dim)

        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2  = nn.GroupNorm(8, inter_dims[1])

        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3  = nn.GroupNorm(8, inter_dims[2])

        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4  = nn.GroupNorm(8, inter_dims[3])

        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5  = nn.GroupNorm(8, inter_dims[4])

        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        # FPN adapters
        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    # ---------------------------------------------------------------------
    def execute(self, x: jt.Var, bbox_mask: jt.Var, fpns: List[jt.Var]):
        """
        x         : encoder C5 feature map after proj (B,C,Hc,Wc)
        bbox_mask : attention weight maps [B*Q,1,Hc,Wc]  (via MHAttentionMap)
        fpns      : list [P3,P2,P1] – features for up‑sampling
        """
        x = jt.concat([_expand(x, bbox_mask.shape[1]),     # repeat image features per query
                       bbox_mask.reshape(-1, *bbox_mask.shape[2:])], dim=1)

        # ---- block 1 -----------------------------------------------------
        x = F.relu(self.gn1(self.lay1(x)))
        x = F.relu(self.gn2(self.lay2(x)))

        # Add P3
        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")

        # ---- block 2 -----------------------------------------------------
        x = F.relu(self.gn3(self.lay3(x)))

        # Add P2
        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")

        # ---- block 3 -----------------------------------------------------
        x = F.relu(self.gn4(self.lay4(x)))

        # Add P1
        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")

        # ---- block 4 -----------------------------------------------------
        x = F.relu(self.gn5(self.lay5(x)))
        x = self.out_lay(x)
        return x

    forward = execute

# ----------------------------------------------------------------------------
class MHAttentionMap(nn.Module):
    """
    2‑D multi‑head attention that returns only the attention weights
    (no value‑projection multiply). Used to produce per‑query spatial masks.
    """

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads   = num_heads
        self.hidden_dim  = hidden_dim
        self.normalize   = (hidden_dim / num_heads) ** -0.5
        self.dropout     = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.q_linear.bias)
        nn.init.zeros_(self.k_linear.bias)
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)

    # ---------------------------------------------------------------------
    def execute(self, q: jt.Var, k: jt.Var, mask: Optional[jt.Var] = None):
        """
        q : [B,Q,C]      – decoder output
        k : [B,C,H,W]    – encoder memory
        mask : [B,H,W] bool  (pad mask)
        returns : weights [B,Q,H,W]
        """
        q = self.q_linear(q)                             # B,Q,C
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1),
                        self.k_linear.bias)              # B,C,H,W

        qh = q.reshape(q.shape[0], q.shape[1], self.num_heads,
                       self.hidden_dim // self.num_heads)           # B,Q,Hd,Hc
        kh = k.reshape(k.shape[0], self.num_heads,
                       self.hidden_dim // self.num_heads,
                       k.shape[-2], k.shape[-1])                    # B,Hd,Hc,H,W

        weights = jt.einsum("bqnc,bnchw->bqnhw", qh * self.normalize, kh)

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        weights = F.softmax(weights.reshape(weights.shape[0], weights.shape[1],
                                            weights.shape[2], -1), dim=-1)
        weights = weights.reshape(weights.shape)         # restore H,W dims
        return self.dropout(weights)

    forward = execute

# ===========================================================================
# Loss utilities – unchanged except for Jittor ops
# ---------------------------------------------------------------------------
def dice_loss(inputs: jt.Var, targets: jt.Var, num_boxes: float):
    inputs  = inputs.sigmoid().reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    numerator   = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

# ----------------------------------------------------------------------------
def sigmoid_focal_loss(inputs: jt.Var, targets: jt.Var, num_boxes: float,
                       alpha: float = 0.25, gamma: float = 2.0):
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t     = prob * targets + (1 - prob) * (1 - targets)
    loss    = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss    = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

# ===========================================================================
# Post‑processing classes – these operate mostly on CPU / PIL, so minimal change
# ---------------------------------------------------------------------------
class PostProcessSegm(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    @jt.no_grad()
    def execute(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = map(int, max_target_sizes.max(0)[0])
        outputs_masks = outputs["pred_masks"].squeeze(2)             # [B,Q,H,W]
        outputs_masks = F.interpolate(outputs_masks,
                                      size=(max_h, max_w),
                                      mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).int()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = int(t[0]), int(t[1])
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(results[i]["masks"].float(),
                                                size=tuple(map(int, tt)),
                                                mode="nearest").int()
        return results

    forward = execute

# ----------------------------------------------------------------------------
class PostProcessPanoptic(nn.Module):
    """
    Convert model outputs into COCO‑panoptic PNG + segment‑info dict list.
    """

    def __init__(self, is_thing_map, threshold: float = 0.85):
        super().__init__()
        self.threshold     = threshold
        self.is_thing_map  = is_thing_map

    # ---------------------------------------------------------------------
    def execute(self, outputs, processed_sizes, target_sizes=None):
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)

        out_logits, raw_masks, raw_boxes = (outputs["pred_logits"],
                                            outputs["pred_masks"],
                                            outputs["pred_boxes"])
        preds = []

        def to_tuple(t):
            if isinstance(t, tuple):
                return t
            return tuple(map(int, t))

        for cur_logits, cur_masks, cur_boxes, size, tgt_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = (labels.ne(out_logits.shape[-1] - 1) &
                    (scores > self.threshold))

            cur_scores  = scores[keep]
            cur_classes = labels[keep]
            cur_masks   = cur_masks[keep]
            cur_masks   = interpolate(cur_masks[:, None], to_tuple(size),
                                      mode="bilinear").squeeze(1)
            cur_boxes   = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            cur_masks = cur_masks.reshape(cur_masks.shape[0], -1)

            stuff_equiv = defaultdict(list)
            for k, lab in enumerate(cur_classes):
                if not self.is_thing_map[lab.item()]:
                    stuff_equiv[lab.item()].append(k)

            def get_ids_area(masks, scrs, dedup=False):
                m_id = masks.transpose(0, 1).softmax(-1)
                if m_id.shape[-1] == 0:
                    m_id = jt.zeros((h, w), dtype=jt.int64)
                else:
                    m_id = m_id.argmax(-1).reshape(h, w)

                if dedup:
                    for eq in stuff_equiv.values():
                        if len(eq) > 1:
                            for e in eq:
                                m_id = m_id.where(m_id != e, jt.int32(eq[0]))

                final_h, final_w = to_tuple(tgt_size)
                seg_img = Image.fromarray(id2rgb(m_id.numpy()))
                seg_img = seg_img.resize((final_w, final_h), resample=Image.NEAREST)

                np_seg   = jt.array(seg_img).numpy()
                m_id_out = jt.array(rgb2id(np_seg))

                area = [int((m_id_out == i).sum()) for i in range(len(scrs))]
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                while True:
                    small = jt.array([a <= 4 for a in area]).bool()
                    if small.any():
                        cur_scores  = cur_scores[~small]
                        cur_classes = cur_classes[~small]
                        cur_masks   = cur_masks[~small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break
            else:
                cur_classes = jt.ones(1, dtype=jt.int64)

            segments_info = []
            for i, a in enumerate(area):
                cat = int(cur_classes[i])
                segments_info.append({
                    "id": i, "isthing": self.is_thing_map[cat],
                    "category_id": cat, "area": a
                })
            with io.BytesIO() as out_buf:
                seg_img.save(out_buf, format="PNG")
                preds.append({
                    "png_string": out_buf.getvalue(),
                    "segments_info": segments_info
                })
        return preds

    forward = execute
