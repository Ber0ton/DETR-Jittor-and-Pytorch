import jittor as jt
import jittor.nn as nn
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou   # ← Jittor port

# -----------------------------------------------------------------------------
class HungarianMatcher(nn.Module):
    """
    Computes a 1‑to‑1 bipartite matching between **predictions** and **targets**.

    cost = w_cls * CE + w_bbox * L1(box) + w_giou * (1 − GIoU)
    """

    def __init__(self, cost_class: float = 1.0,
                       cost_bbox : float = 1.0,
                       cost_giou : float = 1.0):
        super().__init__()
        assert cost_class or cost_bbox or cost_giou, "all costs cannot be zero!"
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    # -------------------------------------------------------------------------
    @jt.no_grad()
    def execute(self, outputs: dict, targets: list):
        """
        Args
        ----
        outputs : dict with
            • pred_logits : [B, Q, C]
            • pred_boxes  : [B, Q, 4] (cx,cy,w,h)
        targets : list(length B) of dicts:
            • labels : [N_i]
            • boxes  : [N_i, 4]
        Returns
        -------
        list of size B containing tuples (idx_pred, idx_tgt) – both 1‑D int64 tensors
        """
        bs, num_queries, num_classes = outputs["pred_logits"].shape

        # ---------------------------------------------------------------------
        # Flatten predictions
        out_prob = nn.softmax(
            outputs["pred_logits"].reshape(bs * num_queries, num_classes), dim=-1
        )                                               # [B*Q, C]

        out_bbox = outputs["pred_boxes"].reshape(bs * num_queries, 4)   # [B*Q, 4]

        # ---------------------------------------------------------------------
        # Concatenate targets
        tgt_ids   = jt.concat([v["labels"] for v in targets], dim=0)    # [ΣN]
        tgt_boxes = jt.concat([v["boxes"]  for v in targets], dim=0)    # [ΣN,4]

        # ---------------------------------------------------------------------
        # Classification cost: −P(class)
        cost_class = -out_prob[:, tgt_ids]                               # [B*Q, ΣN]

        # L1 box cost
        # pairwise |p - t|₁  → broadcast: (N_pred, 1, 4) − (1, N_tgt, 4)
        diff = jt.abs(out_bbox.unsqueeze(1) - tgt_boxes.unsqueeze(0))    # [B*Q, ΣN,4]
        cost_bbox = diff.sum(-1)                                         # [B*Q, ΣN]

        # −GIoU cost
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),   # [B*Q,4]
            box_cxcywh_to_xyxy(tgt_boxes)   # [ΣN,4]
        )                                   # → [B*Q, ΣN]
        cost_giou = -giou

        # ---------------------------------------------------------------------
        # Weighted sum
        C = (self.cost_class * cost_class +
             self.cost_bbox  * cost_bbox  +
             self.cost_giou  * cost_giou)                      # [B*Q, ΣN]
        C = C.reshape(bs, num_queries, -1).numpy()             # to NumPy for SciPy

        # ---------------------------------------------------------------------
        sizes   = [len(v["boxes"]) for v in targets]           # per‑image N_i
        splits  = []
        start   = 0
        for s in sizes:                                        # manual split
            splits.append(C[:, :, start:start+s])
            start += s

        indices = []
        for i, cost_sub in enumerate(splits):                  # cost_sub: [B,Q,N_i]
            if cost_sub.shape[2] == 0:                         # no targets in img
                indices.append((jt.int64([]), jt.int64([])))
                continue
            row_ind, col_ind = linear_sum_assignment(cost_sub[i])
            indices.append((
                jt.int64(row_ind), jt.int64(col_ind)
            ))

        return indices

    forward = execute

# -----------------------------------------------------------------------------
def build_matcher(args):
    """
    Helper used by training script.  Expects args to have
    • set_cost_class
    • set_cost_bbox
    • set_cost_giou
    """
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox =args.set_cost_bbox,
                            cost_giou =args.set_cost_giou)
