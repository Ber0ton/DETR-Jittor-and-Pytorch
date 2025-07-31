import os
os.environ["JITTOR_FLAGS"] = "log_silent_param_warning=1"

from typing import List, Dict

import jittor as jt
import jittor.nn as nn
import jittor.nn as F

import util.box_ops as box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                          accuracy, get_world_size, interpolate,
                          is_dist_avail_and_initialized)

from .backbone      import build_backbone
from .matcher       import build_matcher
from .segmentation  import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                               dice_loss, sigmoid_focal_loss)
from .transformer   import build_transformer



# -------------------------------------------------------------------------
# TorchScript‑style decorator stub (Jittor has no jt.jit)
# -------------------------------------------------------------------------
if not hasattr(jt, "jit"):
    class _DummyJit:
        @staticmethod
        def ignore(fn=None, *args, **kwargs):
            # supports both @jt.jit.ignore and @jt.jit.ignore()
            if fn is None:
                return lambda f: f        # decorator with parentheses
            return fn                     # decorator without parentheses
    jt.jit = _DummyJit()                 # register stub
# 1.  jt.device()  ---------------------------------------------------------
#    * Enables CUDA when the string starts with "cuda"
#    * Returns the same string so downstream .to(device) calls work
if not hasattr(jt, "device"):
    def _device(name="cpu"):
        name = str(name)
        if name.startswith("cuda"):
            jt.flags.use_cuda = 1
            # optional: handle explicit GPU index "cuda:1"
            if ":" in name:
                idx = int(name.split(":")[1])
                jt.cuda.set_device(idx)
            return "cuda"
        jt.flags.use_cuda = 0
        return "cpu"
    jt.device = _device                      # register stub

# 2.  nn.Module.to(dev)  ----------------------------------------------------
#     Minimal implementation: moves the whole module to GPU if dev starts
#     with "cuda", otherwise leaves it on CPU. Returns self for chaining.
if not hasattr(nn.Module, "to"):
    def _module_to(self, dev):
        if str(dev).startswith("cuda"):
            self.cuda()                      # Jittor’s built‑in move
        return self
    nn.Module.to = _module_to               # monkey‑patch base class
# -------------------------------------------------------------------------
# 1.  jt.Var.device  -------------------------------------------------------
#     Provides a PyTorch‑like .device property
if not hasattr(jt.Var, "device"):
    def _var_device(self):
        try:               # Jittor Var has .is_cuda() in recent builds
            return "cuda" if self.is_cuda() else "cpu"
        except AttributeError:
            return "cuda" if jt.flags.use_cuda else "cpu"
    jt.Var.device = property(_var_device)

# -------------------------------------------------------------------------
# 2.  jt.arange wrapper ----------------------------------------------------
#     Silently ignore unknown 'device' kwarg, keep all other behaviour.
if "_orig_arange" not in globals():     # don’t wrap twice
    _orig_arange = jt.arange

    def _arange(*args, **kwargs):
        kwargs.pop("device", None)      # drop PyTorch‑specific kwarg
        return _orig_arange(*args, **kwargs)

    jt.arange = _arange


# ===========================================================================
class DETR(nn.Module):
    """
    End‑to‑end object detector (DETR) – backbone + transformer + output heads.
    """

    def __init__(self, backbone, transformer,
                 num_classes: int, num_queries: int,
                 aux_loss: bool = False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim       = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj  = nn.Conv2d(backbone.num_channels, hidden_dim, 1)
        self.backbone    = backbone
        self.aux_loss    = aux_loss

    # ---------------------------------------------------------------------
    def execute(self, samples: NestedTensor):
        if isinstance(samples, (list, jt.Var)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        hs, _ = self.transformer(self.input_proj(src),
                                 mask,
                                 self.query_embed.weight,
                                 pos[-1])

        outputs_class = self.class_embed(hs)          # [L,B,Q,C+1]
        outputs_coord = self.bbox_embed(hs).sigmoid() # [L,B,Q,4]

        out = {
            "pred_logits": outputs_class[-1],         # last decoder layer
            "pred_boxes" : outputs_coord[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    forward = execute

    # ---------------------------------------------------------------------
    @jt.jit.ignore
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # drop last layer – keep intermediate decoder outputs
        return [{"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

# ===========================================================================
class SetCriterion(nn.Module):
    """
    Loss‑computation module for DETR (classification, L1 box, GIoU, mask, …).
    """

    def __init__(self, num_classes: int,
                       matcher,
                       weight_dict: Dict[str, float],
                       eos_coef: float,
                       losses: List[str]):
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher
        self.weight_dict = weight_dict
        self.eos_coef    = eos_coef
        self.losses      = losses

        empty_weight = jt.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # ---------------------------------------------------------------------
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs["pred_logits"]          # ← 同一个 Var 对象

        # ------- 正确地散/取匹配上的 query ---------------------------------
        num_queries = src_logits.shape[1]
        lin_idx = self._get_src_permutation_idx(indices, num_queries)

        target_classes = jt.full((src_logits.shape[0]*num_queries,),
                                 self.num_classes, dtype=jt.int64).to(src_logits)
        target_classes_o = jt.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[lin_idx] = target_classes_o

        logits_flat = src_logits.reshape(-1, src_logits.shape[-1])
        loss_ce = F.cross_entropy_loss(logits_flat, target_classes,
                                       weight=self.empty_weight, reduction="mean") 
        losses = {"loss_ce": loss_ce}
    
        if log:   # 只在“正样本”上算分类错误
            losses["class_error"] = 100 - accuracy(
                logits_flat[lin_idx], target_classes_o
            )[0]
        return losses
    # ---------------------------------------------------------------------
    @jt.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        # ② 取出分类 logits，形状 [B, Q, C+1]
        pred_logits = outputs['pred_logits']    
    
        # ③ 记录设备；之前在兼容层里给 jt.Var 加过 .device 属性
        device = pred_logits.device           
    
        # ④ 统计每张图的 GT 数量，并放到 logits 同一设备
        tgt_lengths = jt.array(
            [len(v["labels"]) for v in targets],        # Python 列表 → jt.Array
            dtype=jt.int32                              # 明确 dtype，Jittor 不推断 long
        ).to(pred_logits)                               # 与 logits 同设备
    
        # ⑤ 统计模型预测的前景 query 数
        #    - jt.argmax(dim=-1) 得到 [B, Q] 的类别索引
        #    - 与 no‑object 索引比较，结果是 bool Var
        #    - .sum(dim=1) 计算每张图的个数
        pred_classes = jt.argmax(pred_logits, dim=-1)[0]  #tuple 取前一个即可
        no_obj_idx = pred_logits.shape[-1] - 1          # no‑object 类索引
        temp = pred_classes != no_obj_idx
        card_pred  = (pred_classes != no_obj_idx).sum(dim=1)
    
        # ⑥ L1 绝对误差（转成 float32，再计算）
        card_err = nn.l1_loss(card_pred.float32(), tgt_lengths.float32())
    
        # ⑦ 返回字典形式，接口保持一致
        losses = {"cardinality_error": card_err}
        return losses


    # ---------------------------------------------------------------------
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        num_queries = outputs["pred_boxes"].shape[1]
        lin_idx = self._get_src_permutation_idx(indices, num_queries)
    
        src_boxes = outputs["pred_boxes"].reshape(-1, 4)[lin_idx]
        target_boxes = jt.concat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
    
        # ------ L1 loss 与官方一致 ------------------------------
        loss_bbox = (src_boxes - target_boxes).abs()     # shape (N,4)
        loss_bbox = loss_bbox.sum() / num_boxes          # sum 再 /N
    
        # ------ GIoU 保持不变 -----------------------------------
        giou = 1 - jt.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))
        return {
            "loss_bbox": loss_bbox,
            "loss_giou": giou.sum() / num_boxes,
        }


    # ---------------------------------------------------------------------
    def loss_masks(self, outputs, targets, indices, num_boxes):
        num_queries = outputs["pred_boxes"].shape[1]
        src_idx = self._get_src_permutation_idx(indices, num_queries)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"][src_idx]          # [ΣQ,Hs,Ws]
        masks = [t["masks"] for t in targets]
        tgt_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        tgt_masks = tgt_masks.to(src_masks)[tgt_idx]

        src_masks = interpolate(src_masks[:, None], size=tgt_masks.shape[-2:],
                                mode="bilinear", align_corners=False)[:, 0]

        loss_mask = sigmoid_focal_loss(src_masks, tgt_masks, num_boxes)
        loss_dice = dice_loss(src_masks, tgt_masks, num_boxes)
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    # ---------------------------------------------------------------------
    # helpers --------------------------------------------------------------
    # def _get_src_permutation_idx(self, indices):
    #     batch_idx = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
    #     src_idx   = jt.concat([src for src, _ in indices])
    #     return batch_idx, src_idx

    # def _get_tgt_permutation_idx(self, indices):
    #     batch_idx = jt.concat([jt.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    #     tgt_idx   = jt.concat([tgt for _, tgt in indices])
    #     return batch_idx, tgt_idx

    # 让 Jittor 避免笛卡儿积——把 (b,q) 二元索引 → 单一线性索引
    def _make_linear_idx(self, batch_idx, src_idx, num_queries):
        return batch_idx * num_queries + src_idx            # 1‑D Var

    def _get_src_permutation_idx(self, indices, num_queries=100):
        batch_idx = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = jt.concat([src for src, _ in indices])
        return self._make_linear_idx(batch_idx, src_idx, num_queries)

    def _get_tgt_permutation_idx(self, indices, num_queries=100):
        batch_idx = jt.concat([jt.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx   = jt.concat([tgt for _, tgt in indices])
        return self._make_linear_idx(batch_idx, tgt_idx, num_queries)

    # ---------------------------------------------------------------------
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # ---------------------------------------------------------------------
    def execute(self, outputs, targets):
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_no_aux, targets)

        # Normalisation factor (average #boxes across processes)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = jt.float32([num_boxes])
        if is_dist_avail_and_initialized():
            jt.distribute.all_reduce(num_boxes, op="add")
        num_boxes = jt.maximum(num_boxes / get_world_size(), jt.float32([1.0]))[0]

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                idxs = self.matcher(aux, targets)
                for loss in self.losses:
                    if loss == "masks":   # too heavy for every layer
                        continue
                    kwargs = {"log": False} if loss == "labels" else {}
                    l_dict = self.get_loss(loss, aux, targets, idxs, num_boxes, **kwargs)
                    losses.update({f"{k}_{i}": v for k, v in l_dict.items()})
        return losses

    forward = execute

# ===========================================================================
# models/detr_jt.py
class PostProcess(jt.nn.Module):
    """
    把网络输出转换成 COCO 评估需要的格式。
    与原 PyTorch 版保持一致：对每张图取 top‑100 个 (score, label, box)。
    """
    @jt.no_grad()
    def execute(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        prob = jt.nn.softmax(out_logits, dim=-1)[..., :-1]   # 去掉 no‑obj 类

        bs, nq, num_cls = prob.shape
        topk = 100
        # ① 直接在 (bs, nq*num_cls) 维度取 topk
        topk_scores, topk_idxs = jt.topk(prob.reshape(bs, -1), k=topk, dim=1)  # value, index

        labels = topk_idxs % num_cls                          # (bs, 100)
        query_idxs = topk_idxs // num_cls                     # (bs, 100)

        # ② 把对应的 box 也 gather 出来
        boxes = jt.gather(out_bbox, 1,                       # (bs, 100, 4)
                          query_idxs.unsqueeze(-1).repeat(1, 1, 4))

        # ③ 归一化坐标 → 绝对坐标
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
        scale = jt.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale[:, None, :]                    # broadcast

        # ④ 打包成 list[dict]
        results = []
        for s, l, b in zip(topk_scores, labels, boxes):
            results.append({
                "scores": s,      # (100,)
                "labels": l,      # (100,)
                "boxes":  b       # (100, 4)
            })
        return results

    forward = execute


# ===========================================================================
class MLP(nn.Module):
    """Simple M‑layer perceptron."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim_in  = in_dim if i == 0 else hidden_dim
            dim_out = out_dim if i == num_layers-1 else hidden_dim
            self.layers.append(nn.Linear(dim_in, dim_out))

    def execute(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    forward = execute

# ===========================================================================
def build(args):
    """
    Factory that assembles backbone, transformer, DETR head, criterion, and
    post‑processors according to CLI args (mirrors original training script).
    """
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    device = jt.device(args.device)

    backbone    = build_backbone(args)
    transformer = build_transformer(args)

    model = DETR(backbone, transformer,
                 num_classes=num_classes,
                 num_queries=args.num_queries,
                 aux_loss=args.aux_loss)
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce"   : 1.0,
        "loss_bbox" : args.bbox_loss_coef,
        "loss_giou" : args.giou_loss_coef,
    }
    if args.masks:
        weight_dict.update({
            "loss_mask": args.mask_loss_coef,
            "loss_dice": args.dice_loss_coef,
        })

    if args.aux_loss:
        aux_w = {f"{k}_{i}": v
                 for i in range(args.dec_layers - 1)
                 for k, v in weight_dict.items()}
        weight_dict.update(aux_w)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses.append("masks")

    criterion = SetCriterion(num_classes, matcher,
                             weight_dict, args.eos_coef, losses).to(device)

    post = {"bbox": PostProcess()}
    if args.masks:
        post["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            post["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model.to(device), criterion, post
