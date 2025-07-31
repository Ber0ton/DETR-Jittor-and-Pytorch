import math, os, sys
from typing import Iterable, Dict

import jittor as jt
import util.misc as utils
from datasets.coco_eval import CocoEvaluator      
from datasets.panoptic_eval import PanopticEvaluator 
from jittor import optim


# ---------------------------------------------------------------------------
def train_one_epoch(model: jt.nn.Module,
                    criterion: jt.nn.Module,
                    data_loader: Iterable,
                    optimizer: jt.optim.Optimizer,
                    device: str,
                    epoch: int,
                    max_norm: float = 0.0,
                    print_freq: int = 10) -> Dict[str, float]:
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr",           utils.SmoothedValue(1, '{value:.6f}'))
    metric_logger.add_meter("class_error",  utils.SmoothedValue(1, '{value:.2f}'))
    header = f"Epoch: [{epoch}]"

    # ---------------------------------------------------------------------
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples  = samples.to(device)
        targets  = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs  = model(samples)
        loss_dict= criterion(outputs, targets)
        w_dict   = criterion.weight_dict

        losses   = sum(loss_dict[k] * w_dict[k] for k in loss_dict if k in w_dict)

        # ---- 分布式环境下聚合 loss ---
        loss_dict_red = utils.reduce_dict(loss_dict)
        loss_dict_red_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_red.items()}
        loss_dict_red_scaled   = {k: v * w_dict[k] for k, v in loss_dict_red.items() if k in w_dict}
        loss_val = float(sum(loss_dict_red_scaled.values()))

        if not math.isfinite(loss_val):
            print(f"Loss = {loss_val}, 停止训练")
            print(loss_dict_red)
            sys.exit(1)

        optimizer.zero_grad()
        optimizer.backward(losses)                 # ← 用优化器做 backward
        optimizer.clip_grad_norm(max_norm, 2)

        optimizer.step()

        metric_logger.update(loss=loss_val,
                             **loss_dict_red_scaled,
                             **loss_dict_red_unscaled)
        metric_logger.update(class_error=loss_dict_red["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["learning_rate"])
        jt.sync_all()
        jt.gc()

    # ---------------------------------------------------------------------
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}

@jt.no_grad()
def evaluate(model: jt.nn.Module,
             criterion: jt.nn.Module,
             postprocessors,
             data_loader: Iterable,
             base_ds,
             device: str,
             output_dir: str,
             print_freq: int = 10):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(1, '{value:.2f}'))
    header = "Test:"

    
    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors)
    coco_evaluator = CocoEvaluator(base_ds, iou_types) if len(iou_types) else None

    panoptic_evaluator = None
    if "panoptic" in postprocessors:
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            os.path.join(output_dir, "panoptic_eval")
        )

    # ----------------------------------------------
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples  = samples.to(device)
        targets  = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs  = model(samples)
        loss_dict= criterion(outputs, targets)
        w_dict   = criterion.weight_dict

        # --- 日志用的 reduce --------------------------------------------
        loss_dict_red = utils.reduce_dict(loss_dict)
        loss_dict_red_scaled   = {k: v * w_dict[k] for k, v in loss_dict_red.items() if k in w_dict}
        loss_dict_red_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_red.items()}

        metric_logger.update(loss = sum(loss_dict_red_scaled.values()),
                             **loss_dict_red_scaled,
                             **loss_dict_red_unscaled)
        metric_logger.update(class_error = loss_dict_red["class_error"])

        # --- 后处理 ------------------------------------------------------
        orig_sizes = jt.stack([t["orig_size"] for t in targets], dim=0)
        results    = postprocessors["bbox"](outputs, orig_sizes)

        if "segm" in postprocessors:
            tgt_sizes = jt.stack([t["size"] for t in targets], dim=0)
            results   = postprocessors["segm"](results, outputs, orig_sizes, tgt_sizes)
        
        res = {int(t["image_id"].item()): r for t, r in zip(targets, results)}
        if coco_evaluator:
            coco_evaluator.update(res)

        if panoptic_evaluator:
            res_pano = postprocessors["panoptic"](outputs, tgt_sizes, orig_sizes)
            for i, t in enumerate(targets):
                img_id = int(t["image_id"].item())
                res_pano[i]["image_id"]  = img_id
                res_pano[i]["file_name"] = f"{img_id:012d}.png"
            panoptic_evaluator.update(res_pano)
        jt.sync_all()
        jt.gc()
    # ---------------------------------------------------------------------
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator:      coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator:  panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = panoptic_evaluator.summarize() if panoptic_evaluator else None

    stats = {k: m.global_avg for k, m in metric_logger.meters.items()}
    if coco_evaluator:
        if "bbox" in postprocessors:
            stats["coco_eval_bbox"]  = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res:
        stats.update(PQ_all = panoptic_res["All"],
                     PQ_th  = panoptic_res["Things"],
                     PQ_st  = panoptic_res["Stuff"])
    return stats, coco_evaluator
