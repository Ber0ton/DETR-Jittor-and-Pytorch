import os, copy, contextlib
import numpy as np
import jittor as jt

from pycocotools.cocoeval import COCOeval
from pycocotools.coco      import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather          

# ===========================================================================
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        self.coco_gt   = copy.deepcopy(coco_gt)   # 不要改原始 COCO 对象
        self.iou_types = list(iou_types)

        self.coco_eval = {t: COCOeval(self.coco_gt, iouType=t)
                          for t in self.iou_types}

        self.img_ids   = []                       # 收集本进程见过的 imgId
        self.eval_imgs = {t: [] for t in self.iou_types}

    # ---------------------------------------------------------------------
    @jt.no_grad()                                # 评估阶段无梯度
    def update(self, predictions):
        """predictions: Dict[int, Dict[str, jt.Var]]，键是 image_id"""
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # pycocotools 在 loadRes 时会打印信息；用 redirect_stdout 静默
            with open(os.devnull, "w") as devnull, \
                 contextlib.redirect_stdout(devnull):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt       = coco_dt
            coco_eval.params.imgIds = img_ids

            img_ids_, eval_imgs = evaluate(coco_eval)   # ↓ 下方保留原函数
            self.eval_imgs[iou_type].append(eval_imgs)

    # ---------------------------------------------------------------------
    def synchronize_between_processes(self):
        """多进程 (DDP) 汇总：把各 GPU 的 eval_imgs 拼起来"""
        for t in self.iou_types:
            self.eval_imgs[t] = np.concatenate(self.eval_imgs[t], 2)
            create_common_coco_eval(self.coco_eval[t],
                                    self.img_ids,
                                    self.eval_imgs[t])

    def accumulate(self):
        for e in self.coco_eval.values():
            e.accumulate()

    def summarize(self):
        for t, e in self.coco_eval.items():
            print(f"IoU metric: {t}")
            e.summarize()

    # ---------------------------------------------------------------------
    # ---------- 准备结果 ---------------------------------------------------
    def prepare(self, preds, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(preds)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(preds)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(preds)
        raise ValueError(f"Unknown iou_type {iou_type}")

    # ---- bbox ------------------------------------------------------------
    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for img_id, pred in predictions.items():
            if len(pred) == 0:
                continue
            boxes  = convert_to_xywh(pred["boxes"]).tolist()
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()
    
            coco_results.extend(
                dict(image_id=img_id,
                     category_id=labels[k],  # FIXED: Direct use of labels[k]
                     bbox=boxes[k],
                     score=scores[k])
                for k in range(len(boxes))
            )
        return coco_results

    # ---- segm ------------------------------------------------------------
    def prepare_for_coco_segmentation(self, preds):
        coco_results = []
        for img_id, pred in preds.items():
            if len(pred) == 0:
                continue
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()
            masks  = (pred["masks"] > 0.5)

            rles = [mask_util.encode(
                        np.asarray(m[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                    for m in masks]
            for r in rles:
                r["counts"] = r["counts"].decode("utf-8")   # bytes → str

            coco_results.extend(
                dict(image_id=img_id,
                     category_id=labels[k],  # FIXED: Direct use of labels[k]
                     segmentation=rles[k],
                     score=scores[k])
                for k in range(len(rles))
            )
        return coco_results

    # ---- keypoint --------------------------------------------------------
    def prepare_for_coco_keypoint(self, preds):
        coco_results = []
        for img_id, pred in preds.items():
            if len(pred) == 0:
                continue
            boxes  = convert_to_xywh(pred["boxes"]).numpy().tolist()
            scores = pred["scores"].flatten().numpy().tolist()
            labels = pred["labels"].flatten().numpy().tolist()
            kpts   = pred["keypoints"].reshape(pred["keypoints"].shape[0], -1).tolist()

            coco_results.extend(
                dict(image_id=img_id,
                     category_id=labels[k],  # FIXED: Direct use of labels[k]
                     keypoints=kpts[k],
                     score=scores[k])
                for k in range(len(kpts))
            )
        return coco_results

# ===========================================================================
# ----------------------- 工具函数 ------------------------------------------
def convert_to_xywh(boxes: jt.Var) -> jt.Var:
    """把 [xmin, ymin, xmax, ymax] → [x, y, w, h]"""
    xmin, ymin, xmax, ymax = jt.unbind(boxes, dim=1)
    return jt.stack([xmin,
                     ymin,
                     xmax - xmin,
                     ymax - ymin], dim=1)

# ---- 分布式拼接 -----------------------------------------------------------
def merge(img_ids, eval_imgs):
    all_img_ids  = all_gather(img_ids)      # list[list[int]]
    all_eval_imgs= all_gather(eval_imgs)    # list[np.ndarray]

    merged_img_ids  = np.concatenate(all_img_ids)
    merged_eval_imgs= np.concatenate(all_eval_imgs, 2)

    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs    = merged_eval_imgs[..., idx]
    return merged_img_ids, merged_eval_imgs

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    coco_eval.evalImgs       = list(eval_imgs.flatten())
    coco_eval.params.imgIds  = list(img_ids)
    coco_eval._paramsEval    = copy.deepcopy(coco_eval.params)

# ===========================================================================
def evaluate(self):
    p = self.params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    catIds = p.catIds if p.useCats else [-1]

    computeIoU = {'bbox': self.computeIoU,
                  'segm': self.computeIoU,
                  'keypoints': self.computeOks}[p.iouType]
    self.ious = {(imgId, catId): computeIoU(imgId, catId)
                 for imgId in p.imgIds
                 for catId in catIds}

    maxDet   = p.maxDets[-1]
    evalImgs = [self.evaluateImg(imgId, catId, aRng, maxDet)
                for catId in catIds
                for aRng  in p.areaRng
                for imgId in p.imgIds]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs