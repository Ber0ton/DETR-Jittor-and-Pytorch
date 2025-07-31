# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import jittor as jt           # ← Jittor replaces PyTorch
from PIL import Image

from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes          # must be Jittor‑compatible
from .coco import make_coco_transforms           # your Jittor version of the transforms


class CocoPanoptic(jt.dataset.Dataset):          # inherit to use jt.data.DataLoader
    """
    COCO‑style panoptic dataset implemented for Jittor.
    Every numeric field is returned as jt.Var, so downstream code can stay on GPU.
    """
    def __init__(self, img_folder, ann_folder, ann_file,
                 transforms=None, return_masks=True):
        super().__init__()

        # ----- load COCO‑Panoptic json -------------------------------------------------
        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        # sort images so they align with annotation order
        self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["id"])
        if "annotations" in self.coco:
            for img, ann in zip(self.coco["images"], self.coco["annotations"]):
                assert img["file_name"][:-4] == ann["file_name"][:-4]

        # ----- persistent members ------------------------------------------------------
        self.img_folder  = Path(img_folder)
        self.ann_folder  = Path(ann_folder)
        self.transforms  = transforms
        self.return_masks = return_masks

    # -------------------------------------------------------------------------------
    def __getitem__(self, idx):
        # pick the right json entry (train/val vs. test‑dev)
        ann_info = (self.coco["annotations"][idx]
                    if "annotations" in self.coco
                    else self.coco["images"][idx])

        img_path = self.img_folder / ann_info["file_name"].replace(".png", ".jpg")
        ann_path = self.ann_folder / ann_info["file_name"]

        # ------------------------------------------------------------------ 1. image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # ------------------------------------------------------------------ 2. masks & labels (if available)
        masks, labels = None, None
        if "segments_info" in ann_info:
            mask_rgb = np.asarray(Image.open(ann_path), dtype=np.uint32)
            mask_ids = rgb2id(mask_rgb)                      # H×W int32
            seg_ids  = np.asarray([ann["id"]
                                   for ann in ann_info["segments_info"]])
            masks_np = (mask_ids[None] == seg_ids[:, None, None])  # N×H×W bool
            masks   = jt.array(masks_np.astype(np.uint8))
            labels  = jt.array([ann["category_id"]
                                for ann in ann_info["segments_info"]],
                               dtype=jt.int64)

        # ------------------------------------------------------------------ 3. target dict
        target = {}
        target["image_id"] = jt.array([ann_info.get("image_id",
                                                    ann_info["id"])])
        if self.return_masks:
            target["masks"] = masks
        target["labels"] = labels
        target["boxes"]  = masks_to_boxes(masks)         # Jittor implementation

        size = jt.array([int(h), int(w)])
        target["size"]      = size
        target["orig_size"] = size

        if "segments_info" in ann_info:                  # extra coco keys
            for key in ("iscrowd", "area"):
                target[key] = jt.array([ann[key]
                                        for ann in ann_info["segments_info"]])

        # ------------------------------------------------------------------ 4. transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # -------------------------------------------------------------------------------
    def __len__(self):
        return len(self.coco["images"])

    # -------------------------------------------------------------------------------
    def get_height_and_width(self, idx):
        info = self.coco["images"][idx]
        return info["height"], info["width"]


# ===================================================================================
def build(image_set, args):
    """
    Construct a CocoPanoptic dataset for *train* or *val* split.

    Args
    ----
    image_set : {"train", "val"}
    args      : any object with
                - args.coco_path
                - args.coco_panoptic_path
                - args.masks (bool)         ← whether to return masks

    Returns
    -------
    CocoPanoptic
    """
    img_root = Path(args.coco_path)
    pan_root = Path(args.coco_panoptic_path)
    assert img_root.exists(), f"COCO path {img_root} not found"
    assert pan_root.exists(), f"COCO‑Panoptic path {pan_root} not found"

    mode = "panoptic"
    PATHS = {
        "train": ("train2017", Path("annotations") / f"{mode}_train2017.json"),
        "val"  : ("val2017",   Path("annotations") / f"{mode}_val2017.json"),
    }

    img_subdir, ann_json = PATHS[image_set]
    img_folder  = img_root / img_subdir
    ann_folder  = pan_root / f"{mode}_{img_subdir}"
    ann_file    = pan_root / ann_json

    dataset = CocoPanoptic(
        img_folder,
        ann_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),   # → your Jittor transforms
        return_masks=args.masks
    )
    return dataset
