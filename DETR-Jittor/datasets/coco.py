from pathlib import Path
import random
import math

import jittor as jt
from jittor.dataset import Dataset
import numpy as np
from PIL import Image
from util.misc import nested_tensor_from_tensor_list
from util import box_ops

from pycocotools.coco import COCO
import pycocotools.mask as coco_mask

# ----------------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------------
def convert_coco_poly_to_mask(segmentations, h, w):
    """Convert COCO polygon annotations to binary masks"""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, h, w)
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8).any(axis=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, h, w), dtype=np.uint8)
    return masks

# ----------------------------------------------------------------------------
# Transforms - Now matching PyTorch's implementation more closely
# ----------------------------------------------------------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class ToTensor:
    """Convert PIL image to Jittor tensor - should be called AFTER PIL transforms"""
    def __call__(self, img, target):
        # Convert PIL to numpy then to jittor
        img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return jt.array(img), target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, img, target):
        # img is already a tensor at this point
        mean = jt.array(self.mean).reshape(-1, 1, 1)
        std = jt.array(self.std).reshape(-1, 1, 1)
        img = (img - mean) / std
        return img, target

class RandomHorizontalFlip:
    """Horizontal flip on PIL images"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, target):
        if random.random() < self.p:
            # For PIL images
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Update bounding boxes
            w, h = img.size
            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
            
            # Update masks if present
            if "masks" in target:
                target["masks"] = target["masks"][:, :, ::-1]
            
            # Update keypoints if present
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints[..., 0] = w - keypoints[..., 0]
                target["keypoints"] = keypoints
        
        return img, target

class RandomResize:
    """Resize PIL images"""
    def __init__(self, sizes, max_size=None):
        if isinstance(sizes, int):
            sizes = [sizes]
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, img, target):
        size = random.choice(self.sizes)
        return self.resize(img, target, size, self.max_size)
    
    @staticmethod
    def resize(img, target, size, max_size=None):
        w, h = img.size
        
        # Calculate new size
        short_side = min(h, w)
        scale = size / short_side
        
        if max_size is not None:
            long_side = max(h, w)
            if long_side * scale > max_size:
                scale = max_size / long_side
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize PIL image efficiently
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Update target
        ratio_w = new_w / w
        ratio_h = new_h / h
        
        # Resize boxes
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes * jt.array([ratio_w, ratio_h, ratio_w, ratio_h])
            target["boxes"] = boxes
        
        # Resize masks
        if "masks" in target:
            masks = target["masks"]
            # Convert to PIL for efficient resizing
            masks_pil = [Image.fromarray(m.numpy().astype(np.uint8) * 255) 
                        for m in masks]
            masks_resized = [m.resize((new_w, new_h), Image.NEAREST) 
                           for m in masks_pil]
            masks = jt.array(np.stack([np.array(m) > 128 for m in masks_resized]))
            target["masks"] = masks
        
        # Update keypoints
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints[..., 0] *= ratio_w
            keypoints[..., 1] *= ratio_h
            target["keypoints"] = keypoints
        
        # Update size
        target["size"] = jt.array([new_h, new_w])
        
        return img, target

class RandomSizeCrop:
    """Random crop for PIL images"""
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, img, target):
        w, h = img.size
        crop_h = random.randint(self.min_size, min(h, self.max_size))
        crop_w = random.randint(self.min_size, min(w, self.max_size))
        
        # Random crop position
        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        
        # Crop PIL image
        img = img.crop((j, i, j + crop_w, i + crop_h))
        
        # Update target
        crop_box = jt.array([j, i, j + crop_w, i + crop_h])
        
        # Crop boxes
        boxes = target["boxes"]
        cropped_boxes = boxes - jt.array([j, i, j, i])
        cropped_boxes = jt.clamp(cropped_boxes, min_v=0)
        cropped_boxes[:, [0, 2]] = jt.clamp(cropped_boxes[:, [0, 2]], max_v=crop_w)
        cropped_boxes[:, [1, 3]] = jt.clamp(cropped_boxes[:, [1, 3]], max_v=crop_h)
        
        # Keep valid boxes
        keep = (cropped_boxes[:, 2] > cropped_boxes[:, 0]) & \
               (cropped_boxes[:, 3] > cropped_boxes[:, 1])
        
        target["boxes"] = cropped_boxes[keep]
        target["labels"] = target["labels"][keep]
        
        if "area" in target:
            target["area"] = target["area"][keep]
        if "iscrowd" in target:
            target["iscrowd"] = target["iscrowd"][keep]
        
        # Crop masks
        if "masks" in target:
            masks = target["masks"]
            masks = masks[:, i:i+crop_h, j:j+crop_w]
            target["masks"] = masks[keep]
        
        # Crop keypoints
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = keypoints - jt.array([j, i, 0])
            target["keypoints"] = keypoints[keep]
        
        target["size"] = jt.array([crop_h, crop_w])
        
        return img, target

class RandomSelect:
    """Randomly select one transform to apply"""
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
    
    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

# ----------------------------------------------------------------------------
# Transform builders
# ----------------------------------------------------------------------------
def make_coco_transforms(image_set):
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    
    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            RandomSelect(
                RandomResize(scales, max_size=1333),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(384, 600),
                    RandomResize(scales, max_size=1333),
                ])
            ),
            ToTensor(),
            normalize,
        ])
    
    if image_set == 'val':
        return Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            normalize,
        ])
    
    raise ValueError(f'unknown {image_set}')

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super().__init__()
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Category mapping
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat2label = {cat['id']: i for i, cat in enumerate(cats)}
        
        self._transforms = transforms
        self.return_masks = return_masks
        
        self.set_attrs(total_len=len(self.ids))
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Load image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(self.img_folder / path).convert('RGB')
        
        # Filter crowd annotations
        anns = [obj for obj in anns if obj.get('iscrowd', 0) == 0]
        
        # Extract annotations
        w, h = img.size
        boxes = []
        classes = []
        areas = []
        iscrowd = []
        segmentations = []
        keypoints = []
        
        for obj in anns:
            xmin, ymin, width, height = obj['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            classes.append(self.cat2label[obj['category_id']])
            areas.append(obj['area'])
            iscrowd.append(obj.get('iscrowd', 0))
            
            if self.return_masks:
                segmentations.append(obj['segmentation'])
            
            if 'keypoints' in obj:
                keypoints.append(obj['keypoints'])
        
        # Convert to tensors
        boxes = jt.array(boxes, dtype=jt.float32).reshape(-1, 4)
        boxes[:, 0::2] = jt.clamp(boxes[:, 0::2], min_v=0, max_v=w)
        boxes[:, 1::2] = jt.clamp(boxes[:, 1::2], min_v=0, max_v=h)
        
        classes = jt.array(classes, dtype=jt.int64)
        
        target = {
            'boxes': boxes,
            'labels': classes,
            'image_id': jt.array([img_id]),
            'area': jt.array(areas),
            'iscrowd': jt.array(iscrowd),
            'orig_size': jt.array([h, w]),
            'size': jt.array([h, w])
        }
        
        if self.return_masks:
            target['masks'] = jt.array(convert_coco_poly_to_mask(segmentations, h, w))
        
        if keypoints:
            target['keypoints'] = jt.array(keypoints, dtype=jt.float32).reshape(len(keypoints), -1, 3)
        
        # Remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        for k in ['boxes', 'labels', 'area', 'iscrowd']:
            target[k] = target[k][keep]
        
        if self.return_masks and 'masks' in target:
            target['masks'] = target['masks'][keep]
        
        if 'keypoints' in target:
            target['keypoints'] = target['keypoints'][keep]
        
        # Apply transforms
        img, target = self._transforms(img, target)
        
        # Convert boxes to cxcywh format (after all transforms)
        h, w = target['size'].numpy()
        boxes = target['boxes']
        boxes = boxes / jt.array([w, h, w, h])
        boxes = box_ops.box_xyxy_to_cxcywh(boxes)
        target['boxes'] = boxes
        
        return img, target
    
    def collate_batch(self, batch):
        """Custom collate function to handle different image sizes"""
        imgs, targets = zip(*batch)
        imgs = nested_tensor_from_tensor_list(list(imgs))
        return imgs, list(targets)

# ----------------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------------
def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder, ann_file, 
        transforms=make_coco_transforms(image_set), 
        return_masks=args.masks
    )
    return dataset