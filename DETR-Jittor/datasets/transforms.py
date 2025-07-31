import random, math
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
import jittor as jt

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate

Tensor = jt.Var   # 简单别名，保持与旧代码一致

# ===========================================================================
# --- 基础几何操作 ----------------------------------------------------------
# ===========================================================================

def _crop_pil(img: Image.Image, top, left, height, width):
    return img.crop((left, top, left + width, top + height))

def _hflip_pil(img: Image.Image):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def _resize_pil(img: Image.Image, size: Tuple[int,int]):
    return img.resize(size[::-1], Image.BILINEAR)

def _pad_pil(img: Image.Image, pad: Tuple[int,int]):
    # pad = (right, bottom)
    w, h = img.size
    new  = Image.new(img.mode, (w + pad[0], h + pad[1]), 0)
    new.paste(img, (0,0))
    return new

# ===========================================================================
# --- 带 target 更新的高阶函数 ----------------------------------------------
# ===========================================================================

def crop(image, target, region):
    top, left, height, width = region
    image = _crop_pil(image, top, left, height, width)

    target = target.copy()
    target["size"] = jt.array([height, width])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"] - jt.array([left, top, left, top])
        # 交到 [0,w]×[0,h]
        max_size = jt.array([width, height]).float32()
        boxes = jt.clamp(boxes.reshape(-1,2,2), min_v=0)
        boxes = jt.minimum(boxes, max_size)
        area  = (boxes[:,1,:] - boxes[:,0,:]).prod(dim=1)
        target.update(boxes = boxes.reshape(-1,4), area = area)
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, top:top+height, left:left+width]
        fields.append("masks")

    # 移除空对象
    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            b = target["boxes"].reshape(-1,2,2)
            keep = jt.all(b[:,1,:] > b[:,0,:], dim=1)
        else:
            keep = target["masks"].reshape(target["masks"].shape[0], -1).any(1)
        for f in fields:
            target[f] = target[f][keep]

    return image, target


def hflip(image, target):
    image = _hflip_pil(image)
    w, _ = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2,1,0,3]] * jt.array([-1,1,-1,1]) + jt.array([w,0,w,0])
        target["boxes"] = boxes
    if "masks" in target:
        target["masks"] = jt.flip(target["masks"], dims=[-1])
    return image, target


def resize(image, target, size, max_size=None):
    # size: int(min_side) or (h,w)
    def _get_size(orig, size):
        w0,h0 = orig
        if isinstance(size, Sequence):
            return tuple(size)
        # 保持长宽比
        min_orig, max_orig = min(h0,w0), max(h0,w0)
        if max_size and max_orig/min_orig*size > max_size:
            size = int(round(max_size * min_orig / max_orig))
        if w0 < h0:
            ow, oh = size, int(size*h0/w0)
        else:
            oh, ow = size, int(size*w0/h0)
        return oh, ow

    new_h, new_w = _get_size(image.size, size)
    image_resized = _resize_pil(image, (new_h,new_w))

    if target is None:
        return image_resized, None

    ratio_w, ratio_h = new_w / image.size[0], new_h / image.size[1]
    target = target.copy()
    if "boxes" in target:
        target["boxes"] = target["boxes"] * jt.array([ratio_w,ratio_h,ratio_w,ratio_h])
    if "area" in target:
        target["area"] = target["area"] * (ratio_h*ratio_w)
    target["size"] = jt.array([new_h, new_w])
    if "masks" in target:
        target["masks"] = interpolate(target["masks"][:,None].float32(),
                                      (new_h,new_w), mode="nearest")[:,0] > 0.5
    return image_resized, target


def pad(image, target, padding):
    image = _pad_pil(image, padding)
    if target is None:
        return image, None
    target = target.copy()
    target["size"] = jt.array(image.size[::-1])
    if "masks" in target:
        hpad, wpad = padding[1], padding[0]
        target["masks"] = jt.nn.pad(target["masks"],
                                    (0, wpad, 0, hpad))
    return image, target

# ===========================================================================
# --- Transform 组件 ---------------------------------------------------------
# ===========================================================================

class RandomCrop:
    def __init__(self, size): self.size=size
    def __call__(self,img,t):
        region = _get_random_crop(img, self.size)
        return crop(img,t,region)

def _get_random_crop(img:Image.Image, size):
    th, tw = size
    w,h = img.size
    if w==tw and h==th: return (0,0,h,w)
    i = random.randint(0, h-th)
    j = random.randint(0, w-tw)
    return (i,j,th,tw)

class RandomSizeCrop:
    def __init__(self,min_size,max_size):
        self.min,self.max=min_size,max_size
    def __call__(self,img,t):
        w = random.randint(self.min,min(img.width, self.max))
        h = random.randint(self.min,min(img.height,self.max))
        region = _get_random_crop(img,(h,w))
        return crop(img,t,region)

class CenterCrop:
    def __init__(self,size): self.size=size
    def __call__(self,img,t):
        w,h = img.size
        ch,cw = self.size
        top   = int(round((h-ch)/2))
        left  = int(round((w-cw)/2))
        return crop(img,t,(top,left,ch,cw))

class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p=p
    def __call__(self,img,t):
        return hflip(img,t) if random.random()<self.p else (img,t)

class RandomResize:
    def __init__(self,sizes,max_size=None):
        self.sizes = sizes; self.max_size=max_size
    def __call__(self,img,t):
        size = random.choice(self.sizes)
        return resize(img,t,size,self.max_size)

class RandomPad:
    def __init__(self,max_pad): self.max=max_pad
    def __call__(self,img,t):
        pad_x = random.randint(0,self.max)
        pad_y = random.randint(0,self.max)
        return pad(img,t,(pad_x,pad_y))

class RandomSelect:
    def __init__(self, t1, t2, p=0.5):
        self.t1=t1; self.t2=t2; self.p=p
    def __call__(self,img,t):
        return self.t1(img,t) if random.random()<self.p else self.t2(img,t)

class ToTensor:
    def __call__(self,img,t):
        arr = np.asarray(img, dtype=np.float32).transpose(2,0,1) / 255.0
        return jt.array(arr), t

class Normalize:
    def __init__(self, mean, std):
        self.mean = jt.array(mean).reshape(-1,1,1)
        self.std  = jt.array(std ).reshape(-1,1,1)
    def __call__(self,img,t):
        img = (img - self.mean) / self.std
        if t is None: return img,None
        t = t.copy()
        h,w = img.shape[-2:]
        if "boxes" in t:
            boxes = box_xyxy_to_cxcywh(t["boxes"])
            t["boxes"] = boxes / jt.array([w,h,w,h])
        return img,t

# 简易随机擦除（可选）
class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02,0.33)):
        self.p=p; self.scale=scale
    def __call__(self,img,t):
        if random.random()>self.p: return img,t
        c,h,w = img.shape
        area = h*w
        erase_area = random.uniform(*self.scale)*area
        ratio = random.uniform(0.3, 3.3)
        h_erase = int(round(math.sqrt(erase_area*ratio)))
        w_erase = int(round(math.sqrt(erase_area/ratio)))
        if h_erase==0 or w_erase==0: return img,t
        x1 = random.randint(0, h-h_erase)
        y1 = random.randint(0, w-w_erase)
        img[:, x1:x1+h_erase, y1:y1+w_erase] = 0.0
        return img,t

class Compose:
    def __init__(self, ts): self.transforms=list(ts)
    def __call__(self,img,t):
        for tr in self.transforms:
            img,t = tr(img,t)
        return img,t
    def __repr__(self):
        return "Compose(\n  " + "\n  ".join(str(t) for t in self.transforms) + "\n)"

# ===========================================================================
# helper to export -----------------------------------------------------------
__all__ = [
    "RandomCrop","RandomSizeCrop","CenterCrop","RandomHorizontalFlip",
    "RandomResize","RandomPad","RandomSelect","ToTensor",
    "Normalize","RandomErasing","Compose"
]
