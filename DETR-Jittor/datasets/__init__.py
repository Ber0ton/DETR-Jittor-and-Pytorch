# datasets/__init__.py  (torch / torchvision free)
# ---------------------------------------------------------------------------
from typing import Any

from .coco import build as build_coco           # 早前改写的纯 Python 版
# 若使用 panoptic，可继续延迟导入
# from .coco_panoptic import build as build_coco_panoptic


def get_coco_api_from_dataset(dataset: Any):
    """
    递归向下剥离 `.dataset` 属性，直到找到拥有 `.coco` 成员的对象。
    这样就不需要知道外层包装类的具体类型（Subset、ConcatDataset 等）。
    """
    for _ in range(10):                         # 最多展开 10 层，防止死循环
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        else:
            break

    return getattr(dataset, "coco", None)       # 若不存在 .coco，返回 None


# ---------------------------------------------------------------------------
def build_dataset(image_set: str, args):
    """
    根据参数构建数据集。支持:
      - coco
      - coco_panoptic (可选，需另外移植 coco_panoptic.py)
    """
    if args.dataset_file == "coco":
        return build_coco(image_set, args)

    if args.dataset_file == "coco_panoptic":
        from .coco_panoptic import build as build_coco_panoptic  # 懒加载
        return build_coco_panoptic(image_set, args)

    raise ValueError(f"dataset {args.dataset_file} not supported")
