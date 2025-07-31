import jittor as jt

# ----------------------------------------------------------------------------
def box_cxcywh_to_xyxy(x: jt.Var):
    """
    Convert [cx,cy,w,h] → [x0,y0,x1,y1]
    """
    x_c, y_c, w, h = x.unbind(-1)            # four [N] tensors
    x0 = x_c - 0.5 * w
    y0 = y_c - 0.5 * h
    x1 = x_c + 0.5 * w
    y1 = y_c + 0.5 * h
    return jt.stack([x0, y0, x1, y1], dim=-1)

# ----------------------------------------------------------------------------
def box_xyxy_to_cxcywh(x: jt.Var):
    """
    Convert [x0,y0,x1,y1] → [cx,cy,w,h]
    """
    x0, y0, x1, y1 = x.unbind(-1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w  =  x1 - x0
    h  =  y1 - y0
    return jt.stack([cx, cy, w, h], dim=-1)

# ----------------------------------------------------------------------------
def box_area(boxes: jt.Var):
    """
    Compute area of boxes in [x0,y0,x1,y1] (no clamp).
    boxes : [...,4]
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

# ----------------------------------------------------------------------------
# IoU helpers ----------------------------------------------------------------
def box_iou(boxes1: jt.Var, boxes2: jt.Var):
    """
    Pairwise IoU between two sets of boxes.
    Returns
        iou   : [N,M]
        union : [N,M]
    """
    area1 = box_area(boxes1)                 # [N]
    area2 = box_area(boxes2)                 # [M]

    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])   # [N,M,2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])   # [N,M,2]

    wh = (rb - lt).clamp(min_v=0)                          # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]                        # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min_v=1e-6)
    return iou, union

# ----------------------------------------------------------------------------
def generalized_box_iou(boxes1: jt.Var, boxes2: jt.Var):
    """
    Generalised IoU (GIoU) from https://giou.stanford.edu/.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min_v=0)
    area = wh[..., 0] * wh[..., 1]

    return iou - (area - union) / area.clamp(min_v=1e-6)

# ----------------------------------------------------------------------------
def masks_to_boxes(masks: jt.Var):
    """
    Compute tight bounding boxes from binary masks.
    masks : [N,H,W] uint8 / bool / float
    Returns
        boxes : [N,4] in xyxy
    """
    if masks.numel() == 0:
        return jt.zeros((0, 4), dtype=masks.dtype)

    N, h, w = masks.shape
    device  = masks.device

    ys = jt.arange(h, dtype=jt.float32, device=device)
    xs = jt.arange(w, dtype=jt.float32, device=device)
    ys, xs = jt.meshgrid(ys, xs)             # H,W

    xs = xs.unsqueeze(0).expand(N, -1, -1)   # N,H,W
    ys = ys.unsqueeze(0).expand(N, -1, -1)

    x_mask = masks * xs
    x_max  = x_mask.reshape(N, -1).max(1)[0]
    x_min  = x_mask.masked_fill(~masks.bool(), 1e8).reshape(N, -1).min(1)[0]

    y_mask = masks * ys
    y_max  = y_mask.reshape(N, -1).max(1)[0]
    y_min  = y_mask.masked_fill(~masks.bool(), 1e8).reshape(N, -1).min(1)[0]

    return jt.stack([x_min, y_min, x_max, y_max], dim=1)
