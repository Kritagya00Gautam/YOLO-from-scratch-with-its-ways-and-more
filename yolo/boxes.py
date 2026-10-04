"""Box geometry: format conversion, IoU / CIoU, anchor decoding, NMS."""
import math
import torch


def xywh2xyxy(x):
    """[..., (cx, cy, w, h)] -> [..., (x1, y1, x2, y2)]"""
    y = torch.empty_like(x)
    half = x[..., 2:4] / 2
    y[..., :2] = x[..., :2] - half
    y[..., 2:4] = x[..., :2] + half
    return y


def xyxy2xywh(x):
    y = torch.empty_like(x)
    y[..., :2] = (x[..., :2] + x[..., 2:4]) / 2
    y[..., 2:4] = x[..., 2:4] - x[..., :2]
    return y


def box_area(b):
    return (b[..., 2] - b[..., 0]).clamp(0) * (b[..., 3] - b[..., 1]).clamp(0)


def box_iou(a, b, eps=1e-7):
    """Pairwise IoU. a: [N,4], b: [M,4] (xyxy) -> [N,M]."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(2)
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter + eps)


def bbox_iou(b1, b2, ciou=False, eps=1e-7):
    """Element-wise IoU (or CIoU) of two broadcastable xyxy box tensors [...,4].

    CIoU = IoU - centre_distance^2 / enclosing_diag^2 - alpha * v
      * second term: penalises boxes whose centres are far apart
      * v: penalises aspect-ratio mismatch
    It still gives a useful gradient when boxes do not overlap at all.
    """
    x1a, y1a, x2a, y2a = b1.unbind(-1)
    x1b, y1b, x2b, y2b = b2.unbind(-1)
    wa, ha = x2a - x1a, y2a - y1a + eps
    wb, hb = x2b - x1b, y2b - y1b + eps

    inter = ((torch.min(x2a, x2b) - torch.max(x1a, x1b)).clamp(0) *
             (torch.min(y2a, y2b) - torch.max(y1a, y1b)).clamp(0))
    union = wa * ha + wb * hb - inter + eps
    iou = inter / union
    if not ciou:
        return iou

    cw = torch.max(x2a, x2b) - torch.min(x1a, x1b)          # enclosing box
    ch = torch.max(y2a, y2b) - torch.min(y1a, y1b)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((x1b + x2b - x1a - x2a) ** 2 + (y1b + y2b - y1a - y2a) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(wb / hb) - torch.atan(wa / ha)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def make_anchors(feats, strides, offset=0.5):
    """One anchor POINT per grid cell, in grid units (cell centre = +0.5).

    Returns anchors [A,2] (x,y) and strides [A,1]. Order is level by level,
    row by row, which matches how we flatten the prediction maps.
    """
    pts, st = [], []
    for f, s in zip(feats, strides):
        h, w = f.shape[2:]
        sx = torch.arange(w, device=f.device, dtype=f.dtype) + offset
        sy = torch.arange(h, device=f.device, dtype=f.dtype) + offset
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        pts.append(torch.stack((xx, yy), -1).reshape(-1, 2))
        st.append(torch.full((h * w, 1), float(s), device=f.device, dtype=f.dtype))
    return torch.cat(pts), torch.cat(st)


def dist2bbox(ltrb, anchors):
    """Distances (left, top, right, bottom) from an anchor point -> xyxy box."""
    lt, rb = ltrb.chunk(2, -1)
    return torch.cat((anchors - lt, anchors + rb), -1)


def bbox2dist(anchors, bbox, max_dist):
    """Inverse of dist2bbox; clamped to the range the DFL head can express."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchors - x1y1, x2y2 - anchors), -1).clamp(0, max_dist)


# ------------------------------------------------------------------ NMS
def nms(boxes, scores, iou_thr):
    """Greedy Non-Maximum Suppression. boxes [N,4] xyxy, scores [N].
    Returns indices of the boxes to keep, highest score first."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]      # drop boxes overlapping the winner
    if not keep:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    return torch.stack(keep)


@torch.no_grad()
def non_max_suppression(pred, conf_thr=0.25, iou_thr=0.45, max_det=300, max_pre=3000):
    """pred: [B, A, 4+nc] (xyxy pixels + sigmoid class scores).
    Returns a list (one per image) of [n, 6] = x1, y1, x2, y2, conf, cls."""
    out = []
    for p in pred:
        boxes, conf_all = p[:, :4], p[:, 4:]
        conf, cls = conf_all.max(1)                     # best class per anchor
        m = conf > conf_thr
        boxes, conf, cls = boxes[m], conf[m], cls[m]
        if boxes.shape[0] > max_pre:
            top = conf.argsort(descending=True)[:max_pre]
            boxes, conf, cls = boxes[top], conf[top], cls[top]
        if boxes.shape[0] == 0:
            out.append(p.new_zeros((0, 6)))
            continue
        clsf = cls.to(boxes.dtype)
        # class-aware NMS trick: shift each class far apart so they never overlap
        keep = nms(boxes + clsf[:, None] * 4096.0, conf, iou_thr)[:max_det]
        out.append(torch.cat((boxes[keep], conf[keep, None], clsf[keep, None]), 1))
    return out
