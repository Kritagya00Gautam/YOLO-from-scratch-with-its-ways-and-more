"""mAP@0.5 in plain numpy."""
import numpy as np


def iou_np(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def map50(preds, gts, nc):
    """preds: list per image of [n,6] (x1,y1,x2,y2,conf,cls); gts: list per image of
    [m,5] (cls,x1,y1,x2,y2). Returns (mAP50, per-class AP list, per-class recall list)."""
    aps, recalls = [], []
    for c in range(nc):
        gt_c = {i: g[g[:, 0] == c, 1:] for i, g in enumerate(gts)}
        npos = sum(len(v) for v in gt_c.values())
        if npos == 0:
            continue
        dets = [(i, p[:4], p[4]) for i, pi in enumerate(preds) for p in pi if int(p[5]) == c]
        dets.sort(key=lambda d: -d[2])
        used = {i: np.zeros(len(v), bool) for i, v in gt_c.items()}
        tp = np.zeros(len(dets))
        for k, (i, box, _) in enumerate(dets):
            g = gt_c[i]
            if len(g) == 0:
                continue
            ious = iou_np(box[None], g)[0]
            j = int(ious.argmax())
            if ious[j] >= 0.5 and not used[i][j]:
                used[i][j] = True
                tp[k] = 1
        tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
        rec = tpc / npos
        prec = tpc / np.maximum(tpc + fpc, 1e-9)
        ap = 0.0
        if len(rec):
            env = np.flip(np.maximum.accumulate(np.flip(prec)))       # precision envelope
            x = np.linspace(0, 1, 101)                                # 101 recall thresholds (COCO style)
            idx = np.searchsorted(rec, x, side="left")
            ok = idx < len(rec)                                       # recall levels never reached count as 0
            q = np.zeros_like(x)
            q[ok] = env[idx[ok]]
            ap = q.mean()
        aps.append(float(ap))
        recalls.append(float(rec[-1]) if len(rec) else 0.0)
    return (float(np.mean(aps)) if aps else 0.0), aps, recalls
