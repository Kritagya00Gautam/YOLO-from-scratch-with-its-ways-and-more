"""Run this FIRST:  python tests/smoke_test.py   (a few minutes on CPU)
Checks shapes, NMS correctness, loss/backward, then overfits 8 synthetic images."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from yolo import YOLO, YOLOLoss, nms, non_max_suppression
from yolo.boxes import box_iou
from yolo.data import SyntheticDataset, collate
from yolo.metrics import map50
from yolo.boxes import xywh2xyxy

torch.manual_seed(0)
S, NC = 256, 3

# 1 ---- shapes ------------------------------------------------------------
m = YOLO(NC, "n")
x = torch.randn(2, 3, S, S)
raw = m(x)
assert [tuple(r.shape) for r in raw] == [(2, 64 + NC, S // 8, S // 8),
                                          (2, 64 + NC, S // 16, S // 16),
                                          (2, 64 + NC, S // 32, S // 32)], [r.shape for r in raw]
A = (S // 8) ** 2 + (S // 16) ** 2 + (S // 32) ** 2
p = m.eval().predict(x)
assert p.shape == (2, A, 4 + NC), p.shape
print(f"[1] shapes OK  ({sum(q.numel() for q in m.parameters()) / 1e6:.2f}M params, {A} anchors)")

# 2 ---- NMS vs brute-force reference ---------------------------------------
b = torch.rand(200, 2) * 200
b = torch.cat((b, b + torch.rand(200, 2) * 60 + 5), 1)
s = torch.rand(200)
keep = nms(b, s, 0.5).tolist()
ref, order = [], s.argsort(descending=True).tolist()
iou = box_iou(b, b)
sup = set()
for i in order:
    if i in sup:
        continue
    ref.append(i)
    sup.update(j for j in range(200) if j != i and iou[i, j] > 0.5)
assert keep == ref, "NMS differs from reference"
print(f"[2] NMS matches reference ({len(keep)} kept of 200)")

# 3 ---- loss + backward ----------------------------------------------------
ds = SyntheticDataset(8, S, seed=0, augment=False)
imgs, targets = collate([ds[i] for i in range(8)])
m = YOLO(NC, "n").train()
crit = YOLOLoss(m)
loss, items = crit(m(imgs), targets)
assert torch.isfinite(loss), loss
loss.backward()
assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters() if q.requires_grad)
print(f"[3] loss finite, grads finite  (box {items[0]:.3f} cls {items[1]:.3f} dfl {items[2]:.3f})")

# empty-target batch must not crash
l0, _ = crit(m(imgs), torch.zeros(0, 6))
assert torch.isfinite(l0)
print("[3b] empty-target batch OK")

# 4 ---- overfit 8 images ---------------------------------------------------
m = YOLO(NC, "n").train()
crit = YOLOLoss(m)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0)
first, t0 = None, time.time()
for step in range(1, 201):
    loss, items = crit(m(imgs), targets)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
    opt.step()
    first = first or loss.item()
    if step % 40 == 0:
        print(f"    step {step:3d}  loss {loss.item():.3f}  (box {items[0]:.3f} cls {items[1]:.3f} dfl {items[2]:.3f})  {time.time() - t0:.0f}s")
assert loss.item() < 0.7 * first, f"loss did not drop enough: {first:.2f} -> {loss.item():.2f}"
print(f"[4] overfit OK: loss {first:.2f} -> {loss.item():.2f}")

# 5 ---- decode + NMS + mAP on the training images ---------------------------
m.eval()
out = non_max_suppression(m.predict(imgs), conf_thr=0.01, iou_thr=0.6)
gts = []
for i in range(8):
    t = targets[targets[:, 0] == i]
    g = np.zeros((len(t), 5), np.float32)
    g[:, 0] = t[:, 1].numpy()
    g[:, 1:] = xywh2xyxy(t[:, 2:6] * S).numpy()
    gts.append(g)
mp, aps, _ = map50([o.numpy() for o in out], gts, NC)
print(f"[5] mAP50 on the 8 memorised images after 200 steps: {mp:.3f}\n"
      f"    (it should be clearly above 0 and keep rising with more steps; if it stays ~0 even after\n"
      f"     `python train.py --data synthetic --epochs 30`, suspect the decode/assigner coordinates)")
print("ALL CHECKS PASSED")
