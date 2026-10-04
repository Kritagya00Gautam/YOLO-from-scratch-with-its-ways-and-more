"""Train the from-scratch YOLO.

Quick pipeline check (no data needed):
    python train.py --data synthetic --epochs 30

Your own data (YOLO txt format: class cx cy w h, normalised):
    python train.py --train-images d/images/train --train-labels d/labels/train \
                    --val-images d/images/val --val-labels d/labels/val \
                    --names brand_block,batch_line,exp_line --imgsz 640 --epochs 100
"""
import argparse
import copy
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolo import YOLO, YOLOLoss, non_max_suppression
from yolo.boxes import xywh2xyxy
from yolo.data import YoloDataset, SyntheticDataset, collate
from yolo.metrics import map50


class EMA:
    """Exponential moving average of the weights. Evaluating the averaged model is
    almost always better than the raw, noisy, latest weights."""

    def __init__(self, model, decay=0.999, tau=500):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))      # ramp up early
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


@torch.no_grad()
def validate(model, loader, device, imgsz, nc, conf=0.001, iou=0.6):
    model.eval()
    preds, gts = [], []
    for imgs, targets in loader:
        out = non_max_suppression(model.predict(imgs.to(device)), conf, iou)
        for i, o in enumerate(out):
            preds.append(o.cpu().numpy())
            t = targets[targets[:, 0] == i]
            g = np.zeros((len(t), 5), np.float32)
            g[:, 0] = t[:, 1].numpy()
            g[:, 1:] = xywh2xyxy(t[:, 2:6] * imgsz).numpy()
            gts.append(g)
    return map50(preds, gts, nc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="custom", choices=["custom", "synthetic"])
    ap.add_argument("--train-images"); ap.add_argument("--train-labels")
    ap.add_argument("--val-images"); ap.add_argument("--val-labels")
    ap.add_argument("--names", default="", help="comma separated class names")
    ap.add_argument("--scale", default="n", choices=list("nsmlx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--out", default="runs/last.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data == "synthetic":
        args.imgsz = min(args.imgsz, 320)
        train_ds = SyntheticDataset(400, args.imgsz, seed=0, augment=True)
        val_ds = SyntheticDataset(60, args.imgsz, seed=100_000, augment=False)
        names = SyntheticDataset.names
    else:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        assert names, "--names is required for custom data"
        train_ds = YoloDataset(args.train_images, args.train_labels, args.imgsz, augment=True)
        val_ds = YoloDataset(args.val_images, args.val_labels, args.imgsz, augment=False)
    nc = len(names)
    assert args.imgsz % 32 == 0, "imgsz must be divisible by 32"

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = YOLO(nc, args.scale).to(device)
    print(f"YOLO-{args.scale}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{nc} classes, device={device}")
    criterion = YOLOLoss(model)
    ema = EMA(model)

    # no weight decay on BN / biases
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 5e-4},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    total_it = args.epochs * len(train_dl)
    warm = min(300, total_it // 5)
    it, best = 0, -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, sums = time.time(), torch.zeros(3)
        for imgs, targets in train_dl:
            # warmup then cosine decay to 1% of the base lr
            if it < warm:
                lr = args.lr * (0.1 + 0.9 * it / warm)
            else:
                p = (it - warm) / max(1, total_it - warm)
                lr = args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))
            for g in opt.param_groups:
                g["lr"] = lr

            raw = model(imgs.to(device))
            loss, items = criterion(raw, targets.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ema.update(model)
            sums += items.cpu()
            it += 1

        sums /= len(train_dl)
        msg = (f"epoch {epoch + 1:3d}/{args.epochs}  box {sums[0]:.3f}  cls {sums[1]:.3f}  "
               f"dfl {sums[2]:.3f}  lr {lr:.5f}  {time.time() - t0:.0f}s")
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            m, aps, recs = validate(ema.model, val_dl, device, args.imgsz, nc)
            msg += f"  | val mAP50 {m:.3f}"
            if m > best:
                best = m
                torch.save({"model": ema.model.state_dict(), "nc": nc, "scale": args.scale,
                            "names": names, "imgsz": args.imgsz}, args.out.replace("last", "best"))
        print(msg)
        torch.save({"model": ema.model.state_dict(), "nc": nc, "scale": args.scale,
                    "names": names, "imgsz": args.imgsz}, args.out)


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    main()
