"""Run a trained checkpoint on an image and draw the detections.
    python predict.py --weights runs/best.pt --source carton.jpg --out out.jpg
"""
import argparse

import cv2
import numpy as np
import torch

from yolo import YOLO, non_max_suppression
from yolo.data import letterbox


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = YOLO(ck["nc"], ck["scale"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["names"], ck["imgsz"]


@torch.no_grad()
def detect(model, bgr, imgsz, conf=0.25, iou=0.45, device="cpu"):
    """Returns an [n,6] array (x1,y1,x2,y2,conf,cls) in ORIGINAL image pixels."""
    img, r, left, top = letterbox(bgr, imgsz)
    x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float() / 255
    out = non_max_suppression(model.predict(x[None].to(device)), conf, iou)[0].cpu().numpy()
    out[:, [0, 2]] = ((out[:, [0, 2]] - left) / r).clip(0, bgr.shape[1])   # undo letterbox
    out[:, [1, 3]] = ((out[:, [1, 3]] - top) / r).clip(0, bgr.shape[0])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="out.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, names, imgsz = load_model(a.weights, dev)
    im = cv2.imread(a.source)
    for x1, y1, x2, y2, cf, c in detect(model, im, imgsz, a.conf, a.iou, dev):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(im, p1, p2, (0, 200, 0), 2)
        cv2.putText(im, f"{names[int(c)]} {cf:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        print(f"{names[int(c)]:>12s} {cf:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    cv2.imwrite(a.out, im)
