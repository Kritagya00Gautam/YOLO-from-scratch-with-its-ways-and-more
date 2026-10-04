"""Datasets, letterboxing and augmentation (all hand-written, numpy + OpenCV)."""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox(img, size, color=114):
    """Resize keeping aspect ratio, pad to a size x size square.
    Returns image, scale ratio, left pad, top pad (needed to map boxes back)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, left, top


def random_affine(img, labels, scale=(0.6, 1.4), translate=0.1):
    """Random zoom + shift about the image centre. labels: [n,5] (cls,x1,y1,x2,y2) px.
    No rotation, and deliberately NO horizontal flip (mirrored text is not real text)."""
    size = img.shape[0]
    s = random.uniform(*scale)
    tx = (1 - s) * size / 2 + random.uniform(-translate, translate) * size
    ty = (1 - s) * size / 2 + random.uniform(-translate, translate) * size
    M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    img = cv2.warpAffine(img, M, (size, size), borderValue=(114, 114, 114))
    if len(labels):
        b = labels.copy()
        b[:, [1, 3]] = b[:, [1, 3]] * s + tx
        b[:, [2, 4]] = b[:, [2, 4]] * s + ty
        area0 = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])              # before clipping
        b[:, 1:] = b[:, 1:].clip(0, size)
        w, h = b[:, 3] - b[:, 1], b[:, 4] - b[:, 2]
        keep = (w > 2) & (h > 2) & (w * h / (area0 + 1e-9) > 0.25)     # drop mostly-cut boxes
        labels = b[keep]
    return img, labels


def augment_hsv(img, h=0.015, s=0.7, v=0.4):
    """Random hue / saturation / brightness jitter (lighting robustness)."""
    r = np.random.uniform(-1, 1, 3) * np.array([h, s, v]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _finish(img, labels, imgsz, augment):
    """img: BGR uint8 [imgsz,imgsz,3]; labels: [n,5] (cls,x1,y1,x2,y2) pixels.
    Returns (image tensor [3,H,W] in 0..1, targets [n,5] = cls,cx,cy,w,h normalised)."""
    if augment:
        img, labels = random_affine(img, labels)
        img = augment_hsv(img)
        if random.random() < 0.1:
            k = random.choice((3, 5, 7))
            img = cv2.GaussianBlur(img, (k, k), 0)
    n = len(labels)
    t = np.zeros((n, 5), np.float32)
    if n:
        t[:, 0] = labels[:, 0]
        t[:, 1] = (labels[:, 1] + labels[:, 3]) / 2 / imgsz
        t[:, 2] = (labels[:, 2] + labels[:, 4]) / 2 / imgsz
        t[:, 3] = (labels[:, 3] - labels[:, 1]) / imgsz
        t[:, 4] = (labels[:, 4] - labels[:, 2]) / imgsz
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))      # BGR->RGB, HWC->CHW
    return torch.from_numpy(x).float() / 255.0, torch.from_numpy(t)


class YoloDataset(Dataset):
    """Standard YOLO format: images/xxx.jpg + labels/xxx.txt, each label line is
    `class cx cy w h` with coordinates normalised to 0..1."""

    def __init__(self, img_dir, label_dir, imgsz=640, augment=False):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.files = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                            if p.lower().endswith(exts))
        if not self.files:
            raise FileNotFoundError(f"no images found in {img_dir}")
        self.label_dir, self.imgsz, self.augment = label_dir, imgsz, augment

    def __len__(self):
        return len(self.files)

    def _labels(self, img_path):
        f = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if not os.path.isfile(f):
            return np.zeros((0, 5), np.float32)
        a = np.loadtxt(f, ndmin=2, dtype=np.float32)
        return a.reshape(-1, 5) if a.size else np.zeros((0, 5), np.float32)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        if img is None:
            raise IOError(f"cannot read {self.files[i]}")
        h0, w0 = img.shape[:2]
        lab = self._labels(self.files[i])
        boxes = np.zeros((len(lab), 5), np.float32)                    # cls,x1,y1,x2,y2 (original px)
        boxes[:, 0] = lab[:, 0]
        boxes[:, 1] = (lab[:, 1] - lab[:, 3] / 2) * w0
        boxes[:, 2] = (lab[:, 2] - lab[:, 4] / 2) * h0
        boxes[:, 3] = (lab[:, 1] + lab[:, 3] / 2) * w0
        boxes[:, 4] = (lab[:, 2] + lab[:, 4] / 2) * h0
        img, r, left, top = letterbox(img, self.imgsz)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + left
        boxes[:, [2, 4]] = boxes[:, [2, 4]] * r + top
        return _finish(img, boxes, self.imgsz, self.augment)


class SyntheticDataset(Dataset):
    """Procedurally generated images with 3 classes (rectangle, ellipse, triangle)
    on a noisy background. Lets you verify the whole pipeline with zero data."""

    names = ["rect", "ellipse", "triangle"]

    def __init__(self, n=200, imgsz=320, seed=0, augment=False):
        self.n, self.imgsz, self.seed, self.augment = n, imgsz, seed, augment

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed + i)
        s = self.imgsz
        base = rng.randint(60, 200)
        img = np.clip(base + rng.randint(-15, 16, (s, s, 3)), 0, 255).astype(np.uint8)
        boxes = []
        for _ in range(rng.randint(1, 5)):
            for _try in range(20):                                     # place without overlap
                w, h = rng.randint(s // 8, s // 3), rng.randint(s // 8, s // 3)
                x1, y1 = rng.randint(0, s - w), rng.randint(0, s - h)
                if all(x1 >= b[3] or x1 + w <= b[1] or y1 >= b[4] or y1 + h <= b[2] for b in boxes):
                    break
            else:
                continue
            c = rng.randint(3)
            color = tuple(int(v) for v in rng.randint(0, 256, 3))
            if c == 0:
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
            elif c == 1:
                cv2.ellipse(img, (x1 + w // 2, y1 + h // 2), (w // 2, h // 2), 0, 0, 360, color, -1)
            else:
                pts = np.array([[x1 + w // 2, y1], [x1, y1 + h], [x1 + w, y1 + h]], np.int32)
                cv2.fillPoly(img, [pts], color)
            boxes.append([c, x1, y1, x1 + w, y1 + h])
        return _finish(img, np.array(boxes, np.float32).reshape(-1, 5), s, self.augment)


def collate(batch):
    """Stack images; concatenate targets, prepending the image index in the batch."""
    imgs, labels = zip(*batch)
    rows = [torch.cat((torch.full((len(l), 1), float(i)), l), 1) for i, l in enumerate(labels)]
    return torch.stack(imgs), torch.cat(rows, 0)
