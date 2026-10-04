"""YOLOv8-style detector: backbone -> PAN-FPN neck -> decoupled anchor-free head."""
import math
import torch
import torch.nn as nn

from .blocks import Conv, C2f, SPPF
from .boxes import make_anchors, dist2bbox

# scale: (depth multiplier, width multiplier, max channels)
SCALES = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


def make_divisible(x, d=8):
    return int(math.ceil(x / d) * d)


def flatten_preds(raw, reg_max, nc):
    """List of 3 maps [B, 4*R+nc, H, W] -> dist [B,A,4R], scores(logits) [B,A,nc]."""
    B = raw[0].shape[0]
    x = torch.cat([r.reshape(B, 4 * reg_max + nc, -1) for r in raw], 2)
    dist, scores = x.split((4 * reg_max, nc), 1)
    return dist.permute(0, 2, 1).contiguous(), scores.permute(0, 2, 1).contiguous()


def dfl_decode(dist, proj):
    """Distribution Focal Loss decoding.

    Each box side is predicted as a distribution over R bins (0..R-1 grid cells).
    The side length is the EXPECTATION of that distribution.
    dist [B,A,4R] -> [B,A,4] distances (l,t,r,b) in grid units."""
    B, A, _ = dist.shape
    return (dist.view(B, A, 4, -1).softmax(3) * proj).sum(3)


class Detect(nn.Module):
    """Decoupled head: separate conv branches for box regression and class."""

    def __init__(self, nc, ch, strides=(8, 16, 32), reg_max=16):
        super().__init__()
        self.nc, self.reg_max, self.strides = nc, reg_max, strides
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.box_head = nn.ModuleList(
            nn.Sequential(Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for c in ch)
        self.cls_head = nn.ModuleList(
            nn.Sequential(Conv(c, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1))
            for c in ch)
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32), persistent=False)
        self._init_bias()

    def _init_bias(self):
        for b, c, s in zip(self.box_head, self.cls_head, self.strides):
            b[-1].bias.data[:] = 1.0
            # start with a very low class probability (~5 objects per 640px image)
            c[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def forward(self, feats):
        return [torch.cat((self.box_head[i](f), self.cls_head[i](f)), 1)
                for i, f in enumerate(feats)]


class YOLO(nn.Module):
    def __init__(self, nc=80, scale="n"):
        super().__init__()
        d, w, mc = SCALES[scale]
        c1, c2, c3, c4, c5 = [make_divisible(min(c, mc) * w) for c in (64, 128, 256, 512, 1024)]
        n = lambda k: max(round(k * d), 1)

        self.nc, self.scale, self.strides, self.reg_max = nc, scale, (8, 16, 32), 16

        # ---- backbone: progressively halves resolution (strides 2,4,8,16,32)
        self.stem = Conv(3, c1, 3, 2)
        self.stage2 = nn.Sequential(Conv(c1, c2, 3, 2), C2f(c2, c2, n(3), True))
        self.stage3 = nn.Sequential(Conv(c2, c3, 3, 2), C2f(c3, c3, n(6), True))   # -> P3
        self.stage4 = nn.Sequential(Conv(c3, c4, 3, 2), C2f(c4, c4, n(6), True))   # -> P4
        self.stage5 = nn.Sequential(Conv(c4, c5, 3, 2), C2f(c5, c5, n(3), True),
                                    SPPF(c5, c5, 5))                               # -> P5

        # ---- neck: top-down (FPN) then bottom-up (PAN)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.td4 = C2f(c5 + c4, c4, n(3), False)
        self.td3 = C2f(c4 + c3, c3, n(3), False)       # output for small objects
        self.down3 = Conv(c3, c3, 3, 2)
        self.bu4 = C2f(c3 + c4, c4, n(3), False)       # output for medium objects
        self.down4 = Conv(c4, c4, 3, 2)
        self.bu5 = C2f(c4 + c5, c5, n(3), False)       # output for large objects

        self.detect = Detect(nc, (c3, c4, c5), self.strides, self.reg_max)

    def forward(self, x):
        """Training forward: returns raw maps, one per scale, [B, 4R+nc, H, W]."""
        x = self.stage2(self.stem(x))
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)

        t4 = self.td4(torch.cat((self.up(p5), p4), 1))
        o3 = self.td3(torch.cat((self.up(t4), p3), 1))
        o4 = self.bu4(torch.cat((self.down3(o3), t4), 1))
        o5 = self.bu5(torch.cat((self.down4(o4), p5), 1))
        return self.detect([o3, o4, o5])

    @torch.no_grad()
    def predict(self, x):
        """Inference: [B, A, 4+nc] = xyxy boxes in input pixels + class probabilities.
        (NMS is applied separately, see boxes.non_max_suppression.)"""
        raw = self.forward(x)
        dist, scores = flatten_preds(raw, self.reg_max, self.nc)
        anchors, st = make_anchors(raw, self.strides)
        boxes = dist2bbox(dfl_decode(dist, self.detect.proj), anchors) * st
        return torch.cat((boxes, scores.sigmoid()), -1)
