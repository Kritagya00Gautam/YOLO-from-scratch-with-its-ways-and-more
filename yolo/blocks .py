"""Building blocks of a YOLOv8-style network, written by hand."""
import torch
import torch.nn as nn


class Conv(nn.Module):
    """Conv2d -> BatchNorm -> SiLU. The basic unit used everywhere."""

    def __init__(self, c_in, c_out, k=1, stride=1, pad=None):
        super().__init__()
        pad = k // 2 if pad is None else pad          # 'same' padding
        self.conv = nn.Conv2d(c_in, c_out, k, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two 3x3 convs with an optional residual (skip) connection."""

    def __init__(self, c_in, c_out, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c_in, c_out, 3)
        self.cv2 = Conv(c_out, c_out, 3)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Cross-Stage-Partial block with 2 convs and 'fast' feature reuse.

    1x1 conv -> split channels in two halves -> run the second half through n
    bottlenecks, keeping EVERY intermediate output -> concat all -> 1x1 conv.
    The concat of all intermediates gives a rich gradient flow at low cost.
    """

    def __init__(self, c_in, c_out, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c_out * e)
        self.cv1 = Conv(c_in, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c_out, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast. Three chained 5x5 max-pools give the
    same receptive fields as parallel 5/9/13 pools, but cheaper."""

    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_ = c_in // 2
        self.cv1 = Conv(c_in, c_, 1)
        self.cv2 = Conv(c_ * 4, c_out, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))
