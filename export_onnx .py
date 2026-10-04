"""Export to ONNX for ONNX Runtime Mobile (no NMS inside; do NMS in app code).
Output shape: [1, 4+nc, A]  (xyxy in input pixels, then class probabilities).
    python export_onnx.py --weights runs/best.pt --out yolo.onnx
"""
import argparse

import torch
import torch.nn as nn

from predict import load_model


class Wrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.predict(x).permute(0, 2, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="yolo.onnx")
    a = ap.parse_args()
    model, names, imgsz = load_model(a.weights)
    torch.onnx.export(Wrapper(model), torch.zeros(1, 3, imgsz, imgsz), a.out,
                      input_names=["images"], output_names=["output0"], opset_version=12)
    print("exported", a.out, "classes:", names)
