# YOLO from scratch (YOLOv8-style, PyTorch)

Every part of the detector is written by hand. PyTorch is used only for tensors,
`nn.Conv2d`/`BatchNorm`, autograd and the optimiser. No `ultralytics`, no `torchvision.ops`.

## Layout

| File | What it implements |
|---|---|
| `yolo/blocks.py` | `Conv` (conv+BN+SiLU), `Bottleneck`, `C2f`, `SPPF` |
| `yolo/model.py` | Backbone, PAN-FPN neck, decoupled anchor-free `Detect` head, DFL decoding |
| `yolo/boxes.py` | xywh/xyxy, IoU, **CIoU**, anchor grid, ltrb<->box, **NMS** |
| `yolo/loss.py` | **Task-Aligned Assigner**, CIoU box loss, **DFL** loss, BCE class loss |
| `yolo/data.py` | Letterbox, random affine, HSV jitter, YOLO-txt dataset, synthetic dataset |
| `yolo/metrics.py` | mAP@0.5 |
| `train.py` | AdamW + warmup + cosine LR, EMA weights, validation, checkpoints |
| `predict.py` | Inference on an image, undoing the letterbox |
| `export_onnx.py` | ONNX export for ONNX Runtime Mobile |
| `tests/smoke_test.py` | Shape checks, NMS vs reference, loss/backward, overfit test |

## Data flow (matches the explanation)

```
image 640x640 --backbone--> P3 (80x80), P4 (40x40), P5 (20x20)
              --neck------> fused maps at the same 3 scales
              --head-----> per cell: 4*16 box-distribution logits + nc class logits
8400 anchors --> DFL decode (softmax expectation) --> ltrb distances --> boxes
--> confidence filter --> NMS --> final detections
```

Training adds: Task-Aligned Assigner (which anchor owns which GT) then
`7.5*CIoU + 0.5*BCE + 1.5*DFL`.

## Run

```bash
pip install -r requirements.txt
python tests/smoke_test.py                      # run this first
python train.py --data synthetic --epochs 30    # end-to-end check, no data needed
```

Your own data (YOLO txt format, `class cx cy w h` normalised):

```bash
python train.py --train-images data/images/train --train-labels data/labels/train \
                --val-images data/images/val --val-labels data/labels/val \
                --names brand_block,batch_line,exp_line --imgsz 640 --epochs 100
python predict.py --weights runs/best.pt --source carton.jpg
python export_onnx.py --weights runs/best.pt --out yolo.onnx
```

## Notes for the Ausadhi carton task

* Augmentation has **no horizontal flip** (mirrored text does not exist).
* Split train/val **by carton**, not by image, so val mAP is honest.
* Look at **per-class recall** (`batch_line`, `exp_line`), not only mAP50.
* Use `--scale n` and `--imgsz 640` for the phone model.
* From scratch (random init) needs far more data than fine-tuning a pretrained model;
  with only a few hundred images expect weak results.
