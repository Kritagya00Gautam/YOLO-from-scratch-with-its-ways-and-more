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



Simply, to understand all the necessary aspects of the YOLO model built:

1.Convolution
In computer vision, convolution is the core concept that is indeed embedded in computer vision models to get the meaningful images from the initial image. At its heart, convolution takes a small matrix of numbers called a kernel (or filter) and slides it across the image matrix, performing element-wise multiplication and adding up the results to create a new, transformed matrix called a feature map.


$$\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1$$

2.Max pooling
In computer vision, max pooling is a downstream process used to reduce the spatial size of the of the feature map while protecting its useful portions, it is used to reduce the computation complexity and speeds us training and inference.

3.Bottleneck
In computer vision, bottleneck is a design pattern that is implemented to reduce computation. Simply, it is a sequence of convolution layers .

4.Batch Normalization
In computer vision, BA(Batch Normalization) crucial layer typically inserted right after a convolution layer and before an activation function (often written in code blocks as Conv2d -> BatchNorm2d -> SiLU/ReLU).During training, as the weights in earlier layers change, the distribution of inputs to later layers constantly shifts—a phenomenon known as internal covariate shift. This forces later layers to constantly adapt, slowing down training.Batch Normalization solves this by normalizing the activations of a layer for each training mini-batch so they have a mean of roughly 0 and a variance of roughly 1.

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

5.SILU
In deep learning and modern object detection models like YOLO (such as YOLOv5, YOLOv7, and YOLOv8), SiLU stands for Sigmoid Linear Unit (also widely known as the Swish activation function, originally discovered by researchers at Google).
The SiLU function multiplies the input $x$ by the sigmoid function of $x$:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Older YOLO models (like YOLOv2 and YOLOv3) used LeakyReLU. However, modern YOLO architectures switched to SiLU because empirical tests showed it consistently yields higher accuracy. By avoiding the "dying ReLU" problem (where neurons get stuck outputting zero and stop learning), SiLU helps the network capture finer spatial and semantic details required for precise object detection.

6.Cross Stage Partial Bottleneck with 2 convolutions.
In modern YOLO architectures (specifically starting with YOLOv8), C2f stands for Cross Stage Partial Bottleneck with 2 convolutions.It is the core building block used throughout the backbone and neck of YOLOv8, replacing the older C3 module used in YOLOv5. The C2f block is designed to capture richer gradient flow and multi-scale features while keeping the model lightweight and fast. It combines ideas from CSPNet (Cross Stage Partial Network) and residual bottlenecks.

7.Spatial Pyramid Pooling - Fast.
In modern YOLO architectures (such as YOLOv5, YOLOv8, and YOLOv11), SPPF stands for Spatial Pyramid Pooling - Fast. It is a crucial module placed right at the bridge between the backbone and the neck of the network. Its primary job is to widen the model's receptive field and capture multi-scale features so the network can successfully detect objects of vastly different sizes (from tiny distant objects to large close-up objects).Objects in images appear at many different scales. A standard convolutional layer only looks at a fixed local neighborhood. To understand an object in context, the network needs to see both local details (fine textures) and global context (surrounding background).The original SPP module solved this by passing feature maps through multiple parallel max-pooling branches with different kernel sizes (e.g., $5 \times 5$, $9 \times 9$, and $13 \times 13$) and concatenating them. However, running these separate parallel branches was computationally heavy and slow. SPPF achieves the exact same mathematical receptive field as standard SPP, but with a fraction of the computational overhead.

8.The Backbone
The first part of the network, typically built using C2f blocks and SPPF (Spatial Pyramid Pooling - Fast). It takes the raw input image and progressively downsamples it using convolutions, batch normalization, and SiLU activations. Its job is to extract raw visual features—starting with low-level details (edges, lines, textures) in early layers and building up to deep semantic features (shapes, parts of objects) in deeper layers.


9.FPN (Feature Pyramid Network) combined with PAN (Path Aggregation Network).
Objects in images come in all sizes. Deep layers have great semantic context but poor spatial resolution (bad for tiny objects), while shallow layers have great spatial resolution but weak semantic context.FPN creates a top-down pathway to pass rich semantic information down to smaller feature maps. PAN adds a bottom-up pathway to pass accurate spatial information up to deeper feature maps. This ensures the network has access to multi-scale features simultaneously.







