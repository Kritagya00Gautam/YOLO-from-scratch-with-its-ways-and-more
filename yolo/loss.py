"""Training loss: Task-Aligned label assignment + CIoU + DFL + BCE."""
import torch
import torch.nn as nn

from .boxes import bbox_iou, bbox2dist, dist2bbox, make_anchors, xywh2xyxy
from .model import flatten_preds, dfl_decode


class TaskAlignedAssigner(nn.Module):
    """Decides which of the A anchor points is responsible for which ground truth.

    For every GT box:
      1. candidates  = anchors whose centre lies inside the GT box
      2. alignment   = (predicted score of the GT's class)^alpha * IoU(pred, GT)^beta
      3. positives   = top-k candidates by alignment
      4. an anchor claimed by several GTs goes to the one with the highest IoU
    The soft target score of a positive is its alignment, normalised so the best
    anchor of each GT gets a target equal to that GT's best IoU.
    """

    def __init__(self, topk=10, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk, self.alpha, self.beta, self.eps = topk, alpha, beta, eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pd_scores  [B,A,nc]  sigmoid class probabilities
        pd_bboxes  [B,A,4]   predicted xyxy (pixels)
        anc_points [A,2]     anchor centres (pixels)
        gt_labels  [B,M,1]   gt_bboxes [B,M,4] xyxy pixels   mask_gt [B,M,1] (1 = real box)
        returns: target_bboxes [B,A,4], target_scores [B,A,nc], fg_mask [B,A] (bool)
        """
        B, A, nc = pd_scores.shape
        M = gt_bboxes.shape[1]

        # 1) anchor centre inside GT
        gt = gt_bboxes.unsqueeze(2)                                   # [B,M,1,4]
        a = anc_points[None, None]                                    # [1,1,A,2]
        deltas = torch.cat((a - gt[..., :2], gt[..., 2:] - a), -1)    # [B,M,A,4]
        mask_in = (deltas.amin(3) > self.eps).float() * mask_gt       # [B,M,A]

        # 2) alignment metric
        idx_b = torch.arange(B, device=pd_scores.device).view(-1, 1).expand(-1, M)
        idx_c = gt_labels.squeeze(-1).long()
        cls_scores = pd_scores[idx_b, :, idx_c]                       # [B,M,A]
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1)).clamp(0)
        overlaps = overlaps * mask_in                                 # [B,M,A]
        align = cls_scores.pow(self.alpha) * overlaps.pow(self.beta)  # [B,M,A]

        # 3) top-k anchors per GT
        k = min(self.topk, A)
        vals, idx = align.topk(k, dim=-1)                             # [B,M,k]
        mask_topk = torch.zeros_like(align)
        mask_topk.scatter_(-1, idx, (vals > self.eps).to(align.dtype))
        mask_pos = mask_topk * mask_in                                # [B,M,A]

        # 4) resolve anchors claimed by more than one GT
        fg = mask_pos.sum(1)                                          # [B,A]
        if fg.max() > 1:
            multi = (fg.unsqueeze(1) > 1)                             # [B,1,A]
            best = (overlaps * mask_pos).argmax(1)                    # [B,A]
            is_best = torch.zeros_like(mask_pos).scatter_(1, best.unsqueeze(1), 1.0)
            mask_pos = torch.where(multi, is_best, mask_pos)
            fg = mask_pos.sum(1)
        gt_idx = mask_pos.argmax(1)                                   # [B,A]

        # ---- build targets
        flat = gt_idx + torch.arange(B, device=gt_idx.device).unsqueeze(-1) * M
        t_labels = gt_labels.reshape(-1).long()[flat]                 # [B,A]
        t_boxes = gt_bboxes.reshape(-1, 4)[flat]                      # [B,A,4]
        t_scores = torch.zeros(B, A, nc, device=pd_scores.device, dtype=pd_scores.dtype)
        t_scores.scatter_(2, t_labels.unsqueeze(-1), 1.0)
        t_scores = t_scores * (fg > 0).unsqueeze(-1)

        # soft targets: scale the one-hot by the normalised alignment metric
        align = align * mask_pos
        pos_align = align.amax(-1, keepdim=True)                      # [B,M,1]
        pos_iou = (overlaps * mask_pos).amax(-1, keepdim=True)        # [B,M,1]
        norm = (align * pos_iou / (pos_align + self.eps)).amax(1).unsqueeze(-1)  # [B,A,1]
        t_scores = t_scores * norm
        return t_boxes, t_scores, fg > 0


def dfl_loss(pred_dist, target):
    """Cross-entropy on the two integer bins surrounding the continuous target,
    weighted by proximity. pred_dist [n,4,R] logits, target [n,4] in [0,R-1)."""
    tl = target.long()
    tr = tl + 1
    wl = tr.to(target.dtype) - target
    wr = 1.0 - wl
    logp = torch.log_softmax(pred_dist, -1)
    ll = -logp.gather(-1, tl.unsqueeze(-1)).squeeze(-1)
    lr = -logp.gather(-1, tr.unsqueeze(-1)).squeeze(-1)
    return (ll * wl + lr * wr).mean(-1, keepdim=True)                 # [n,1]


class YOLOLoss:
    def __init__(self, model, w_box=7.5, w_cls=0.5, w_dfl=1.5):
        self.model = model
        self.nc, self.R, self.strides = model.nc, model.reg_max, model.strides
        self.w_box, self.w_cls, self.w_dfl = w_box, w_cls, w_dfl
        self.assigner = TaskAlignedAssigner(topk=10)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _prep(targets, B, scale):
        """[N,6] (img_idx, cls, cx, cy, w, h normalised) -> padded [B,M,5] (cls, xyxy px)."""
        if targets.numel() == 0:
            counts = torch.zeros(B, dtype=torch.long)
        else:
            counts = torch.bincount(targets[:, 0].long().cpu(), minlength=B)
        M = max(int(counts.max()), 1)
        out = torch.zeros(B, M, 5, device=targets.device)
        for j in range(B):
            t = targets[targets[:, 0] == j][:, 1:]
            out[j, : len(t)] = t
        out[..., 1:] = xywh2xyxy(out[..., 1:] * scale)
        return out

    def __call__(self, raw, targets):
        """raw: list of 3 head maps. targets: [N,6]. Returns (total_loss, [box, cls, dfl])."""
        B, dev = raw[0].shape[0], raw[0].device
        dist, scores = flatten_preds(raw, self.R, self.nc)           # [B,A,4R], [B,A,nc]
        anchors, st = make_anchors(raw, self.strides)                 # [A,2], [A,1]

        h, w = raw[0].shape[2] * self.strides[0], raw[0].shape[3] * self.strides[0]
        scale = torch.tensor([w, h, w, h], device=dev, dtype=torch.float32)
        gt = self._prep(targets.to(dev).float(), B, scale)
        gt_labels, gt_bboxes = gt[..., :1], gt[..., 1:]
        mask_gt = (gt_bboxes.sum(2, keepdim=True) > 0).float()

        proj = self.model.detect.proj
        pred_boxes = dist2bbox(dfl_decode(dist, proj), anchors)       # grid units

        t_boxes, t_scores, fg = self.assigner(
            scores.detach().sigmoid(), pred_boxes.detach() * st, anchors * st,
            gt_labels, gt_bboxes, mask_gt)
        t_boxes = t_boxes / st                                        # back to grid units
        t_sum = t_scores.sum().clamp(min=1.0)

        loss_cls = self.bce(scores, t_scores).sum() / t_sum
        loss_box = torch.zeros((), device=dev)
        loss_dfl = torch.zeros((), device=dev)
        if fg.any():
            weight = t_scores.sum(-1)[fg].unsqueeze(-1)               # [n,1]
            ciou = bbox_iou(pred_boxes[fg], t_boxes[fg], ciou=True)   # [n]
            loss_box = ((1.0 - ciou).unsqueeze(-1) * weight).sum() / t_sum
            t_ltrb = bbox2dist(anchors, t_boxes, self.R - 1 - 0.01)   # [B,A,4]
            loss_dfl = (dfl_loss(dist[fg].view(-1, 4, self.R), t_ltrb[fg]) * weight).sum() / t_sum

        total = self.w_box * loss_box + self.w_cls * loss_cls + self.w_dfl * loss_dfl
        return total, torch.stack((loss_box, loss_cls, loss_dfl)).detach()
