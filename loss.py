import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=1, iou_type='ciou', lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.iou_type = iou_type
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        """
        preds: (cls_pred, obj_pred, box_pred)
            cls_pred: [B, A, C]
            obj_pred: [B, A, 1]
            box_pred: [B, A, 4] -> [cx, cy, w, h]
        targets: dict
            cls: [B, A, C]
            obj: [B, A, 1]
            box: [B, A, 4]
        """
        cls_pred, obj_pred, box_pred = preds
        cls_target = targets['cls']
        obj_target = targets['obj']
        box_target = targets['box']

        cls_loss = self.bce_cls(cls_pred, cls_target)
        cls_loss = (cls_loss * obj_target).sum() / obj_target.sum().clamp(min=1.0)

        obj_loss = self.bce_obj(obj_pred, obj_target)
        obj_loss = obj_loss.mean()

        box_loss = self.iou_loss(box_pred, box_target, obj_target)

        total_loss = self.lambda_box * box_loss + self.lambda_obj * obj_loss + self.lambda_cls * cls_loss

        return total_loss, box_loss, obj_loss, cls_loss

    def iou_loss(self, box1, box2, obj_mask):
        """
        box1, box2: [B, A, 4] -> [cx, cy, w, h]
        obj_mask: [B, A, 1]
        """
        box1 = box1[obj_mask.bool()]
        box2 = box2[obj_mask.bool()]
        if box1.numel() == 0:
            return torch.tensor(0.0, device=box1.device)

        box1_xy = self.cxcywh_to_xyxy(box1)
        box2_xy = self.cxcywh_to_xyxy(box2)

        return (1.0 - self.bbox_ciou(box1_xy, box2_xy)).mean()

    def cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def bbox_ciou(self, box1, box2, eps=1e-7):
        """
        box1, box2: [N, 4] -> [x1, y1, x2, y2]
        return: CIoU
        """
        inter = (torch.min(box1[:, 2:], box2[:, 2:]) - torch.max(box1[:, :2], box2[:, :2])).clamp(0).prod(1)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - inter + eps
        iou = inter / union

        c1 = (box1[:, 0] + box1[:, 2]) / 2
        c2 = (box2[:, 0] + box2[:, 2]) / 2
        d1 = (box1[:, 1] + box1[:, 3]) / 2
        d2 = (box2[:, 1] + box2[:, 3]) / 2
        center_dist = (c1 - c2) ** 2 + (d1 - d2) ** 2

        enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
        enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
        enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
        enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
        enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

        w1 = box1[:, 2] - box1[:, 0]
        h1 = box1[:, 3] - box1[:, 1]
        w2 = box2[:, 2] - box2[:, 0]
        h2 = box2[:, 3] - box2[:, 1]
        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w1 / h1 + eps) - torch.atan(w2 / h2 + eps), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - center_dist / enclose_diagonal - alpha * v
        return ciou
