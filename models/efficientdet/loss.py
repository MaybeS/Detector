from typing import Tuple, List

import torch
import torch.nn as nn


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(
        a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(
        a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) *
                         (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = .25, gamma: float = 2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], labels: List[torch.Tensor]):
        def decode(boxes: torch.Tensor) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            return width, height, boxes[:, 0] + .5 * width, boxes[:, 1] + .5 * height
        classifications, regressions, anchor = predictions
        classification_losses, regression_losses = [], []
        device = anchor.device

        anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = decode(anchor)

        for classification, regression, target in zip(classifications, regressions, labels):
            if target.size == 0:
                regression_losses.append(torch.tensor(0, dtype=torch.float32, device=device))
                classification_losses.append(torch.tensor(0, dtype=torch.float32, device=device))
                continue

            classification = torch.clamp(classification, 1e-4, 1. - 1e-4)

            # num_anchors x num_annotations
            IoU = calc_iou(anchor, target[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape, device=device) * -1
            targets[torch.lt(IoU_max, .4), :] = 0

            positive_indices = torch.ge(IoU_max, .5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = target[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape, device=device) * self.alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1. - targets) * torch.log(1. - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.), cls_loss, torch.zeros(cls_loss.shape, device=device))
            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths, gt_heights, gt_ctr_x, gt_ctr_y = decode(assigned_annotations)

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

                negative_indices = 1 + (~positive_indices)
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1. / 9.),
                    .5 * 9. * torch.pow(regression_diff, 2),
                    regression_diff - .5 / 9.
                )
                regression_losses.append(regression_loss.mean())

            else:
                regression_losses.append(torch.tensor(0, dtype=torch.float32, device=device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses).mean(dim=0, keepdim=True)
