import torch
import torch.nn as nn

from lib.evaluate import compute_iou_


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = .25, gamma: float = 2.,
                 device=None, **kwargs):
        super(FocalLoss, self).__init__()
        self.device = device

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classifications, regressions, anchors, annotations):
        def extract(box):
            w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
            return w, h, box[:, 0] + .5 * w, box[:, 1] + .5 * h

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = extract(anchors[0, :, :])

        for classification, regression, bbox_annotation in zip(classifications, regressions, annotations):
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(self.device))
                classification_losses.append(torch.tensor(0).float().to(self.device))
                continue

            classification = torch.clamp(classification, 1e-4, 1. - 1e-4)
            IoU = compute_iou_(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = (torch.ones(classification.shape) * -1).to(self.device)

            targets[torch.lt(IoU_max, .4), :] = 0

            positive_indices = torch.ge(IoU_max, .5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).to(self.device) * self.alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1. - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.), cls_loss, torch.zeros(cls_loss.shape).to(self.device))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.))

            # compute the loss for regression
            if positive_indices.sum().item() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths, gt_heights, gt_ctr_x, gt_ctr_y = extract(assigned_annotations)

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[.1, .1, .2, .2]]).to(self.device)

                negative_indices = 1 - positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1. / 9.),
                    .5 * 9. * torch.pow(regression_diff, 2),
                    regression_diff - .5 / 9.
                )
                regression_losses.append(regression_loss.mean())

            else:
                regression_losses.append(torch.tensor(0).float().to(self.device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)
