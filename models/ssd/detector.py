from typing import List

import torch
from torch.autograd import Function

from lib.box import decode, nms


class Detector(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    @classmethod
    def init(cls, num_classes: int, batch_size, bg_label: int = 0, top_k: int = 200,
             conf_thresh: float = .01, nms: bool = True, nms_thresh: float = .45,
             variance: List[int] = None):
        cls.num_classes = num_classes
        cls.batch_size = batch_size
        cls.background_label = bg_label
        cls.top_k = top_k
        cls.nms = nms
        cls.nms_thresh = nms_thresh
        cls.conf_thresh = conf_thresh
        cls.variance = variance or [.1, .2]

    @classmethod
    def forward(cls, locations: torch.Tensor,
                confidences: torch.Tensor,
                prior_boxes: torch.Tensor):
        """
        Args:
            locations: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors*4]
            confidences: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors, num_classes]
            prior_boxes: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1, num_priors,4]
        """
        num_priors = prior_boxes.size(0)

        output = torch.zeros(cls.batch_size, cls.num_classes, cls.top_k, 5) if cls.nms else None
        confidences = confidences.view(cls.batch_size, num_priors, cls.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for batch_index, (location, confidence) in enumerate(zip(locations, confidences)):
            decoded_boxes = decode(location, prior_boxes, cls.variance)
            conf_scores = confidence.clone()

            if cls.nms:
                for class_index in range(1, cls.num_classes):
                    # idx of highest scoring and non-overlapping boxes per class
                    conf_mask = conf_scores[class_index].gt(cls.conf_thresh)
                    scores = conf_scores[class_index][conf_mask]

                    if scores.size(0) == 0:
                        continue

                    loc_mask = conf_mask.unsqueeze(1).expand_as(decoded_boxes)
                    boxes = decoded_boxes[loc_mask].view(-1, 4)

                    ids, count = nms(boxes, scores, cls.nms_thresh, cls.top_k)
                    output[batch_index, class_index, :count] = torch.cat((
                        scores[ids[:count]].unsqueeze(1),
                        boxes[ids[:count]]
                    ), dim=1)

            # skip nms process for ignore torch script export error
            else:
                if output is None:
                    output = torch.cat((
                        conf_scores.unsqueeze(-1),
                        decoded_boxes.repeat(cls.num_classes, 1).view(-1, *decoded_boxes.shape),
                    ), dim=-1).unsqueeze(0)

                else:
                    output = torch.cat((
                        output,
                        torch.cat((
                            conf_scores.unsqueeze(-1),
                            decoded_boxes.repeat(cls.num_classes, 1).view(-1, *decoded_boxes.shape),
                        ), dim=-1).unsqueeze(0)
                    ))

        flt = output.contiguous().view(cls.batch_size, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < cls.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Detection does not have gradient to backward
        must be called with torch.no_grad

        :param ctx:
        :param grad_outputs:
        :return:
        """
        pass
