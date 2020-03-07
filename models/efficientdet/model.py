from typing import Tuple
import math

import torch
import torch.nn as nn
from torchvision.ops import nms

from models import Model
from .efficientnet import EfficientNet
from .retinahead import RetinaHead
from .bifpn import BIFPN
from .layers import Anchors, ClipBoxes, BBoxTransform
from .loss import FocalLoss


class EfficientDet(Model):
    LOSS = FocalLoss
    BACKBONE = EfficientNet
    NECK = BIFPN
    HEAD = RetinaHead
    FPN_D, FPN_W, CLASS_D, OUT = 0, 0, 0, 5

    @classmethod
    def new(cls, num_classes: int, batch_size: int,
            config=None, **kwargs):
        assert cls is not EfficientDet, "Create new model instance by subclass caller"

        backbone = EfficientDet.BACKBONE.from_pretrained(cls.BACKBONE)
        neck = cls.NECK(in_channels=backbone.get_list_features()[-cls.OUT:],
                        out_channels=cls.FPN_W, stack=cls.FPN_D, num_outs=cls.OUT)
        head = cls.HEAD(num_classes=num_classes, in_channels=cls.FPN_W)

        return cls(num_classes, batch_size,
                   backbone, neck, head,
                   config, **kwargs)

    def __init__(self, num_classes: int, batch_size: int,
                 backbone: nn.Module, neck: nn.Module, head: nn.Module,
                 config=None, **kwargs):
        super(EfficientDet, self).__init__()
        self.num_classes = num_classes
        self.batch_size_ = batch_size
        self.batch_size = batch_size
        self.config = config

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.anchors = Anchors(size=config.size).forward()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes(size=config.size)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                module.eval()

    def detect(self, x: torch.Tensor,
               classifications: torch.Tensor, regressions: torch.Tensor, anchors: torch.Tensor) \
            -> torch.Tensor:
        if self.training:
            raise RuntimeError('use detect after enable eval mode')

        with torch.no_grad():
            output = torch.zeros(self.batch_size, self.num_classes, self.config.nms_top_k, 5) \
                if self.config.nms else None

            for batch_index, (classification, regression) in enumerate(zip(classifications, regressions)):
                transformed_anchors = self.regressBoxes(anchors, regression)
                transformed_anchors = self.clipBoxes(transformed_anchors, x)
                conf_scores, classes = classification.max(dim=-1)

                if self.config.nms:
                    for class_index in range(1, self.num_classes):
                        class_mask = classes == class_index
                        conf_mask = conf_scores[class_mask].gt(self.config.conf_thresh)

                        if conf_mask.sum() == 0:
                            continue

                        boxes = transformed_anchors[class_mask][conf_mask]
                        scores = conf_scores[class_mask][conf_mask]

                        nms_index = nms(boxes, scores, iou_threshold=self.config.nms_thresh)
                        (size, *_) = nms_index.size()
                        output[batch_index, class_index, :min(size, self.config.nms_top_k)] = torch.cat((
                            scores[nms_index[:self.config.nms_top_k]].unsqueeze(1),
                            boxes[nms_index[:self.config.nms_top_k]]
                        ), dim=1)

                # skip nms process for ignore torch script export error
                else:
                    if output is None:
                        output = torch.cat((
                            conf_scores.unsqueeze(-1),
                            transformed_anchors.repeat(self.num_classes, 1).view(-1, *transformed_anchors.shape),
                        ), dim=-1).unsqueeze(0)

                    else:
                        output = torch.cat((
                            output,
                            torch.cat((
                                conf_scores.unsqueeze(-1),
                                transformed_anchors.repeat(self.num_classes, 1).view(-1, *transformed_anchors.shape),
                            ), dim=-1).unsqueeze(0)
                        ))

        return output

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = self.neck(features[-self.OUT:])

        def reshape(tensor: torch.Tensor) \
                -> torch.Tensor:
            return torch.cat(tensor, dim=1)

        output = *map(reshape, self.head(features)), self.anchors.clone().to(x.device)

        if not self.training:
            output = self.detect(x, *output)

        return output


class D0(EfficientDet):
    BACKBONE = 'efficientnet-b0'
    FPN_D, FPN_W, CLASS_D = 2, 64, 3


class D1(EfficientDet):
    BACKBONE = 'efficientnet-b1'
    FPN_D, FPN_W, CLASS_D = 3, 88, 3


class D2(EfficientDet):
    BACKBONE = 'efficientnet-b2'
    FPN_D, FPN_W, CLASS_D = 4, 112, 3


class D3(EfficientDet):
    BACKBONE = 'efficientnet-b3'
    FPN_D, FPN_W, CLASS_D = 5, 160, 4


class D4(EfficientDet):
    BACKBONE = 'efficientnet-b4'
    FPN_D, FPN_W, CLASS_D = 6, 224, 4


class D5(EfficientDet):
    BACKBONE = 'efficientnet-b5'
    FPN_D, FPN_W, CLASS_D = 7, 288, 4


class D6(EfficientDet):
    BACKBONE = 'efficientnet-b6'
    FPN_D, FPN_W, CLASS_D = 8, 384, 5


class D7(EfficientDet):
    BACKBONE = 'efficientnet-b7'
    FPN_D, FPN_W, CLASS_D = 8, 384, 5
