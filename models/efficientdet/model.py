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
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError('use detect after enable eval mode')

        with torch.no_grad():
            output = torch.zeros(self.batch_size, self.num_classes, self.config.nms_top_k, 5).to(x.device)

            for batch_index, (classification, regression) in enumerate(zip(classifications, regressions)):
                transformed_anchors = self.regressBoxes(anchors, regression)
                transformed_anchors = self.clipBoxes(transformed_anchors, x)

                scores = torch.max(classification, dim=-1, keepdim=True)[0]
                scores_over_thresh = (scores < self.config.conf_thresh)[:, 0]

                # if scores_over_thresh.sum() == 0:
                #     return output

                classification = classification[scores_over_thresh, :]
                transformed_anchors = transformed_anchors[scores_over_thresh, :]
                scores = scores[scores_over_thresh, :]

                anchors_nms_idx = nms(transformed_anchors, scores[:, 0], iou_threshold=self.config.nms_thresh)
                nms_scores, nms_class = classification[anchors_nms_idx, :].max(dim=1)

                return nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]

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
