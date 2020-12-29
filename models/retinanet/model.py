from typing import Union

import numpy as np
import torch
import torch.nn as nn

from lib.box import nms
from models import Model
from .layers import BasicBlock, Bottleneck, PyramidFeatures, Regression
from .layers import BBoxTransform, ClipBoxes
from .anchors import Anchors
from .loss import FocalLoss


class RetinaNet(Model):
    LOSS = FocalLoss

    BLOCK_TYPES = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
    BLOCK_SIZES = {
        18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]
    }

    @classmethod
    def new(cls, num_classes: int, batch_size: int, block: int = 101,
            config=None, **kwargs):

        assert block in cls.BLOCKS, f"Only support {cls.BLOCKS.keys()}"

        return cls(num_classes, batch_size,
                   cls.BLOCK_TYPES[block], cls.BLOCK_SIZES[block], **kwargs)

    def __init__(self, num_classes: int, batch_size: int, block: Union[BasicBlock, Bottleneck], layers,
                 prior: float = .01):
        super(RetinaNet, self).__init__()
        self.batch_size = batch_size
        self.inplanes = 64

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self.make_layer(block, 64, self.inplanes, layers[0])
        self.layer2 = self.make_layer(block, 128, self.inplanes, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, self.inplanes, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, self.inplanes, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [
                self.layer2[layers[1] - 1].conv2.out_channels,
                self.layer3[layers[2] - 1].conv2.out_channels,
                self.layer4[layers[3] - 1].conv2.out_channels
            ]

        elif block == Bottleneck:
            fpn_sizes = [
                self.layer2[layers[1] - 1].conv3.out_channels,
                self.layer3[layers[2] - 1].conv3.out_channels,
                self.layer4[layers[3] - 1].conv3.out_channels
            ]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regression = Regression(256, 4)
        self.classification = Regression(256, num_classes)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.loss = FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.prior = prior

        self.regression.output.weight.data.fill_(0)
        self.regression.output.bias.data.fill_(0)

        self.classification.output.weight.data.fill_(0)
        self.classification.output.bias.data.fill_(-np.log((1. - self.prior) / self.prior))

        self.freeze_bn()

    @staticmethod
    def make_layer(block, inplanes, planes, blocks, stride=1):
        down_sample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        ) if stride != 1 or inplanes != planes * block.expansion else None

        return nn.Sequential(
            block(inplanes, planes, stride, down_sample),
            *[block(planes * block.expansion, planes) for _ in range(1, blocks)]
        )

    def freeze_bn(self):
        """ Freeze BatchNorm layers. """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.head(img_batch)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        regression = torch.cat((self.regressionModel(feature) for feature in features), dim=1)
        classification = torch.cat((self.classificationModel(feature) for feature in features), dim=1)
        anchors = self.anchors(img_batch)

        if self.training:
            return self.loss(classification, regression, anchors, annotations)

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        scores, *_ = torch.max(classification, dim=2, keepdim=True)
        scores_over_thresh = (scores > .05)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            return torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx, *_ = nms(transformed_anchors.view(-1, 4), scores.view(-1, 1), .5)

        return (*classification[0, anchors_nms_idx, :].max(dim=1), transformed_anchors[0, anchors_nms_idx, :])
