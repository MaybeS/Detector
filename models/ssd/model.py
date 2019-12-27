import re
from typing import List, Iterable, Tuple, Union
from functools import reduce
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models import Model
from .loss import Loss
from .detector import Detector
from .priorbox import PriorBox
from .layers import GraphPath, Warping, SeperableConv2d


class SSD(nn.Module, Model):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Custom SSD backbone requires below things
        - backbone: return feature layers
        - extra: return extra layers
        - head: return location, confidence layers as tuple
        - APPENDIX: list of extract information (index, preprocess, name)
            e.g. [(23, nn.BatchNorm2d(512), 'L2Norm'), (35, None, None)]
    """
    LOSS = Loss

    @classmethod
    def new(cls, num_classes: int, batch_size: int, size: Tuple[int, int] = (300, 300),
            base=None, config=None, **kwargs):
        base = cls.get(f'SSD_{base}', SSD_VGG16)

        backbone = base.backbone(pretrained=True)
        extras = list(base.extra())
        loc, conf = base.head(backbone, extras, num_classes)
        appendix = base.APPENDIX
        prior = base.PRIOR
        config = config or {}

        return cls(num_classes, batch_size, size,
                   backbone, extras, loc, conf, appendix, prior,
                   config, **kwargs)

    def __init__(self, num_classes: int, batch_size: int, size: Tuple[int, int],
                 backbone, extras, loc, conf, appendix, prior,
                 config=None, warping: bool = False, warping_mode: str = 'sum'):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.size = size
        self.appendix = appendix
        self.config = config or {}

        self.priors = PriorBox(**self.config, config=prior).forward()

        self.features = backbone
        self.extras, self.loc, self.conf = map(nn.ModuleList, (extras, loc, conf))

        for _, layer, name in self.appendix:
            if isinstance(layer, nn.Module):
                self.add_module(name, layer)

        self.warping = warping
        self.warping_mode = warping_mode

    def detect(self, loc: torch.Tensor, conf: torch.Tensor, prior: torch.Tensor) \
            -> torch.Tensor:
        if self.training:
            raise RuntimeError('use detect after enable eval mode')

        with torch.no_grad():
            result = Detector.forward(loc, F.softmax(conf, dim=-1), prior)

        return result

    def eval(self):
        super(SSD, self).eval()
        Detector.init(self.num_classes, self.batch_size)

    def train(self, mode: bool = True):
        super(SSD, self).train(mode)

    def forward(self, x: torch.Tensor) \
            -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch, topk, 7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors, num_classes]
                    2: localization layers, Shape: [batch, num_priors*4]
                    3: priorbox layers, Shape: [2, num_priors*4]
        """
        def _forward(tensor: torch.Tensor, module: nn.Module) \
                -> torch.Tensor:
            return module.forward(tensor)

        if self.warping == 'first':
            x = Warping.forward(x, self.warping_mode)

        start, sources = 0, []

        # forward layers for extract sources
        for index, layer, *_ in self.appendix:
            x = reduce(_forward, [x, *self.features[start:index]])

            if isinstance(layer, GraphPath):
                x, y = layer(x, self.features[index])
                index += 1

            elif layer is not None:
                y = layer(x)

            else:
                y = x

            sources.append(y)
            start = index

        # forward remain parts
        x = reduce(_forward, [x, *self.features[start:]])

        for i, layer in enumerate(self.extras):
            x = _forward(x, layer)
            sources.append(x)

        if self.warping == 'all':
            sources = list(map(lambda s: Warping.forward(s, self.warping_mode), sources))

        elif self.warping == 'head':
            sources[0] = Warping.forward(sources[0], self.warping_mode)
            sources[1] = Warping.forward(sources[1], self.warping_mode)

        def refine(source: torch.Tensor):
            return source.permute(0, 2, 3, 1).contiguous()

        def reshape(tensor: torch.Tensor):
            return torch.cat(tuple(map(lambda t: t.view(t.size(0), -1), tensor)), 1)

        locations, confidences = map(reshape, zip(*[(refine(loc(source)), refine(conf(source)))
                                                    for source, loc, conf in zip(sources, self.loc, self.conf)]))

        locations = locations.view(self.batch_size, -1, 4)
        confidences = confidences.view(self.batch_size, -1, self.num_classes)

        output = (locations, confidences, self.priors.to(x.device))

        if not self.training:
            output = self.detect(*output).to(x.device)

        return output

    @staticmethod
    def initializer(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                m.bias.data.zero_()

    def load(self, state_dict: dict = None):
        try:
            self.load_state_dict(state_dict)

        # if state dict is only vgg features
        except RuntimeError:
            try:
                self.features.load_state_dict(state_dict)

                self.extras.apply(self.initializer)
                self.loc.apply(self.initializer)
                self.conf.apply(self.initializer)

            # if state dict is legacy pre-trained features
            except RuntimeError:
                def refine(text, replace_map_, pattern_):
                    return pattern_.sub(lambda m: next(k for k, v in replace_map_.items() if m.group(0) in v), text)

                remove_prefix = ['source_layer_add_ons']
                replace_map = {
                    # https://github.com/qfgaohao/pytorch-ssd weights
                    'features': ['vgg', 'base_net'],
                    'loc': ['regression_headers'], 'conf': ['classification_headers'],
                    'extras.0.0': ['extras.0'], 'extras.0.2': ['extras.1'],
                    'extras.1.0': ['extras.2'], 'extras.1.2': ['extras.3'],
                    'extras.2.0': ['extras.4'], 'extras.2.2': ['extras.5'],
                    'extras.3.0': ['extras.6'], 'extras.3.2': ['extras.7'],

                    # https://github.com/qfgaohao/pytorch-ssd mobilenet weights
                    '.conv.0.0.': ['.conv.0.'], '.conv.0.1.': ['.conv.1.'],
                    '.conv.1.': ['.conv.3.'], '.conv.2.': ['.conv.4.'],
                }
                pattern = re.compile('|'.join(chain(*replace_map.values())))

                self.load_state_dict(state_dict.__class__({
                    refine(key, replace_map, pattern): value for key, value in state_dict.items()
                    if not any(map(key.startswith, remove_prefix))
                }), strict=False)

        except AttributeError:
            self.extras.apply(self.initializer)
            self.loc.apply(self.initializer)
            self.conf.apply(self.initializer)

    @classmethod
    def extra(cls, in_channels: int = 1024) \
            -> Iterable[nn.Module]:
        pass

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:
        pass


class SSD_VGG16(SSD):
    BACKBONE = models.vgg16
    APPENDIX = [(23, nn.BatchNorm2d(512), 'L2Norm'), (35, None, None)]
    EXTRAS = [(256, 512, 1), (128, 256, 1), (128, 256, 0), (128, 256, 0)]
    BOXES = [4, 6, 6, 6, 4, 4]

    PRIOR = [
        (38, 8, (30, 60), (2,)),
        (19, 16, (60, 111), (2, 3)),
        (10, 32, (111, 162), (2, 3)),
        (5, 64, (162, 213), (2, 3)),
        (3, 100, (213, 264), (2,)),
        (1, 300, (264, 315), (2,)),
    ]

    @classmethod
    def backbone(cls, pretrained):
        backbone = cls.BACKBONE(pretrained=pretrained).features[:-1]
        backbone[16].ceil_mode = True

        for i, layer in enumerate([
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        ], 30):
            backbone.add_module(str(i), layer)

        return backbone

    @classmethod
    def extra(cls, in_channels: int = 1024) \
            -> Iterable[nn.Module]:

        for mid_channels, out_channels, option in cls.EXTRAS:
            yield nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3,
                          stride=1 + option, padding=option),
                nn.ReLU(),
            )
            in_channels = out_channels

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:
        def gen(count_feature):
            count, feature = count_feature
            return nn.Conv2d(feature, count * 4, kernel_size=3, padding=1), \
                nn.Conv2d(feature, count * num_classes, kernel_size=3, padding=1)

        return tuple(zip(*map(gen, zip(cls.BOXES, chain(
            map(lambda layer: layer.out_channels, map(lambda index: backbone[index[0] - 2], cls.APPENDIX)),
            map(lambda module: module[2].out_channels, extras)),
        ))))


class SSD_MOBILENET2_LITE(SSD):
    BACKBONE = models.mobilenet_v2
    APPENDIX = [(14, GraphPath('conv', 1), 'GraphPath'), (19, None, None)]
    EXTRAS = [(512, .2), (256, .25), (256, .5), (64, .25)]

    PRIOR = [
        (19, 16, (60, 105), (2, 3)),
        (10, 32, (105, 150), (2, 3)),
        (5, 64, (150, 195), (2, 3)),
        (3, 100, (195, 240), (2, 3)),
        (2, 150, (240, 285), (2, 3)),
        (1, 300, (285, 330), (2, 3)),
    ]

    @classmethod
    def backbone(cls, pretrained):
        backbone = cls.BACKBONE(pretrained=pretrained).features

        return backbone

    @classmethod
    def extra(cls, in_channels: int = 1280) \
            -> Iterable[nn.Module]:

        for feature, ratio in cls.EXTRAS:
            yield models.mobilenet.InvertedResidual(in_channels, feature, stride=2, expand_ratio=ratio)
            in_channels = feature

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int, width_mult: float = 1.0) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:
        in_channels = round(576 * width_mult)

        regression_headers = [
            SeperableConv2d(in_channels, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(1280, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ]

        classification_headers = [
            SeperableConv2d(in_channels, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ]

        return regression_headers, classification_headers
