from typing import List, Iterable, Tuple, Union
from functools import reduce
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from .detector import Detector
from .priorbox import PriorBox
from .layers import L2Norm, Warping


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size, backbone, extras, loc, conf, num_classes: int, cfg=None,
                 warping: bool=False):
        super(SSD, self).__init__()
        self.size = size
        self.num_classes = num_classes
        self.cfg = cfg or {}

        self.priorbox = PriorBox(**self.cfg)
        self.priors = Variable(self.priorbox.forward(), requires_grad=False)

        self.features = backbone
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        self.warping = warping

    def detect(self, loc: torch.Tensor, conf: torch.Tensor, prior: torch.Tensor) \
            -> torch.Tensor:
        if self.training:
            raise RuntimeError('use detect after enable eval mode')

        with torch.no_grad():
            result = Detector.forward(loc, F.softmax(conf, dim=-1), prior)

        return result

    def eval(self):
        super(SSD, self).eval()
        Detector.init(self.num_classes)

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
        f = lambda param, layer: layer(param)

        sources = list()

        x = reduce(f, [x, *self.features[:23]])
        sources.append(self.L2Norm(x))

        x = reduce(f, [x, *self.features[23:]])
        sources.append(x)

        for i, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                sources.append(x)

        if self.warping:
            sources = list(map(lambda s: Warping.forward(s), sources))

        def refine(source: torch.Tensor):
            return source.permute(0, 2, 3, 1).contiguous()

        def reshape(tensor: torch.Tensor):
            return torch.cat(tuple(map(lambda x: x.view(x.size(0), -1), tensor)), 1)

        loc, conf = map(reshape, zip(*[(refine(loc(source)), refine(conf(source)))
                                       for source, loc, conf in zip(sources, self.loc, self.conf)]))

        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors.to(x.device),
        )

        if not self.training:
            output = self.detect(*output).to(x.device)

        return output

    @staticmethod
    def initializer(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()

    def load(self, state_dict: dict = None):
        if state_dict is None:
            self.extras.apply(self.initializer)
            self.loc.apply(self.initializer)
            self.conf.apply(self.initializer)

        else:
            try:
                self.load_state_dict(state_dict)

            # Legacy handling
            except RuntimeError:
                self.load_state_dict(state_dict.__class__({
                    key.replace('vgg', 'features'): value for key, value in state_dict.items()
                }))


class SSD300(SSD):

    SIZE = 300
    EXTRAS = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    BOXES = [4, 6, 6, 6, 4, 4]

    def __init__(self, num_classes: int, cfg=None, **kwargs):
        backbone = models.vgg16(pretrained=True).features[:-1]
        backbone[16].ceil_mode = True

        for i, layer in enumerate([
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        ], 30):
            backbone.add_module(str(i), layer)

        extras = list(self.extra(backbone[-2].in_channels))
        loc, conf = self.head(backbone, extras, num_classes)

        super(SSD300, self).__init__(self.SIZE, backbone, extras, loc, conf, num_classes, cfg, **kwargs)

    @classmethod
    def extra(cls, in_channels: int = 1024) \
            -> Iterable[nn.Module]:
        kernel = iter(cycle((1, 3)))
        for i, feature in enumerate(cls.EXTRAS):
            if in_channels != 'S':
                yield nn.Conv2d(in_channels, cls.EXTRAS[i + 1] if feature == 'S' else feature, kernel_size=next(kernel),
                                stride=(2 if feature == 'S' else 1), padding=(1 if feature == 'S' else 0))
            in_channels = feature

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:
        def gen(count_layer):
            count, layer = count_layer
            return nn.Conv2d(layer.out_channels, count * 4, kernel_size=3, padding=1), \
                   nn.Conv2d(layer.out_channels, count * num_classes, kernel_size=3, padding=1)

        return tuple(zip(*map(gen, zip(cls.BOXES, list(backbone[21::12]) + extras[1::2]))))
