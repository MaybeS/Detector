from typing import Tuple, List, Union
from math import sqrt
from itertools import product
from collections import namedtuple

import torch

PriorSpec = namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


class PriorBox(object):
    SPEC = PriorSpec

    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 variance: List[int] = None, aspect_ratios: List[List[int]] = None,
                 steps: List[int] = None, feature_map: List[int] = None,
                 min_sizes: List[int] = None, max_sizes: List[int] = None,
                 config: List[Union[Tuple, PriorSpec]] = None,
                 clip: bool = True, **kwargs):
        super(PriorBox, self).__init__()

        self.size = size
        self.size_ = size[0]
        self.variance = variance or [.1, .2]

        if config is None:
            self.config = [
                PriorSpec(feature_map, step, (min_size, max_size), aspect_ratio)
                for aspect_ratio, step, feature_map, min_size, max_size in zip(
                    aspect_ratios or [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                    steps or [8, 16, 32, 64, 100, 300],
                    feature_map or [38, 19, 10, 5, 3, 1],
                    min_sizes or [21, 45, 99, 153, 207, 261],
                    max_sizes or [45, 99, 153, 207, 261, 315],
                )
            ]

        elif all(isinstance(c, PriorSpec) for c in config):
            self.config = config

        else:
            self.config = list(map(lambda c: PriorSpec(*c), config))

        self.num_priors = len(self.config)
        self.clip = clip

        if any(filter(lambda x: x <= 0, self.variance)):
            raise ValueError('Variances must be greater than 0')

    def forward(self):
        priors = []

        for spec in self.config:
            scale = self.size_ / spec.shrinkage
            box_min, box_max = spec.box_sizes

            for j, i in product(range(spec.feature_map_size), repeat=2):
                x_center, y_center = (i + .5) / scale, (j + .5) / scale

                # small sized square box
                size = box_min
                h = w = size / self.size_
                priors.append([x_center, y_center, w, h])

                # big sized square box
                size = sqrt(box_max * box_min)
                h = w = size / self.size_
                priors.append([x_center, y_center, w, h])

                # change h/w ratio of the small sized box
                size = box_min
                h = w = size / self.size_
                for ratio in map(sqrt, spec.aspect_ratios):
                    priors.append([x_center, y_center, w * ratio, h / ratio])
                    priors.append([x_center, y_center, w / ratio, h * ratio])

        priors = torch.tensor(priors)

        if self.clip:
            priors = torch.clamp(priors, 0., 1.)

        return priors
