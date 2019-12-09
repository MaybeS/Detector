from typing import Tuple, List
from math import sqrt
from itertools import product

import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 aspect_ratios: List[List[int]] = None,
                 variance: List[int] = None,
                 feature_map_x: List[int] = None,
                 feature_map_y: List[int] = None,
                 min_sizes: List[int] = None,
                 max_sizes: List[int] = None,
                 steps: List[int] = None,
                 clip: bool = True, **kwargs):
        super(PriorBox, self).__init__()

        self.size = size
        self.aspect_ratios = aspect_ratios or [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.num_priors = len(self.aspect_ratios)
        self.variance = variance or [.1, .2]
        self.feature_map_x = feature_map_x or [38, 19, 10, 5, 3, 1]
        self.feature_map_y = feature_map_y or [38, 19, 10, 5, 3, 1]
        self.min_sizes = min_sizes or [21, 45, 99, 153, 207, 261]
        self.max_sizes = max_sizes or [45, 99, 153, 207, 261, 315]
        self.steps = steps or [8, 16, 32, 64, 100, 300]
        self.clip = clip

        if any(filter(lambda x: x <= 0, self.variance)):
            raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for step, ratio, min_size, max_size, \
            feature_x, feature_y in zip(self.steps, self.aspect_ratios,
                                        self.min_sizes, self.max_sizes,
                                        self.feature_map_x, self.feature_map_y):

            fx, fy = self.size[0] / step, self.size[1] / step
            sx, sy = min_size / self.size[0], min_size / self.size[0]
            px, py = sqrt(sx * (max_size / self.size[0])), sqrt(sx * (max_size / self.size[0]))

            for i, j in product(range(feature_x), range(feature_y)):
                cx, cy = (j + .5) / fy, (i + .5) / fx

                mean += [cx, cy, sx, sy]
                mean += [cx, cy, px, py]

                for r in ratio:
                    mean += [cx, cy, sx * sqrt(r), sy / sqrt(r)]
                    mean += [cx, cy, sx / sqrt(r), sy * sqrt(r)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)

        return output
