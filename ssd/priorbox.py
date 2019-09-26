from typing import List
from math import sqrt
from itertools import product

import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, size: int = 300,
                 aspect_ratios: List[List[int]] = None,
                 variance: List[int] = None,
                 feature_maps: List[int] = None,
                 min_sizes: List[int] = None,
                 max_sizes: List[int] = None,
                 steps: List[int] = None,
                 clip: bool = True, **kwargs):
        super(PriorBox, self).__init__()

        self.size = size
        self.aspect_ratios = aspect_ratios or [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.num_priors = len(self.aspect_ratios)
        self.variance = variance or [.1, .2]
        self.feature_maps = feature_maps or [38, 19, 10, 5, 3, 1]
        self.min_sizes = min_sizes or [30, 60, 111, 162, 213, 264]
        self.max_sizes = max_sizes or [60, 111, 162, 213, 264, 315]
        self.steps = steps or [8, 16, 32, 64, 100, 300]
        self.clip = clip

        if any(filter(lambda x: x <= 0, self.variance)):
            raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for step, min_size, max_size, ratio, feature in zip(self.steps,
                                                            self.min_sizes,
                                                            self.max_sizes,
                                                            self.aspect_ratios,
                                                            self.feature_maps):
            for i, j in product(range(feature), repeat=2):
                f = self.size / step

                cx, cy = (j + .5) / f, (i + .5) / f

                s = min_size / self.size
                mean += [cx, cy, s, s]

                p = sqrt(s * (max_size / self.size))
                mean += [cx, cy, p, p]

                for r in ratio:
                    mean += [cx, cy, s * sqrt(r), s / sqrt(r)]
                    mean += [cx, cy, s / sqrt(r), s * sqrt(r)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)

        return output
