import json
from pathlib import Path


class Config:
    size = (300, 300)
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    num_priors = 6
    variance = [.1, .2]
    feature_map_x = [38, 19, 10, 5, 3, 1]
    feature_map_y = [38, 19, 10, 5, 3, 1]
    min_sizes = [21, 45, 99, 153, 207, 261]
    max_sizes = [45, 99, 153, 207, 261, 315]
    steps = [8, 16, 32, 64, 100, 300]
    clip = True

    def __init__(self, path: str):
        if path is not None and Path(path).exists():
            with open(path) as f:
                context = json.load(f)
            for key, value in context.items():
                setattr(self, key, value)

    @property
    def data(self):
        return {
            attr: getattr(self, attr)
            for attr in filter(lambda attr: not attr.startswith('__') and attr != 'data' and
                                            not callable(getattr(self, attr)), dir(self))
        }
