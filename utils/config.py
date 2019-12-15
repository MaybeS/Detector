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

    optimizer = {
        "lr": .001,
        "momentum": .9,
        "weight_decay": 5e-4
    }
    scheduler = {
        "factor": .1,
        "patience": 3,
    }

    def __init__(self, path: str):
        try:
            with open(path) as f:
                for key, value in json.load(f).items():
                    self.update(key, value)
        except (FileNotFoundError, RuntimeError) as e:
            print(f'Configfile {path} is not exists or can not open')

    def update(self, key, value):
        if isinstance(getattr(self, key), dict):
            getattr(self, key).update(value)
        else:
            setattr(self, key, value)

    def sync(self, arguments: dict):
        for key, value in arguments.items():
            if key in self.data.keys():
                self.update(key, value)

    @property
    def data(self):
        return {
            attr: getattr(self, attr)
            for attr in filter(lambda attr: not attr.startswith('__') and attr != 'data' and
                                            not callable(getattr(self, attr)), dir(self))
        }
