import json
from typing import Tuple, List


class Config:
    """Config stack layers

    - Default config
    - Model default config
    - Load from config file
    - User argument config
    """

    size = (300, 300)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    num_priors = 6
    variance = (.1, .2)
    feature_map = (38, 19, 10, 5, 3, 1)
    sizes = ((30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315))
    steps = (8, 16, 32, 64, 100, 300)
    clip = True

    optimizer = {
        "lr": .0001,
        "momentum": .9,
        "weight_decay": 5e-4
    }
    scheduler = {
        "factor": .1,
        "patience": 3,
    }

    def __init__(self, path: str, prior: List[Tuple] = None):
        if prior is not None:
            for key, value in zip(('feature_map', 'steps', 'sizes', 'aspect_ratios'), zip(*prior)):
                self.update(key, value)

        if path is not None:
            try:
                with open(path) as f:
                    for key, value in json.load(f).items():
                        self.update(key, value)

            except (FileNotFoundError, RuntimeError) as e:
                print(f'Configfile {path} is not exists or can not open')

    def update(self, key, value):
        if isinstance(getattr(self, key, None), dict):
            getattr(self, key).update(value)
        else:
            setattr(self, key, value)

    def sync(self, arguments: dict):
        for key, value in arguments.items():
            if hasattr(arguments, key):
                setattr(arguments, key, value)
            if key in self.dump.keys():
                self.update(key, value)

    @property
    def dump(self):
        return {
            attr: getattr(self, attr)
            for attr in filter(lambda attr: not attr.startswith('__') and attr != 'dump' and
                                            not callable(getattr(self, attr)), dir(self))
        }
