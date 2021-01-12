import torch.nn as nn
import torch.optim as optim

from utils.beholder import Beholder


class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Model(nn.Module, metaclass=Beholder):
    LOSS = None
    OPTIMIZER = optim.SGD, {'lr': .0001, "momentum": .9, "weight_decay": 5e-4}
    SCHEDULER = lambda _: None, {}
    batch_size = 1

    @classmethod
    def new(cls, *args, **kwargs):
        pass

    @classmethod
    def loss(cls, *args, **kwargs):
        try:
            return cls.LOSS(*args, **kwargs)
        except TypeError:
            return cls.LOSS()

    def load(self, state_dict: dict = None):
        if state_dict is not None:
            try:
                self.load_state_dict(state_dict)
            except RuntimeError:
                pass
