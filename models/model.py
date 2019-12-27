import torch.nn as nn

from utils.beholder import Beholder


class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Model(metaclass=Beholder):
    LOSS = None
    SCHEDULER = None

    @classmethod
    def new(cls, *args, **kwargs):
        pass

    @classmethod
    def loss(cls, *args, **kwargs):
        return cls.LOSS(*args, **kwargs)
