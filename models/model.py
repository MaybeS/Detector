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
        return cls.LOSS(*args, **kwargs)

    def eval(self):
        super(Model, self).eval()
        self.batch_size = 1

    def train(self, mode: bool = True):
        super(Model, self).train(mode)
        self.batch_size = self.batch_size_

        if not mode:
            self.batch_size = 1
