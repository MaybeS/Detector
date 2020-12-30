from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


class GraphPath(nn.Module):

    def __init__(self, name, index):
        super(GraphPath, self).__init__()
        self.name = name
        self.index = index

    def forward(self, x: torch.Tensor, layer: nn.Module):
        sub = getattr(layer, self.name)

        for layer in sub[:self.index]:
            x = layer(x)

        y = x

        for layer in sub[self.index:]:
            x = layer(x)

        return x, y


class Warping(Function):
    PADDING = 480, 0
    SHAPE = 2880, 2880
    CALIBRATION = {
        'f': [998.4, 998.4],
        'c': [1997, 1473],
        'k': [0.0711, -0.0715, 0, 0, 0],
    }

    @classmethod
    def forward(cls, x: torch.Tensor, mode: str = '', grid: torch.Tensor = None) \
            -> torch.Tensor:
        size = sum(x.shape[2:]) / 2
        if size == 1:
            return x

        if grid is None:
            grid = torch.from_numpy(
                np.expand_dims(cls.grid(wide=10, step=(20/size)), 0)).to(x.device)

        shape = grid.shape
        grid = grid.view(1, -1).repeat(1, x.shape[0]).view(-1, *shape[1:])

        output = F.grid_sample(x, grid)

        if mode == 'sum':
            output = output/100 + x
        elif mode == 'average':
            output = torch.cat((torch.unsqueeze(output, 0), torch.unsqueeze(x, 0)), 0).mean(axis=0)
        elif mode == 'concat':
            output = torch.cat((output, x), -1)
        else:
            raise NotImplementedError(f'Warping {mode} is not implemented!')

        return output

    @classmethod
    def grid(cls, wide: int = 15, step: float = 1.) \
            -> np.ndarray:
        arange = np.arange(-wide, wide, step)
        grid = np.array(np.meshgrid(arange, arange), dtype=np.float32).transpose(1, 2, 0)
        shape = grid.shape
        grid = np.apply_along_axis(lambda x: cls.ray2pix([*x, 3]), 1, grid.reshape(-1, 2))

        grid[:, 0] -= cls.PADDING[0]
        grid[:, 1] -= cls.PADDING[1]

        grid[:, 0] /= cls.SHAPE[0]
        grid[:, 1] /= cls.SHAPE[1]

        return grid.reshape(shape)

    @classmethod
    def ray2pix(cls, ray: Union[List, np.ndarray]) \
            -> np.ndarray:
        ray = np.array(ray)

        if np.all(ray[:2] == 0):
            return np.array(cls.CALIBRATION['c'])

        nr = ray / np.square(ray).sum()
        d = np.sqrt((nr[:2] * nr[:2]).sum())
        th = np.arctan2(d, nr[-1])
        th = th * (1 + th * (cls.CALIBRATION['k'][0] + th * cls.CALIBRATION['k'][1]))
        q = nr[:2] * (th / d)
        im = np.asarray([[cls.CALIBRATION['f'][0], 0, cls.CALIBRATION['c'][0]],
                         [0, cls.CALIBRATION['f'][1], cls.CALIBRATION['c'][1]],
                         [0, 0, 1]], dtype=np.float32)
        return (im @ np.asarray([[*q, 1]], dtype=np.float32).T).T.squeeze()[:2]


class DilatedWeightedConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, *args, **kwargs):
        super(DilatedWeightedConvolution, self).__init__()
        self.step_size = 32
        self.out_size = int(out_feature / 4)
        # TODO assert out_feature is multiples of 4

        self.warping = Warping()
        self.dilated_1 = nn.Conv2d(in_feature, self.out_size * 2, (1, 1))

        self.dilated_3_1 = nn.Conv2d(in_feature, self.step_size, (1, 1))
        self.dilated_3_2 = nn.Conv2d(self.step_size, self.out_size, (3, 3),
                                     dilation=2, padding=2)

        self.dilated_5_1 = nn.Conv2d(in_feature, self.step_size, (1, 1))
        self.dilated_5_2 = nn.Conv2d(self.step_size, self.out_size, (5, 5),
                                     dilation=3, padding=6)

    def forward(self, x: torch.Tensor):
        warped = torch.mean(torch.stack((
            x,
            self.warping.forward(x),
        )), dim=0)


        d1 = self.dilated_1(warped)

        d3 = self.dilated_3_1(warped)
        d3 = self.dilated_3_2(d3)

        d5 = self.dilated_5_1(warped)
        d5 = self.dilated_5_2(d5)

        x += torch.cat((d1, d3, d5), axis=1)

        return x
