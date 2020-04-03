from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


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
                np.expand_dims(cls.grid(step=(20/size)), 0)).to(x.device)

        shape = grid.shape
        grid = grid.view(1, -1).repeat(1, x.shape[0]).view(-1, *shape[1:])

        output = F.grid_sample(x, grid)

        if mode == 'replace':
            pass

        elif mode == 'fit':
            size = np.array(shape[1:-1])
            scale = 2 ** -.5

            x, y = (1 - scale) / 2 * size
            w, h = size * scale

            resized = F.interpolate(output, size=(int(w), int(h)))
            output[:, :, x:x + w, y:y + h] += resized
            output[:, :, x:x + w, y:y + h] /= 2

        elif mode == 'sum':
            output = output + x

        elif mode == 'average':
            output = torch.cat((
                torch.unsqueeze(output, 0),
                torch.unsqueeze(x, 0)
            ), 0).mean(axis=0)

        elif mode == 'concat':
            output = torch.cat((output, x), -1)

        else:
            raise NotImplementedError(f'Warping {mode} is not implemented!')

        return output

    @classmethod
    def grid(cls, wide: int = 10, step: float = 1.) \
            -> np.ndarray:
        arange = np.arange(-wide, wide + step, step)
        grid = np.array(np.meshgrid(arange, arange), dtype=np.float32).transpose(1, 2, 0)
        shape = grid.shape
        grid = np.apply_along_axis(lambda x: cls.ray2pix([*x, 3]), 1, grid.reshape(-1, 2))

        grid[:, 0] -= cls.PADDING[0]
        grid[:, 1] -= cls.PADDING[1]

        grid[:, 0] /= cls.SHAPE[0]
        grid[:, 1] /= cls.SHAPE[1]

        return grid.reshape(shape) * 2 - 1

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
