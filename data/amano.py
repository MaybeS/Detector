from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import skimage
import torch

from data import Dataset


class Amano(Dataset):
    num_classes = 2

    IMAGE_DIR = 'images'
    IMAGE_EXT = '.jpg'
    DETECTION_DIR = 'annotations'
    DETECTION_EXT = '.csv'

    PADDING = 480, 0
    SHAPE = 2880, 2880
    CALIBRATION = {
        'f': [998.4, 998.4],
        'c': [1997, 1473],
        'k': [0.0711, -0.0715, 0, 0, 0],
    }

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 max_step=15,
                 train: bool = True,
                 eval_only: bool = False):
        self.name = 'AMANO'
        self.root = Path(root)

        self.transform = transform
        self.target_transform = target_transform or self.target_trans
        self.eval_only = eval_only
        self.max_step = max_step
        self.front_only = True
        self.fail_id = set()

        if eval_only:
            self.images = list(sorted(self.root.glob(f'*{self.IMAGE_EXT}')))

        else:
            self.images = list(sorted(self.root.joinpath(self.IMAGE_DIR).glob(f'*{self.IMAGE_EXT}')))
            self.detections = list(sorted(self.root.joinpath(self.DETECTION_DIR).glob(f'*{self.DETECTION_EXT}')))
            assert len(self.images) == len(self.detections), \
                "Image and Detections mismatch"

    @staticmethod
    def target_trans(boxes, width, height):
        boxes[:, 1::2] /= height
        boxes[:, :4:2] /= width

        return boxes

    def __getitem__(self, index):
        item = self.pull_item(index)
        return item

    def __len__(self):
        return len(self.images)

    def pull_name(self, index: int):
        return self.images[index].stem

    def pull_item(self, index: int):
        fail = 0
        while True:
            idx = (index + fail) % len(self)

            if idx in self.fail_id:
                fail += 1
                continue

            image = self.pull_image(idx)
            height, width, channels = image.shape

            if self.eval_only is None:
                uniques = np.arange(0)
                boxes = np.empty((uniques.size, 4))
                labels = np.empty((uniques.size, 1))

            else:
                boxes, labels = self.pull_anno(idx)

                if self.target_transform is not None:
                    boxes = self.target_transform(boxes, width, height)

            if self.transform is not None:
                image, boxes, labels = self.transform(image, boxes, labels)

            # axis = np.logical_and(boxes[:, 1] < boxes[:, 3], boxes[:, 0] < boxes[:, 2])
            #
            # boxes = boxes[axis]
            # labels = labels[axis]

            if boxes.size:
                break

            self.fail_id.add(idx)
            fail += 1

        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image).permute(2, 0, 1), target

    def pull_image(self, index: int) \
            -> np.ndarray:

        image = skimage.io.imread(str(self.images[index]))

        return image[self.PADDING[1]:self.PADDING[1]+self.SHAPE[1],
                     self.PADDING[0]:self.PADDING[0]+self.SHAPE[0]]

    def pull_anno(self, index: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(str(self.detections[index]), header=None).values

        annotations = np.empty((0, 4), dtype=np.float32)

        for i, (*checksum, x1, y1, x2, y2, x3, y3, side) in enumerate(df):
            if sum(checksum) == 0:
                continue

            rays = np.array([
                self.pix2ray([x1, y1]),
                self.pix2ray([x2, y2]),
                self.pix2ray([x3, y3]),
            ]) * 3
            rays = np.concatenate((rays, np.array([rays[2] - rays[1] + rays[0]])))
            x4, y4 = self.ray2pix(rays[-1])

            if self.front_only:
                rays[:, -1] -= 1.4
                rays /= rays[:, -1].mean()

                x3, y3 = self.ray2pix(rays[0])
                x4, y4 = self.ray2pix(rays[1])

            rect = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            rect -= self.PADDING

            (l, t), (r, b) = np.nanmin(rect, axis=0), np.nanmax(rect, axis=0)

            annotations = np.concatenate((
                annotations,
                np.expand_dims(np.array([l, t, r, b]), 0)
            ))

        return annotations, np.zeros(np.size(annotations, 0), dtype=np.uint8)

    @classmethod
    def ray2pix(cls, ray):
        nr = ray / np.square(ray).sum()
        d = np.sqrt((nr[:2] * nr[:2]).sum())
        th = np.arctan2(d, nr[-1])
        th = th * (1 + th * (cls.CALIBRATION['k'][0] + th * cls.CALIBRATION['k'][1]))
        q = nr[:2] * (th / d)
        im = np.asarray([[cls.CALIBRATION['f'][0], 0, cls.CALIBRATION['c'][0]],
                         [0, cls.CALIBRATION['f'][1], cls.CALIBRATION['c'][1]],
                         [0, 0, 1]])
        return (im @ np.asarray([[*q, 1]]).T).T.squeeze()[:2]

    @classmethod
    def pix2ray(cls, pix):
        pix = np.array([*pix, 1])
        im = np.asarray([[cls.CALIBRATION['f'][0], 0, cls.CALIBRATION['c'][0]],
                         [0, cls.CALIBRATION['f'][1], cls.CALIBRATION['c'][1]],
                         [0, 0, 1]])

        npix = ray = (np.linalg.inv(im) @ np.array([pix]).T).squeeze()

        for i in range(1000):
            np_temp = ray
            ray = ray / np.square(ray).sum()
            d = np.sqrt((ray[:2] * ray[:2]).sum())
            th = np.arctan2(d, ray[-1])
            th = th * (1 + th * (cls.CALIBRATION['k'][0] + th * cls.CALIBRATION['k'][1]))

            ray = np.array([*ray[:2] * (th / d), 1])
            error = ray - npix
            ray = np_temp - error

            if np.linalg.norm(error) < 1e-12:
                break

        return ray