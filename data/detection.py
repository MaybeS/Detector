from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import skimage
import torch
import torch.utils.data as data

from data import Dataset


class Detection(data.Dataset, Dataset):
    num_classes = 2

    IMAGE_DIR = 'images'
    IMAGE_EXT = '.jpg'
    DETECTION_DIR = 'annotations'
    DETECTION_EXT = '.txt'

    SHAPE = 1920, 512

    cfg = {
        'num_classes': 2,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'Detection',
    }

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 eval_only=False,
                 max_step=15):
        self.name = 'Detection'
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

        self.shape = self.pull_image(0).shape

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

        return image

    def pull_anno(self, index: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        try:
            annotations = pd.read_csv(str(self.detections[index]), header=None).values.astype(np.float32)
            annotations = annotations[annotations[:, 2] - annotations[:, 0] > 50]

        except pd.errors.EmptyDataError:
            annotations = np.empty((0, 4), dtype=np.float32)

        return annotations, np.zeros(np.size(annotations, 0), dtype=np.uint8)
