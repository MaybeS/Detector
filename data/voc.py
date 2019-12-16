from pathlib import Path
from typing import Tuple
import xml.etree.ElementTree as ET

import numpy as np
import skimage
import torch
import torch.utils.data as data

from data import Dataset


class VOC(data.Dataset, Dataset):
    num_classes = 21
    class_names = ('BACKGROUND',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    TRAIN_IMAGE_SET = 'ImageSets/Main/trainval.txt'
    TEST_IMAGE_SET = 'ImageSets/Main/test.txt'

    IMAGE_DIR = 'JPEGImages'
    IMAGE_EXT = '.jpg'
    DETECTION_DIR = 'Annotations'
    DETECTION_EXT = '.xml'

    SHAPE = 300, 300

    cfg = {
        'num_classes': 21,
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
        'name': 'VOC',
    }

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 train: bool = True,
                 eval_only: bool = False):
        self.name = 'VOC'
        self.root = Path(root)

        self.transform = transform
        self.target_transform = target_transform or self.target_trans
        self.eval_only = eval_only
        self.fail_id = set()

        if eval_only:
            self.images = list(sorted(self.root.glob(f'*{self.IMAGE_EXT}')))

        else:
            with open(str(self.root.joinpath(self.TRAIN_IMAGE_SET if train else self.TEST_IMAGE_SET))) as f:
                ids = f.read().splitlines()

            self.images = sorted(self.root.joinpath(self.IMAGE_DIR).joinpath(f'{ii}{self.IMAGE_EXT}')
                                 for ii in ids)
            self.detections = sorted(self.root.joinpath(self.DETECTION_DIR).joinpath(f'{ii}{self.DETECTION_EXT}')
                                     for ii in ids)

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
        boxes, labels = [], []

        for element in ET.parse(str(self.detections[index])).findall("object"):
            class_name = element.find('name').text.lower().strip()

            if class_name in self.class_names:
                bbox = element.find('bndbox')

                x1, y1 = float(bbox.find('xmin').text) - 1, float(bbox.find('ymin').text) - 1
                x2, y2 = float(bbox.find('xmax').text) - 1, float(bbox.find('ymax').text) - 1

                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_names.index(class_name) - 1)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
