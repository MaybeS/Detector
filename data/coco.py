from typing import Tuple, List
from pathlib import Path

import numpy as np
import skimage
import torch

from data import Dataset
from pycocotools.coco import COCO as CC


class COCO(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    num_classes = 91
    class_names = ('BACKGROUND', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                   'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                   'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                   'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                   'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                   'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                   'hair drier', 'toothbrush', )

    TRAIN_NAME = 'train2017'
    VAL_NAME = 'val2017'

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 train: bool = True,
                 eval_only: bool = False):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            transform: image transformer.
            target_transform: label transformer.
            train: true if train scope
            eval_only: image transformer.
        """
        self.root = Path(root)
        self.mode_name = self.TRAIN_NAME if train else self.VAL_NAME
        self.transform = transform
        self.target_transform = target_transform or self.target_trans
        self.coco = CC(self.root.joinpath('annotations').joinpath(f'instances_{self.mode_name}.json'))

        self.index = list(self.coco.imgs.keys())
        self.fail_id = set()

        self.eval_only = eval_only
        self.front_only = True

    @staticmethod
    def target_trans(boxes, width, height):
        boxes[:, 1::2] /= height
        boxes[:, :4:2] /= width

        return boxes

    def __getitem__(self, index):
        item = self.pull_item(index)

        return item

    def __len__(self):
        return len(self.index)

    def pull_item(self, index: int):
        fail = 0
        while True:
            idx = (index + fail) % len(self)

            if idx in self.fail_id:
                fail += 1
                continue

            image = self.pull_image(idx)
            height, width, *channel = image.shape

            if not channel:
                image = np.stack((image,) * 3, axis=-1)

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

    def pull_image_info(self, index: int) \
            -> dict:
        try:
            image_ids = self.coco.getImgIds(self.index[index])
            image_info = self.coco.loadImgs(image_ids)

        except (IndexError, KeyError) as e:
            raise Exception(e)

        return next(iter(image_info))

    def pull_image(self, index: int) \
            -> np.ndarray:

        try:
            info = self.pull_image_info(index)
            image = skimage.io.imread(str(self.root.joinpath(self.mode_name).joinpath(info['file_name'])))

        except (KeyError, IndexError) as e:
            raise Exception(e)

        return image

    def pull_anno_info(self, index: int) \
            -> List[dict]:
        try:
            anns_ids = self.coco.getAnnIds(index)
            anns_info = self.coco.loadAnns(anns_ids)

        except IndexError as e:
            raise Exception(e)

        return anns_info

    def pull_anno(self, index: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        try:
            info = self.pull_anno_info(index)
            annotations = np.array(list(map(
                lambda x: (x.get('category_id'), *x.get('bbox')), info
            )), dtype=np.float32)

            if not annotations.size:
                raise IndexError

        except IndexError:
            annotations = np.empty((0, 5), dtype=np.float32)

        annotations[:, 3] += annotations[:, 1]
        annotations[:, 4] += annotations[:, 2]

        return annotations[:, 1:], annotations[:, 0].astype(np.int) - 1
