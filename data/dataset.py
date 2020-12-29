from typing import Tuple

import numpy as np
import torch
from torch.utils import data

from utils.beholder import Beholder


class Dataset(data.Dataset, metaclass=Beholder):
    num_classes = 1
    class_names = ('BG', )

    @staticmethod
    def collate(batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on
                                     0 dim
        """
        images, targets, *_ = zip(*batch)

        return torch.stack(images, 0), list(map(torch.FloatTensor, targets))

    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0

    def pull_name(self, index: int) -> str:
        pass

    def pull_item(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def pull_image(self, index: int) -> np.ndarray:
        pass

    def pull_anno(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        pass
