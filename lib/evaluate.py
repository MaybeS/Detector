from typing import Tuple, Union

import torch
import numpy as np


def compute_iou_(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def compute_iou(box, boxes, box_area, boxes_area):
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        overlaps[:, i] = compute_iou(boxes2[i], boxes1, area2[i], area1)
    return overlaps


class Evaluator:
    def __init__(self, n_class: int, sample_patch: int = 11, threshold: float = .5):
        self.n_class = n_class
        self.patch = np.linspace(0, 1, sample_patch)
        self.threshold = threshold

        self.TP, self.FP, self.FN = np.zeros((3, sample_patch, n_class), dtype=np.uint32)
        self.gt_counts, self.pd_counts = np.zeros((2, n_class), dtype=np.uint32)

    def update(self, predictions: Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]],
               groundtruths: Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]) \
            -> None:
        """Update predictions and groundtruths for each frames.

        :param predictions: contains (class_ids, scores, bounding_boxes, masks)
        :param groundtruths: contains (class_ids, bounding_boxes, masks)
        :return:
        """
        pd_class_ids, pd_scores, pd_bboxes, pd_masks = predictions
        gt_class_ids, gt_bboxes, gt_masks = groundtruths

        if pd_scores is None:
            pd_scores = np.ones_like(pd_class_ids)

        if pd_bboxes.size > 0:
            iou = self.compute_iou(pd_bboxes, gt_bboxes, self.threshold)

        for klass in range(self.n_class):
            gt_number = np.sum(gt_class_ids == klass)

            self.gt_counts[klass] += gt_number
            self.pd_counts[klass] += (pd_class_ids == klass).sum()

            for p, patch in enumerate(self.patch):
                pd_mask = np.logical_and(pd_class_ids == klass, pd_scores >= patch)
                pd_number = np.sum(pd_mask)

                if pd_number == 0:
                    self.FN[p][klass] += gt_number
                    continue

                true_positive = np.sum(iou[pd_mask][:, gt_class_ids == klass].any(axis=0))

                self.TP[p][klass] += true_positive
                self.FP[p][klass] += pd_number - true_positive
                self.FN[p][klass] += gt_number - true_positive

    @staticmethod
    def compute_iou(tar_boxes: np.ndarray, src_boxes: np.ndarray, threshold: float) \
            -> np.ndarray:
        iou = compute_overlaps(tar_boxes, src_boxes) >= threshold
        axis, argm = np.arange(np.size(iou, 0)), iou.argmax(axis=1)
        outputs = np.zeros_like(iou)
        outputs[axis, argm] = iou[axis, argm]
        return outputs >= threshold

    @property
    def precision(self) \
            -> np.ndarray:
        return np.nan_to_num(self.TP / (self.TP + self.FP)).mean(axis=0)

    @property
    def recall(self) \
            -> np.ndarray:
        return np.nan_to_num(self.TP / (self.TP + self.FN)).mean(axis=0)

    @property
    def mAP(self) \
            -> np.ndarray:
        prev_recall, ap = np.zeros((2, self.n_class))

        for precision, recall in zip(self.precision[::-1], self.recall[::-1]):
            ap += precision * (recall - prev_recall)
            prev_recall = recall

        return ap

    def dump(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.mAP, self.precision, self.recall
