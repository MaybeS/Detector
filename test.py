import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from data.dataset import Dataset
from lib.evaluate import Evaluator
from lib.augmentation import Augmentation, Base
from utils.arguments import Arguments


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> nn.Module:

    model.load(torch.load(args.model, map_location=lambda s, l: s))
    model.eval()

    if device.type == 'cuda':
        model = nn.DataParallel(model)
        model.state_dict = model.module.state_dict
        torch.backends.cudnn.benchmark = True

    model.to(device)

    return model


def test(model: nn.Module, dataset: Dataset, transform: Augmentation,
         device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> None:
    evaluator = Evaluator(n_class=dataset.num_classes)
    dest = Path(args.dest)

    for index in tqdm(range(len(dataset))):
        gt_boxes, labels = dataset.pull_anno(index)
        name = dataset.pull_name(index)
        destination = Path(dest).joinpath(f'{name}.txt')

        try:
            if args.overwrite:
                raise AssertionError('Force overwrite enabled')
            detection = pd.read_csv(str(destination), header=None).values

        except (FileNotFoundError, AssertionError, pd.errors.EmptyDataError):
            image = dataset.pull_image(index)
            scale = torch.Tensor([image.shape[1], image.shape[0],
                                  image.shape[1], image.shape[0]]).to(device)

            image = Variable(torch.from_numpy(transform(image)[0]).permute(2, 0, 1).unsqueeze(0)).to(device)

            detections, *_ = model(image).data

            detection = np.empty((0, 6), dtype=np.float32)
            for klass, boxes in enumerate(detections):
                candidates = boxes[boxes[:, 0] >= args.thresh]

                if candidates.size(0) == 0:
                    continue

                candidates[:, 1:] *= scale

                detection = np.concatenate((
                    detection,
                    np.hstack((
                        np.full((np.size(candidates, 0), 1), klass, dtype=np.uint8),
                        candidates.cpu().detach().numpy(),
                    )),
                ))

            pd.DataFrame(detection).to_csv(str(destination), header=None, index=None)

        if not detection.size or not gt_boxes.size:
            continue

        evaluator.update((
            detection[:, 0].astype(np.int),
            detection[:, 1].astype(np.float32),
            detection[:, 2:].astype(np.float32),
            None,
        ), (
            np.ones(np.size(gt_boxes, 0), dtype=np.int),
            gt_boxes.astype(np.float32),
            None,
        ))

    aps, precisions, recalls = [], [], []
    gt_counts, pd_counts = 0, 0

    for klass, (ap, precision, recall) in enumerate(zip(*evaluator.dump())):
        # Skip BG class
        if klass == 0:
            continue

        print(f'AP of {klass}: {ap}')
        print(f'\tPrecision: {precision}, Recall: {recall}')
        print(f'{klass}: Ground Truths: {evaluator.gt_counts[klass]} / Predictions: {evaluator.pd_counts[klass]}')

        aps.append(ap)
        precisions.append(precision)
        recalls.append(recall)
        gt_counts += evaluator.gt_counts[klass]
        pd_counts += evaluator.pd_counts[klass]

    print(f'mAP total: {np.mean(aps)}')
    print(f'\tPrecision: {np.mean(precisions)}, Recall: {np.mean(recalls)}')
    print(f'Ground Truths: {gt_counts} / Predictions: {pd_counts}')

    with open(str(dest.joinpath('results.json')), 'w') as f:
        json.dump({
            'mAP': float(np.mean(aps)),
            'Precision': float(np.mean(precisions)),
            'Recall': float(np.mean(recalls)),
            'GT': int(gt_counts),
            'PD': int(pd_counts),
        }, f)
