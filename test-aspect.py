import json
from typing import Iterator
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data

from data import Dataset
from models import DataParallel
from lib.evaluate import Evaluator
from utils.arguments import Arguments


def arguments(args):
    args.add_argument('--eval-only', required=False, default=False, action='store_true',
                      help="evaluate only, not detecting")
    args.add_argument('--overwrite', required=False, default=False, action='store_true',
                      help="overwrite previous result")
    args.add_argument('--crop', required=False, type=float, default=1/3,
                      help="Crop ratio")

    args.add_argument('--distribution', required=False, default='', type=str,
                      help="Save figure distribution")


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> nn.Module:

    model.load(torch.load(args.model, map_location=lambda s, l: s))
    model.eval()

    if device.type == 'cuda':
        model = DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)

    return model


def test_aspect(model: nn.Module, dataset: Dataset,
                device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> Iterator[dict]:
    evaluator = Evaluator(num_classes=dataset.num_classes, distribution=bool(args.distribution))
    dest = Path(args.dest)
    result = {}

    with tqdm(total=len(dataset)) as tq:
        for index in range(len(dataset)):
            name = dataset.pull_name(index)
            image = dataset.pull_image(index)
            boxes, labels = dataset.pull_anno(index)

            h, w, *_ = image.shape

            target = np.hstack((boxes / np.array((w, h, w, h)), np.expand_dims(labels, axis=1)))

            up, down = image[:int(h*args.crop)], np.flip(image[-int(h*args.crop):], axis=0)
            (up, *_), (down, *_) = map(dataset.transform, (up, down))

            outputs_up, *_ = model(
                Variable(torch.from_numpy(up).permute(2, 0, 1).unsqueeze(0)).to(device))
            outputs_down, *_ = model(
                Variable(torch.from_numpy(down).permute(2, 0, 1).unsqueeze(0)).to(device))

            destination = Path(dest).joinpath(f'{name}.txt')
            detection = np.empty((0, 6), dtype=np.float32)

            for klass, boxes in enumerate(outputs_up):
                candidates = boxes[boxes[:, 0] >= args.thresh]

                if candidates.size(0) == 0:
                    continue

                # calibrate
                candidates[:, [2, 4]] *= args.crop

                detection = np.concatenate((
                    detection,
                    np.hstack((
                        np.full((np.size(candidates, 0), 1), klass, dtype=np.uint8),
                        candidates.cpu().detach().numpy(),
                    )),
                ))

            for klass, boxes in enumerate(outputs_down):
                candidates = boxes[boxes[:, 0] >= args.thresh]

                if candidates.size(0) == 0:
                    continue

                # flip candidates coordinates
                candidates[:, [2, 4]] = 1 - candidates[:, [4, 2]] * args.crop

                detection = np.concatenate((
                    detection,
                    np.hstack((
                        np.full((np.size(candidates, 0), 1), klass, dtype=np.uint8),
                        candidates.cpu().detach().numpy(),
                    )),
                ))

            pd.DataFrame(detection).to_csv(str(destination), header=None, index=None)

            if not args.eval_only:
                if not detection.size or not target.size:
                    continue

                evaluator.update((
                    detection[:, 0].astype(np.int),
                    detection[:, 1].astype(np.float32),
                    detection[:, 2:].astype(np.float32),
                    None,
                ), (
                    target[:, -1].astype(np.int),
                    target[:, :4].astype(np.float32),
                    None,
                ))

            tq.set_postfix(mAP=evaluator.mAP.mean())
            tq.update(args.batch)

    if args.distribution:
        from collections import Counter
        import matplotlib.pyplot as plt

        dest = Path(args.distribution)
        dest.mkdir(exist_ok=True, parents=True)

        total_x, total_y = map(Counter, evaluator.center_total.T)
        positive_x, positive_y = map(Counter, evaluator.center_positive.T)

        div = lambda x, y: x / y if y else 0
        points = np.array([(
            key,
            div(positive_x.get(key, 0), total_x.get(key, 0)),
            div(positive_y.get(key, 0), total_y.get(key, 0)),
        ) for key in range(100)])

        plt.scatter(*points[:, (0, 1)].T)
        plt.ylim(0., 1.)
        plt.xlim(0, 100)
        plt.savefig(str(dest.joinpath('x.jpg')), dpi=200)

        plt.scatter(*points[:, (2, 0)].T)
        plt.xlim(0., 1.)
        plt.ylim(0, 100)
        plt.savefig(str(dest.joinpath('y.jpg')), dpi=200)

    if not args.eval_only:
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
            result.update({
                'mAP': float(np.mean(aps)),
                'Precision': float(np.mean(precisions)),
                'Recall': float(np.mean(recalls)),
                'GT': int(gt_counts),
                'PD': int(pd_counts),
            })
            json.dump(result, f)

    yield result
