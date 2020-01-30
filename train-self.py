from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data

from data import Dataset
from models import DataParallel
from lib.evaluate import Evaluator
from lib.augmentation import Augmentation
from utils.arguments import Arguments


def arguments(parser):
    parser.add_argument('--unlabeled', required=False, type=str, default='',
                        help="unlabeled data")
    parser.add_argument('--unlabeled-gt', required=False, type=str, default='',
                        help="unlabeled data")

    parser.add_argument('--pseudo-step', required=False, type=int, default=10,
                        help="pseudo label using step")
    parser.add_argument('--pseudo-first-step', required=False, type=int, default=0,
                        help="pseudo label using step first time only")


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> nn.Module:

    if args.model != 'None' and args.model != '':
        model.load(torch.load(args.model, map_location=lambda s, l: s))
    model.train()

    if device.type == 'cuda':
        model = DataParallel(model)
        model.state_dict = model.module.state_dict
        torch.backends.cudnn.benchmark = True
    model.to(device)

    return model


def generate_pseudo(model: nn.Module, dataset: Dataset, transform: Augmentation,
                    device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> dict:
    result = {}

    evaluator = Evaluator(n_class=dataset.num_classes)

    ground = Path(args.unlabeled_gt)
    dest = Path(args.dest).joinpath('pseudo').joinpath(f"{kwargs['iteration']:08}")
    dest.mkdir(exist_ok=True, parents=True)

    model.eval()
    for index in tqdm(range(len(dataset))):
        name = dataset.pull_name(index)
        destination = dest.joinpath(f'{name}{dataset.DETECTION_EXT}')
        groundtruth = ground.joinpath(f'{name}{dataset.DETECTION_EXT}')

        image = dataset.pull_image(index)
        scale = torch.Tensor([
            image.shape[1], image.shape[0],
            image.shape[1], image.shape[0],
        ]).to(device)

        image = Variable(torch.from_numpy(transform(image)[0]).permute(2, 0, 1).unsqueeze(0)).to(device)

        detection = np.empty((0, 6), dtype=np.float32)
        detections, *_ = model(image).data

        for klass, boxes in enumerate(detections):
            candidates = boxes[boxes[:, 0] >= args.thresh]
            # filter out of image
            candidates = candidates[(
                torch.sum((candidates < -1) | (candidates > 2), axis=1) == 0
            ).nonzero().squeeze(0), :].reshape(-1, 5)

            # filter nan and inf
            candidates = candidates[(
                torch.sum(torch.isinf(candidates) | torch.isnan(candidates), axis=1) == 0
            ).nonzero().squeeze(0), :].reshape(-1, 5)

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

        if args.unlabeled_gt and groundtruth.exists():
            gt = pd.read_csv(str(groundtruth), header=None).values

            if detection.size and gt.size:
                evaluator.update((
                    detection[:, 0].astype(np.int),
                    detection[:, 1].astype(np.float32),
                    detection[:, 2:].astype(np.float32),
                    None,
                ), (
                    np.ones(np.size(gt, 0), dtype=np.int),
                    gt.astype(np.float32),
                    None,
                ))

        pd.DataFrame(detection).to_csv(str(destination), header=None, index=None)

    if args.unlabeled_gt:
        result.update({
            'acc': evaluator.mAP[1:].mean()
        })

    model.train()
    dataset.detections = list(sorted(dest.glob(f'*{dataset.DETECTION_EXT}')))

    return result


def train_self(model: nn.Module, dataset: Dataset, transform: Augmentation,
               criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.Optimizer,
               device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> None:
    loader = data.DataLoader(dataset, args.batch, num_workers=args.worker, drop_last=True,
                             shuffle=True, collate_fn=Dataset.collate, pin_memory=True)

    pseudo_dataset = Dataset.get(args.type)(args.unlabeled, transform=dataset.transform, eval_only=True)
    pseudo_loader = data.DataLoader(pseudo_dataset, args.batch, num_workers=args.worker, drop_last=True,
                                    shuffle=True, collate_fn=Dataset.collate, pin_memory=True)

    pseudo_step = args.pseudo_step
    pseudo_first_step = args.pseudo_first_step

    postfix = {}
    train_labeled, train_labeled_count = True, 0
    iterator, losses = iter(loader), list()

    with tqdm(total=args.epoch, initial=args.start_epoch) as tq:
        for iteration in range(args.start_epoch, args.epoch + 1):
            try:
                images, targets = next(iterator)

            except StopIteration:
                # generate pseudo label
                train_labeled_count += 1

                if train_labeled:
                    if pseudo_first_step:
                        if train_labeled_count > pseudo_first_step:
                            result = generate_pseudo(model, pseudo_loader.dataset, transform,
                                                     device, args, iteration=iteration)

                            pseudo_first_step = train_labeled_count = 0
                            postfix.update(result)
                            iterator = iter(pseudo_loader)
                        else:
                            iterator = iter(loader)
                    else:
                        if train_labeled_count > pseudo_step:
                            result = generate_pseudo(model, pseudo_loader.dataset, transform,
                                                     device, args, iteration=iteration)

                            train_labeled_count = 0
                            postfix.update(result)
                            iterator = iter(pseudo_loader)
                        else:
                            iterator = iter(loader)
                else:
                    iterator = iter(loader)

                train_labeled = not train_labeled

                images, targets = next(iterator)

                if loss is not None and scheduler is not None:
                    scheduler.step()

            images = Variable(images.to(device), requires_grad=False)
            targets = [Variable(target.to(device), requires_grad=False) for target in targets]

            output = model(images)
            optimizer.zero_grad()

            loc_loss, conf_loss = criterion(output, targets)
            loss = loc_loss + conf_loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if torch.isnan(loss):
                print(f'NaN detected in {iteration}')

            if args.save_epoch and not (iteration % args.save_epoch):
                torch.save(model.state_dict(),
                           str(Path(args.dest).joinpath(f'{args.name}-{iteration:06}.pth')))

            postfix['loss'] = loss.item()

            tq.set_postfix(**postfix)
            tq.update(1)
