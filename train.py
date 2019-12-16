from pathlib import Path

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data

from data import Dataset
from models import DataParallel
from utils.arguments import Arguments


def arguments(parser):
    parser.add_argument('--batch', required=False, default=32, type=int,
                        help="batch")
    parser.add_argument('--lr', required=False, default=.001, type=float,
                        help="learning rate")
    parser.add_argument('--momentum', required=False, default=.9, type=float,
                        help="momentum")
    parser.add_argument('--decay', required=False, default=5e-4, type=float,
                        help="weight decay")
    parser.add_argument('--epoch', required=False, default=10000, type=int,
                        help="epoch")
    parser.add_argument('--start-epoch', required=False, default=0, type=int,
                        help="epoch start")
    parser.add_argument('--save-epoch', required=False, default=500, type=int,
                        help="epoch for save")
    parser.add_argument('--worker', required=False, default=4, type=int,
                        help="worker")

    # season 1 warping
    parser.add_argument('--warping', required=False, type=str, default='none',
                        choices=["none", "head", "all", "first"],
                        help="Warping layer apply")
    parser.add_argument('--warping-mode', required=False, type=str, default='sum',
                        choices=['sum', 'average', 'concat'])

    # season 2 pseudo label
    parser.add_argument('--pseudo', required=False, type=float, default=.0,
                        help="pseudo label using ratio")


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


def train(model: nn.Module, dataset: Dataset,
          criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.Optimizer,
          device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> None:
    loader = data.DataLoader(dataset, args.batch, num_workers=args.worker,
                             shuffle=True, collate_fn=Dataset.collate, pin_memory=True)
    iterator, losses = iter(loader), list()

    with tqdm(total=args.epoch, initial=args.start_epoch) as tq:
        for iteration in range(args.start_epoch, args.epoch):

            try:
                images, targets = next(iterator)

            except StopIteration:
                iterator = iter(loader)
                images, targets = next(iterator)

                if loss is not None and scheduler is not None:
                    scheduler.step(sum(losses) / len(losses))

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

            tq.set_postfix(loss=loss.item())
            tq.update(1)
