from pathlib import Path

from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils import data
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.arguments import Arguments


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> nn.Module:

    model.load(torch.load(args.model, map_location=lambda s, l: s))

    # if device.type == 'cuda':
    #     model = nn.DataParallel(model)
    #     torch.backends.cudnn.benchmark = True

    model.extras.load_state_dict(torch.load('../ssd.pytorch/extras.pth'))
    model.loc.load_state_dict(torch.load('../ssd.pytorch/loc.pth'))
    model.conf.load_state_dict(torch.load('../ssd.pytorch/conf.pth'))
    model.to(device).train()

    return model


def train(model: nn.Module, loader: data.DataLoader, criterion, optimizer,
          device: torch.device = None, args: Arguments.parse.Namespace = None) \
        -> None:
    iterator = iter(loader)

    with tqdm(total=args.epoch) as tq:
        for iteration in range(args.start_epoch, args.epoch):
            try:
                images, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, targets = next(iterator)

            # TODO: DEBUG
            import pickle
            import numpy as np

            # np.save('images.npy', images.detach().numpy())
            # with open('targets.p', 'wb') as f:
            #     pickle.dump([target.detach().numpy() for target in targets], f)

            images = np.load('../FisheyeNet/images.npy')
            images = torch.from_numpy(images)
            targets = pickle.load(open('../FisheyeNet/targets.p', 'rb'))
            targets = [torch.from_numpy(target) for target in targets]

            images = Variable(images.to(device), volatile=True)
            targets = [Variable(target.to(device), volatile=True) for target in targets]

            print(images.mean())
            output = model(images)

            optimizer.zero_grad()

            loc_loss, conf_loss = criterion(output, targets)
            loss = loc_loss + conf_loss
            loss.backward()
            optimizer.step()

            if args.save_epoch and not (iteration % args.save_epoch):
                torch.save(model.state_dict(),
                           str(Path(args.dest).joinpath(f'{args.name}-{iteration:06}.pth')))

            tq.set_postfix(loss=loss.item())
            tq.update(1)
