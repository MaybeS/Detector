import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn


def init(model, device, path: str = None):

    if device.type == 'cuda':
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    if path is not None:
        model.load(torch.load(path, map_location=lambda s, l: s))

    return model.to(device).train()


def train(model, loader, criterion, optimizer, num_epochs, start_epoch:int = 0, device=None):
    iterator = iter(loader)

    for iteration in range(start_epoch, num_epochs):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)

        images = Variable(images.to(device), requires_grad=False)
        targets = [Variable(target, requires_grad=False) for target in targets]

        output = model(images)

        optimizer.zero_grad()
        loc_loss, conf_loss = criterion(output, targets)
        loss = loc_loss + conf_loss
        loss.backward()
        optimizer.step()

    return model, None


if __name__ == '__main__':
    train()
