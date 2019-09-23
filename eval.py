import copy

import torch
from torch.utils import data


def init(model, device, path, *args, **kwargs):
    model = model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def eval(model, loader: data.DataLoader, *args, device=None, **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corrects = 0

    # Iterate over data.
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        corrects += torch.sum(preds == labels.data)

    epoch_acc = corrects.double() / len(loader.dataset)

    return model, epoch_acc
