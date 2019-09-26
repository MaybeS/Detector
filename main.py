import torch
import torch.optim as optim
from torch.utils import data

from ssd.model import SSD300
from data.dataset import Dataset
from lib.augmentation import Augmentation
from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments


def main(args: Arguments.parse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset.get(args.type)(args.dataset, transform=Augmentation.get(args.type)())

    model = SSD300(dataset.num_classes)

    executor = Executable(args.command)
    model = executor.init(model, device, path=args.model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = SSD300.LOSS(dataset.num_classes, device=device)

    loader = data.DataLoader(dataset, args.batch, num_workers=args.worker,
                             shuffle=True, collate_fn=Dataset.collate, pin_memory=True)

    model, acc = executor(
        model, loader, criterion, optimizer, device=device, num_epochs=args.epoch)

    if executor.command == 'train':
        torch.save(model.state_dict(), args.output)
    elif executor.command == 'eval':
        print(acc)


if __name__ == '__main__':
    arguments = Arguments()
    seed(arguments.seed)
    main(arguments)
