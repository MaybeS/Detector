from pathlib import Path

import torch
import torch.optim as optim

from models import Model
from data.dataset import Dataset
from lib.augmentation import Augmentation
from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments
from utils.config import Config


def main(args: Arguments.parse.Namespace, config: Config):
    executor = Executable.s[args.command]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Augmentation.get(args.type)(**config.data)

    dataset = Dataset.get(args.type)(args.dataset,
                                     transform=transform,
                                     train=args.command == 'train',
                                     eval_only=args.eval_only)

    if executor.name != 'train':
        args.batch = 1

    num_classes = args.classes or dataset.num_classes

    model = Model.get('SSD').new(num_classes, args.batch,
                                 base=args.backbone, config=config.data,
                                 warping=args.warping, warping_mode=args.warping_mode)

    model = executor.init(model, device, args)

    Path(args.dest).mkdir(exist_ok=True, parents=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=3)
    scheduler = None

    criterion = model.loss(num_classes, device=device)

    executor(model, dataset=dataset,
             criterion=criterion, optimizer=optimizer, scheduler=scheduler,     # train args
             transform=Augmentation.get('base')(**config.data),                 # test args
             device=device, args=args)


if __name__ == '__main__':
    arguments = Arguments()
    config = Config(arguments.config)
    config.sync(vars(arguments))

    seed(arguments.seed)
    main(arguments, config)
