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
    dataset = Dataset.get(args.type)(args.dataset,
                                     transform=Augmentation.get(args.type)(**config.data),
                                     eval_only=args.eval_only)

    if executor.name != 'train':
        args.batch = 1

    model = Model.get('SSD').new(dataset.num_classes, args.batch,
                                 warping=args.warping, warping_mode=args.warping_mode,
                                 config=config.data)

    model = executor.init(model, device, args)

    Path(args.dest).mkdir(exist_ok=True, parents=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=3)

    criterion = model.loss(dataset.num_classes, device=device)

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
