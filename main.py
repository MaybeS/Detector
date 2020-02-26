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

    num_classes = args.classes or dataset.num_classes

    Executable.log('Config', config.data)
    model = Model.get(f'SSD_{args.backbone}').new(num_classes, args.batch, config=config.data,
                                                  warping=args.warping, warping_mode=args.warping_mode)
    Executable.log('Model', model)

    model = executor.init(model, device, args)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.decay)

    if model.SCHEDULER is not None:
        method, arguments = model.SCHEDULER
        scheduler = method(optimizer, **arguments)

    else:
        scheduler = None

    criterion = model.loss(num_classes, device=device)

    executor(model, dataset=dataset,
             criterion=criterion, optimizer=optimizer, scheduler=scheduler,     # train args
             transform=Augmentation.get('base')(**config.data),                 # test args
             device=device, args=args)

    Executable.close()


if __name__ == '__main__':
    arguments = Arguments()
    config = Config(arguments.config, arguments)
    config.sync(vars(arguments))

    seed(arguments.seed)
    main(arguments, config)
