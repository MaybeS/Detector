import torch

from models import Model
from data.dataset import Dataset
from lib.augmentation import Augmentation
from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments
from utils.config import Config


def main(args: Arguments.parse.Namespace):
    MODEL = Model.get(args.network).get(args.backbone)

    config = Config(args.config, args.network, MODEL)
    config.sync(vars(args))
    Executable.log('Config', config.dump)

    executor = Executable.s[args.command]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Dataset
    transform = Augmentation.get(args.type)(**config.dump)
    dataset = Dataset.get(args.type)(args.dataset,
                                     transform=transform,
                                     train=args.command == 'train',
                                     eval_only=args.eval_only)

    # Create Model
    num_classes = args.classes or dataset.num_classes
    model = MODEL.new(num_classes, args.batch, config=config)
    Executable.log('Model', model)

    # Initialize Model
    model = executor.init(model, device, args)

    # Set optimizer, scheduler and criterion
    optim, optim_args = model.OPTIMIZER
    optimizer = optim(model.parameters(), **(optim_args.update({
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.decay,
    }) or optim_args))

    sched, sched_args = model.SCHEDULER
    scheduler = sched(optimizer, **(sched_args.update({
    }) or sched_args))

    criterion = model.loss(num_classes, device=device)

    # Run main script
    executor(model, dataset=dataset,
             criterion=criterion, optimizer=optimizer, scheduler=scheduler,  # train args
             transform=Augmentation.get('base')(**config.dump),  # test args
             device=device, args=args)

    Executable.close()


if __name__ == '__main__':
    arguments = Arguments()

    seed(arguments.seed)
    main(arguments)
