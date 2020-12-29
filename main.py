import torch

from models import Model
from data.dataset import Dataset
from lib.augmentation import Augmentation
from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments
from utils.config import Config


def main(args: Arguments.parse.Namespace):
    model_class = Model.get(args.network).get(args.backbone)

    config = Config(args.config, args.network, model_class)
    config.sync(vars(args))
    Executable.log('Config', config.dump)

    executor = Executable.s[args.command]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command != 'train':
        args.batch = 1

    # Create Dataset
    augmentation = args.augment or args.type
    transform = Augmentation.get(augmentation)(**config.dump).train(args.command == 'train')
    dataset = Dataset.get(args.type)(args.dataset,
                                     train=args.command == 'train',
                                     transform=transform,
                                     eval_only=args.eval_only)

    # Create Model
    num_classes = args.classes or dataset.num_classes
    model = model_class.new(num_classes, args.batch, config=config)
    Executable.log('Model', model)

    # Initialize Model
    model = executor.init(model, device, args)

    # Set optimizer, scheduler and criterion
    optim, optim_args = model.OPTIMIZER
    optimizer = optim(model.parameters(), **optim_args)

    sched, sched_args = model.SCHEDULER
    scheduler = sched(optimizer, **sched_args)

    criterion = model.loss(num_classes, device=device)

    # Run main script
    executor(model, dataset=dataset,
             # train arguments
             criterion=criterion, optimizer=optimizer, scheduler=scheduler,
             # test arguments
             transform=transform,
             # etc arguments
             device=device, args=args)

    Executable.close()


if __name__ == '__main__':
    arguments = Arguments()

    seed(arguments.seed)
    main(arguments)
