import argparse

from .executable import Executable


class Arguments:
    parse = argparse

    def __new__(cls):
        parser = cls.parse.ArgumentParser(
            description='Single Shot MultiBox Detector')

        # auto executable command
        executables = tuple(Executable.s)
        if len(executables) and Executable.ismain():
            parser.add_argument("command",
                                metavar="<command>",
                                choices=executables,
                                help=f'Choice from {", ".join(executables)}')

        parser.add_argument('--name', required=False, default='SSD300', type=str,
                            help="Name of model")

        parser.add_argument('-s', '--seed', required=False, default=42,
                            help="The answer to life the universe and everything")

        parser.add_argument('-t', '--type', required=False, type=str, default='COCO',
                            help="Dataset type")
        parser.add_argument('-D', '--dataset', required=True, type=str,
                            help="Path to dataset")
        parser.add_argument('-d', '--dest', required=False, default='./weights', type=str,
                            help="Path to output")

        parser.add_argument('--model', required=False, default='weights/vgg16_reducedfc.pth', type=str,
                            help="Path to model")
        parser.add_argument('--thresh', required=False, default=.6, type=float,
                            help="threshold")
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
        parser.add_argument('--save-epoch', required=False, default=250, type=int,
                            help="epoch for save")
        parser.add_argument('--worker', required=False, default=1, type=int,
                            help="worker")

        return parser.parse_args()
