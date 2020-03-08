from sys import stdout
import argparse
import logging
from pathlib import Path

from .executable import Executable


class Arguments:
    parse = argparse
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector')

    @classmethod
    def add_argument(cls, *args, **kwargs):
        cls.parser.add_argument(*args, **kwargs)

    def __new__(cls):
        # auto executable command
        executables = tuple(Executable.s)
        if len(executables) and Executable.ismain():
            cls.parser.add_argument("command", metavar="<command>",
                                    choices=executables,
                                    help=f'Choice from {", ".join(executables)}')

        for executor in executables:
            Executable.s[executor].arguments(cls.parser)

        cls.parser.add_argument('--name', required=False, default='SSD300', type=str,
                                help="Name of model")

        cls.parser.add_argument('-s', '--seed', required=False, default=42,
                                help="The answer to life the universe and everything")
        cls.parser.add_argument('--worker', required=False, default=4, type=int,
                                help="worker")
        cls.parser.add_argument('--log-level', required=False, default='WARNING', type=str,
                                choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                                help="Set logging level")

        cls.parser.add_argument('--network', required=False, type=str, default='SSD',
                                help="Network")
        cls.parser.add_argument('--backbone', required=False, type=str, default='VGG16',
                                help="Backbone of model")
        cls.parser.add_argument('-t', '--type', required=False, type=str, default='DETECTION',
                                help="Dataset type")
        cls.parser.add_argument('-D', '--dataset', required=False, type=str, default='',
                                help="Path to dataset")
        cls.parser.add_argument('-d', '--dest', required=False, default='./weights', type=str,
                                help="Path to output")
        cls.parser.add_argument('--config', required=False, default=None, type=str,
                                help="Path to config file")
        cls.parser.add_argument('--classes', required=False, default=0, type=int,
                                help="Number of class")

        cls.parser.add_argument('--model', required=False, default='weights/vgg16-reducedfc.pth', type=str,
                                help="Path to model")
        cls.parser.add_argument('--thresh', required=False, default=.3, type=float,
                                help="threshold")
        cls.parser.add_argument('--augment', required=False, default=None, type=str,
                                help="Custom augmentation")

        args = cls.parser.parse_args()

        Path(args.dest).mkdir(exist_ok=True, parents=True)

        # Init logger
        logger_std = logging.StreamHandler(stdout)
        logger_std.setFormatter(logging.Formatter(fmt="%(name)s %(levelname)-8s: %(message)s", datefmt='%H:%M:%S'))
        logger_std.setLevel(getattr(logging, args.log_level))

        logger_file = logging.FileHandler(str(Path(args.dest).joinpath(f'events.log')))
        logger_file.setFormatter(logging.Formatter(fmt="%(name)s %(levelname)-8s: %(message)s", datefmt='%H:%M:%S'))
        logger_file.setLevel(logging.INFO)

        Executable.logger.setLevel(logging.DEBUG)
        Executable.logger.addHandler(logger_std)
        Executable.logger.addHandler(logger_file)

        return args
