import argparse

from .executable import Executable


class Arguments:
    parse = argparse

    def __new__(cls):
        parser = cls.parse.ArgumentParser(
            description='Description')

        # auto executable command
        executables = tuple(Executable.s)
        if len(executables) and Executable.ismain():
            parser.add_argument("command",
                                metavar="<command>",
                                choices=executables,
                                help=f'Choice from {", ".join(executables)}')

        parser.add_argument('-s', '--seed', required=False, default=42,
                            help="The answer to life the universe and everything")

        parser.add_argument('-d', '--dataset', required=True, type=str,
                            help="Path to dataset")
        parser.add_argument('-o', '--output', required=False, default='model.pth', type=str,
                            help="Path to output")

        parser.add_argument('--model', required=False, type=str,
                            help="Path to model")
        parser.add_argument('--batch', required=False, default=32, type=int,
                            help="batch")
        parser.add_argument('--epoch', required=False, default=10000, type=int,
                            help="epoch")

        return parser.parse_args()
