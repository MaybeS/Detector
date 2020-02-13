import __main__
import logging
from pathlib import Path
from pprint import pformat


class Executable:
    _ = Path(__main__.__file__)
    s = dict()
    logger = logging.getLogger('Detector')

    def __init__(self, file: str):
        self.command = file
        self.module = __import__(file)
        self.name = self.module.__name__.replace('-', '_')

    def __getattr__(self, key):
        if hasattr(self.module, key):
            return getattr(self.module, key)
        elif hasattr(super(Executable, self), key):
            return super(Executable, self).__getattribute__(self, key)
        else:
            return lambda *args: None

    def init(self, model, device, args):
        self.log('Model initializing ...')
        self.log('Arguments', vars(args))
        results = getattr(self.module, 'init')(model, device, args)

        self.log('Model initialized')
        return results

    def __call__(self, *args, **kwargs):
        self.log(f'Model execute {self.command}')
        for dump in getattr(self.module, self.name)(*args, **kwargs):
            self.log('TRAIN', dump)

        self.log(f'Model execution done!')

    @classmethod
    def log(cls, prefix: str, target: str = None, level: int = logging.INFO):
        if target is None:
            cls.logger.log(level, prefix)

        else:
            cls.logger.log(level, f' {prefix} '.center(80, '='))

            if isinstance(target, dict):
                for key, value in target.items():
                    cls.logger.log(level, f'{pformat(key).ljust(16)}: {pformat(value)}')
            else:
                cls.logger.log(level, pformat(target))

    @classmethod
    def close(cls):
        for handler in cls.logger.handlers:
            handler.close()
            cls.logger.removeFilter(handler)

    @staticmethod
    def ismain():
        return Executable._.stem == 'main'


for executor in map(lambda x: x.stem, filter(lambda x: x.name != Executable._.name, Path('.').glob('*.py'))):
    Executable.s[executor] = Executable(executor)
