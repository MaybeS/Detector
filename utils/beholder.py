from collections import defaultdict


class Beholder(type):
    __inheritors__ = defaultdict(dict)

    def __new__(mcs, name, bases, dct):
        klass = type.__new__(mcs, name, bases, dct)

        for base in klass.mro()[1:-1]:
            mcs.__inheritors__[base][klass.__name__.lower()] = klass

        return klass

    @property
    def __modules__(cls):
        return cls.__inheritors__[cls]

    def get(cls, name):
        return cls.__modules__.get(name.lower(), None)
