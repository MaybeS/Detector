from utils import io
from utils.lazy import lazy


class Model:
    @classmethod
    def init(cls, app):
        cls.config = app.config

    def __new__(cls, image):
        pass

    @classmethod
    @lazy
    def s(cls):
        return list(io.iterdir(cls.config['PATH']['weights']))

    @classmethod
    def load(cls, file):
        pass
