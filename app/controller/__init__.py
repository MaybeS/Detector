from controller.model import Model


class Controller:
    @staticmethod
    def init(app):
        Model.init(app)
