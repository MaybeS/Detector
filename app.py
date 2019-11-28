import json
from flask import Flask

from controller import Controller
from view import Router


def create():
    app = Flask(__name__)
    with open('config.json') as f:
        app.config.update(json.load(f))

    Controller.init(app);
    Router.init(app);

    return app


if __name__ == '__main__':
    app = create()
    app.run(**app.config['RUN'])
