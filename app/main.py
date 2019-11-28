import json
from flask import Flask

from controller import Controller
from view import Router

app = Flask(__name__)
app.name = 'SSD-demo'

if __name__ == '__main__':
    with open('config.json') as f:
        app.config.update(json.load(f))

    Controller.init(app)
    Router.init(app)

    app.run(**app.config['RUN'])
