import json
from flask import Flask

from controller import Controller
from view import Router

app = Flask(__name__)
app.name = 'SSD-demo'

with open('config.json') as f:
    app.config.update(json.load(f))

Controller.init(app)
Router.init(app)

if __name__ == '__main__':
    app.run(**app.config['RUN'])
