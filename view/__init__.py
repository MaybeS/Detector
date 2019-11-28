from os import path

from flask import Blueprint, jsonify, request

from utils.moduletools import import_subclass


class Router(Blueprint):
    s = []
    status = {
        'status': 'ok',
    }

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **{
            'url_prefix': path.join('/', name),
            'template_folder': path.join('../', 'templates', name),
            'import_name': name,
            **kwargs
        })
        Router.s.append(self)

    def __str__(self) -> str:
        return self.name

    @classmethod
    def init(cls, app):
        any(map(app.register_blueprint, cls.s))
        app.route('/')(lambda *_: cls.status)


def render(target):
    if isinstance(target, dict) or isinstance(target, list):
        return jsonify(target)


from view.model import model as _model_
from view.eval import eval as _eval_
