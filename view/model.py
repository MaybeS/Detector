from controller import Model
from view import Router, render, request


model = Router('model')


@model.route('/', methods=['GET'])
def index():
    s = Model.s()
    print(s)
    return render(s)


@model.route('/', methods=['POST'])
def load():
    return render(Model.load(**{
        'num_classes': request.args.get('class', 1),
        'expire': request.args.get('expire', 600),
        'weight': request.args.get('weight', ''),
    }))
