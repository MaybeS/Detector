from controller import Model
from view import Router, render, request


model = Router('model')


@model.route('/', methods=['GET'])
def index():
    return render(Model.s())


@model.route('/', methods=['POST'])
def load():
    response = {'status': 'pending'}

    try:
        Model.load(**{
            'num_classes': request.form.get('class', 2),
            'expire': request.form.get('expire', 600),
            'weight': request.form.get('weight', ''),
        })
        response.update({'status': 'ok'})

    except (FileNotFoundError, RuntimeError) as e:
        response.update({
            'status': 'failed',
            'message': str(e),
        })

    finally:
        return render(response)
