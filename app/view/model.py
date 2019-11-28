from controller import Model
from view import Router, render, request


model = Router('model')


@model.route('/', methods=['GET'])
def index():
    return render({
        'model': str(Model.model),
        'config': {},
        'weight': Model.weight,
        'weights': Model.s(),
    })


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


@model.route('/', methods=['DELETE'])
def release():
    response = {'status': 'pending'}

    try:
        Model.release()
        response.update({'status': 'ok'})

    except Exception as e:
        response.update({
            'status': 'failed',
            'message': str(e),
        })

    finally:
        return render(response)
