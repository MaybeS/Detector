from controller import Model
from view import Router, render, request
from utils import io
from utils import network

eval = Router('eval')


@eval.route('/', methods=['POST'])
def eval():
    response = { 'status': 'pending' }
    url = request.form.get('url', '')

    try:
        image = io.load(url)

        response.update({ k: v.tolist() for k, v in Model(image).items() })
        response.update({
            'status': 'ok',

            'size': image.shape
        })

    except Exception as e:
        response.update({
            'status': 'failed',

            'message': str(e),
        })

    finally:
        return render(response)
