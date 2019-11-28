from controller import Model
from view import Router, render, request
from utils import io
from utils import network

eval = Router('eval')


@eval.route('/', methods=['POST'])
def eval():
    response = { 'status': 'pending' }

    url = request.args.get('url', None)

    try:
        image = io.load(network.download(url))

        result = Model(image)
        response.update({
            'status': 'ok',

            'size': image.shape,
            'boxes': result['rois'],
            'scores': result['scores'].tolist(),
            'classes': result['class_ids'].tolist(),
        })

    except Exception as e:
        response.update({
            'status': 'failed',

            'message': str(e),
        })

    finally:
        return render(response)
