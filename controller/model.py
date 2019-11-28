import numpy as np
import torch
from torch.autograd import Variable

from utils import io
from utils.lazy import lazy
from utils.moduletools import import_module


class Model:
    @classmethod
    def init(cls, app):
        cls.config = app.config
        cls.num_classes = 2
        cls.batch_size = 1

        cls.device = torch.device('cpu')
        cls.model_name = cls.config['PATH']['model_name']
        cls.model_constructor = import_module(**{
            "path": cls.config['PATH']['module'],
            "file": cls.config['PATH']['model_file'],
            "name": cls.model_name,
        })

    def __new__(cls, image):
        if not getattr(cls, 'model', None):
            raise Exception("Load model first")

        h, w, *c = image.shape
        scale = np.array([w, h, w, h])
        image = io.transform(image)
        image = Variable(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)).to(cls.device)

        detections, *_ = cls.model(image).data

        classes = np.empty((0), dtype=np.int)
        scores = np.empty((0), dtype=np.float32)
        boxes = np.empty((0, 4), dtype=np.float32)

        for klass, candidates in enumerate(detections):
            candidates = candidates[candidates[:, 0] >= .3]
            
            if candidates.size(0) == 0:
                continue

            candidates = candidates.cpu().detach().numpy()

            classes = np.concatenate((
                classes,
                np.full(np.size(candidates, 0), klass, dtype=np.uint8),
            ))
            scores = np.concatenate((
                scores,
                candidates[:, 0],
            ))
            boxes = np.concatenate((
                boxes,
                candidates[:, 1:] * scale,
            ))
        
        print(classes.shape, scores.shape, boxes.shape)
        return {
            "classes": classes,
            "scores": scores,
            "boxes": boxes,
        }



    @classmethod
    @lazy
    def s(cls):
        return list(map(lambda p: p.name, io.iterdir(cls.config['PATH']['weights'])))

    @classmethod
    def load(cls, weight, num_classes=2, expire=600):
        cls.num_classes = num_classes
        cls.model = getattr(cls.model_constructor, cls.model_name)(cls.num_classes, cls.batch_size)

        if weight not in cls.s():
            raise FileNotFoundError
        
        cls.model.load(torch.load(io.join(cls.config['PATH']['weights'], weight), map_location=lambda s, l: s))
        cls.model.eval()
        cls.model.to(cls.device)
