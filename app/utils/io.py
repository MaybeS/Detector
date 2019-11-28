from pathlib import Path
from functools import reduce

import skimage.io
import skimage.color
import skimage.transform
import numpy as np


def load(data):
    image = skimage.io.imread(data)
    h, w, *c = image.shape

    if not c:
        image = np.stack((image, ) * 3, axis=-1)
    elif c == [4]:
        image = skimage.color.rgba2rgb(image)

    return image


def join(path, *args):
    return str(reduce(lambda p, a: p.joinpath(a), [Path(path), *args]))


def iterdir(directory):
    path = Path(directory)

    if not path.exists():
        path.mkdir()
    
    yield from path.iterdir()


def transform(image, mean=(104, 117, 123)):
    image = (skimage.transform.resize(image, (300, 300)) * 255).astype(np.float32)
    image -= np.array(mean, dtype=np.float32)
    return image.astype(np.float32)
