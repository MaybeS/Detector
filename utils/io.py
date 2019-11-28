from pathlib import Path

import skimage
import numpy as np


def load(data):
    image = skimage.io.imread(data)
    h, w, *c = image.shape

    if not c:
        image = np.stack((image, ) * 3, axis=-1)
    elif c == [4]:
        image = skimage.color.rgba2rgb(image)

    return image


def iterdir(directory):
    path = Path(directory)

    if not path.exists():
        path.mkdir()
    
    yield from path.iterdir()
