import random

import numpy as np
import torch


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
