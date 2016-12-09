import numpy as np
import math
from constants import *


def mean(l):
    return np.mean(np.array(l))


def median(l):
    return np.median(np.array(l))


def mode(l):
    return max(set(l), key=l.count)


def std(l):
    return np.std(np.array(l))


def infer_nature(l):
    if not len(l):
        raise Exception(EMPTY_DATASET)
    if isinstance(l[0], str):
        return CATEGORICAL


def entropy(l):
    return sum(map(lambda v: 0 if v == 0 else -v * math.log(v, 2), l))
