import numpy as np
from constants import *


def mean(l):
    return sum(l)/len(l)


def median(l):
    return np.median(np.array(l))


def mode(l):
    return max(set(l), key=l.count)


def infer_nature(l):
    if not len(l):
        raise Exception(EMPTY_DATASET)
    if isinstance(l[0], str):
        return CATEGORICAL
