import numpy as np


def mean(l):
    return sum(l)/len(l)


def median(l):
    return np.median(np.array(l))


def mode(l):
    return max(set(l), key=l.count)
