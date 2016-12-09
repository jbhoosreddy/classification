from __future__ import division
from statistics import infer_nature, std
from constants import *
import random

__all__ = ['print_proxy', 'print_list', 'print_dict', 'load_data', 'distance', 'split', 'is_type', 'categorize',
           'most_common', 'normalize']

random.seed(0)


def print_proxy(l):
    print l


def print_list(l, c=None, should_print=True):
    output = ""
    for i in l:
        if should_print:
            print i
        output += str(i)+"\n"
        if c:
            c -= 1
            if not c:
                break
    return output


def print_dict(d, c=None, should_print=True):
    output = ""
    for k,v in d.items():
        if should_print:
            print k,v
        output += str(k)+": "+str(v)+"\n"
        if c:
            c -= 1
            if not c:
                break
    return output


def load_data(file_name, map_to_int=False):

    def mapper(x):
        if not map_to_int:
            return x
        if x not in array:
            array.append(x)
        return array.index(x)

    file = open(file_name)
    data = file.read()
    file.close()
    output = []
    i = 0
    array = list()
    for line in data.split("\n"):
        tokens = line.split("\t")
        i += 1
        output.append({
            'attributes': map(lambda x: float(x) if is_type(float, x) else mapper(x), tokens[:-1]),
            'class': tokens[-1],
            'id': i
        })
    return output


def distance(a1, a2):
    a1, a2, l = list(a1['attributes']), list(a2['attributes']), len(a1['attributes'])
    p = 2
    return pow(reduce(lambda x, y: x+y, map(lambda i: pow(abs(a1[i]-a2[i]), p), range(l))), (1/p))


def split(a, n):
    k, m = int(len(a) / n), len(a) % n
    r = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return r


def is_type(type, s):
    try:
        type(s)
        return True
    except ValueError:
        return False


def categorize(l, expansion_factor=1, min_threshold=0.01):
    if infer_nature(l) == CATEGORICAL:
        return l
    _min_ = min(l)
    _max_ = max(l)
    _std_ = std(l)
    i = 0
    while True:
        if i == 0:
            low = _min_
        else:
            low = round(_min_ + i * expansion_factor * _std_, 3)
        if _std_ == 0:
            high = low + min_threshold
        else:
            high = round(_min_ + (i + 1) * expansion_factor * _std_, 3)
        interval = low, high
        l = map(lambda e: interval if low <= e < high else e, l)
        i += 1
        if high > _max_:
            break
    return l


def most_common(lst):
    return max(set(lst), key=lst.count)


def normalize(lst, m=None):
    m = max(lst) if m is None else m
    return map(lambda l: l/m, lst), m
