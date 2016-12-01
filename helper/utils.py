from __future__ import division
import random

random.seed(0)


def load_data(file_name):
    file = open(file_name)
    data = file.read()
    file.close()
    output = []
    i=0
    for line in data.split("\n"):
        tokens = line.split("\t")
        i += 1
        output.append({
            'attributes': map(lambda x: float(x), tokens[:-1]),
            'class': tokens[-1],
            'id': i
        })
    return output

def distance(a1, a2):
    a1, a2, l = list(a1['attributes']), list(a2['attributes']), len(a1['attributes'])
    p = 2
    return pow(reduce(lambda x, y: x+y, map(lambda i: pow(abs(a1[i]-a2[i]), p), xrange(l))), (1/p))


# def split(a, n):
#     k, m = len(a) / n, len(a) % n
#     return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))
