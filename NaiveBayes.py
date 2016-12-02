from __future__ import division
from helper.constants import *
from helper.statistics import std, infer_nature
from collections import Counter


class NaiveBayes(object):

    def __init__(self):
        self.length = None
        self.shape = None
        self.model = None
        self.labels = None

    def mapper(self, l):
        if infer_nature(l) == CATEGORICAL:
            return l
        _min_ = min(l)
        _max_ = max(l)
        _std_ = std(l)
        i = 0
        while True:
            low = round(_min_ + i * _std_, 3)
            high = round(_min_ + (i + 1) * _std_, 3)
            interval = low, high
            l = map(lambda e: interval if low <= e < high else e, l)
            i += 1
            if high > _max_:
                break
        return l

    def fit(self, train):
        X = map(lambda t: t['attributes'], train)
        y = map(lambda t: t['class'], train)
        self.shape = len(y), len(X[0])
        prior = Counter(y)
        prior = {key: prior[key]/self.shape[0] for key in prior.keys()}
        labels = set(y)
        X = map(lambda i: self.mapper(map(lambda x: x[i], X)), xrange(self.shape[1]))
        posterior = dict()
        for label in labels:
            indices = filter(lambda a: a is not None, map(lambda (i, v): i if v == label else None, enumerate(y)))
            total = len(indices)
            for i in xrange(self.shape[1]):
                x = map(lambda (i, v): v, filter(lambda (j, e): j in indices, enumerate(X[i])))
                counts = Counter(x)
                posterior[label] = {key: (1+counts[key])/total for key in counts.keys()}
        self.model = {'labels': labels, 'posterior': posterior, 'prior': prior}

    def transform(self, test):
        model = self.model
        if model is None:
            raise Exception(MODEL_NOT_TRAINED_ERROR)

        for t in test:
            x = t['attributes']
            expectation = dict()
            for label in model['labels']:
                for i, e in enumerate(x):
                    probability = filter(lambda (k, v): k[0] <= e < k[1], model['posterior'][label].items())
                    if len(probability) == 1:
                        probability = probability[0][1]
                    else:
                        probability = 1/(model['prior'][label]*self.shape[0])
                    expectation[label] = probability * model['prior'][label]
            t['assigned'] = sorted(expectation, key=expectation.get, reverse=True)[0]
        return test

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)
