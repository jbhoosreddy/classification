from __future__ import division
from helper.constants import *
from helper.utils import categorize
from helper.statistics import std, infer_nature
from collections import Counter

__all__ = ['NaiveBayes']


class NaiveBayes(object):

    def __init__(self):
        self.length = None
        self.shape = None
        self.model = None
        self.labels = None

    def fit(self, train):
        X = map(lambda t: t['attributes'], train)
        y = map(lambda t: t['class'], train)
        self.shape = len(y), len(X[0])
        prior = Counter(y)
        prior = {key: prior[key]/self.shape[0] for key in prior.keys()}
        labels = set(y)
        X = map(lambda i: categorize(map(lambda x: x[i], X)), range(self.shape[1]))
        posterior = dict()
        for label in labels:
            indices = filter(lambda a: a is not None, map(lambda (i, v): i if v == label else None, enumerate(y)))
            total = len(indices)
            posterior[label] = list()
            for i in range(self.shape[1]):
                x = map(lambda (k, v): v, filter(lambda (j, e): j in indices, enumerate(X[i])))
                counts = Counter(x)
                posterior[label].append({key: (1+counts[key])/total for key in counts.keys()})
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
                    probability = filter(lambda (k, v): k[0] <= e < k[1] if isinstance(k, tuple) else k == e, model['posterior'][label][i].items())
                    if len(probability) == 1:
                        probability = probability[0][1]
                    else:
                        probability = 1/(model['prior'][label]*self.shape[0])
                    if label not in expectation.keys():
                        expectation[label] = 1
                    expectation[label] = expectation[label] * probability
                    expectation[label] = expectation[label] * model['prior'][label]
            t['assigned'] = sorted(expectation, key=expectation.get, reverse=True)[0]
        return test

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)
