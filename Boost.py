from __future__ import division
from numpy.random import choice
from random import choice as simple_choice
from helper.utils import most_common
from math import log, exp
from KNearestNeighbour import *
from NaiveBayes import *
from RandomForest import *
from DecisionTree import *

__all__ = ['Boost']


class Boost(object):

    def __init__(self, N=5, sample_factor=0.9, **kwargs):
        self.models = list()
        self.alphas = list()
        self.selected_model = DecisionTree
        self.models = list()
        self.sample_factor = sample_factor
        self.N = N

    def __bootstrap__(self, data):
        length = len(data)
        for t in data:
            t['weight'] = 1 / length
        return data

    def __update_weights__(self, data):
        def get_alpha(e):
            if e == 0:
                return 1
            return 0.5 * log((1-e)/e)

        def dot(a, b):
            if a == b == '1':
                return 1
            return 0

        def delta(a, p):
            if a == p:
                return 0
            else:
                return 1
        weighted_error_sum = sum(map(lambda d: d['weight'] * delta(d['class'], d['assigned']), data))
        weight_sum = sum(map(lambda d: d['weight'], data))
        error = weighted_error_sum / weight_sum
        self.alphas.append(get_alpha(error))
        for d in data:
            d['weight'] *= exp(-1 * (self.alphas[-1] * dot(d['assigned'], d['class'])))
        weight_sum = sum(map(lambda d: d['weight'], data))
        for d in data:
            d['weight'] /= weight_sum

    def fit(self, train):
        num_samples = int(self.sample_factor*len(train))
        train = self.__bootstrap__(train)
        for i in range(self.N):
            self.models.append(self.selected_model())
            bootstrap_train = choice(train, num_samples, map(lambda t: t['weight'], train))
            self.__fit__(bootstrap_train)
            train = self.__transform__(train)
            self.__update_weights__(train)

    def __fit__(self, data):
        self.models[-1].fit(data)

    def __transform__(self, data):
        return self.models[-1].transform(data)

    def transform(self, test):
        results = map(lambda m: map(lambda r: r['assigned'], m.transform(test)), self.models)
        results = map(lambda i: most_common(map(lambda r: r[i], results)), range(len(results[0])))
        for i, t in enumerate(test):
            t['assigned'] = results[i]
        return test

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)
