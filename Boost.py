from __future__ import division
from numpy.random import choice
from random import choice as simple_choice
from KNearestNeighbour import *
from NaiveBayes import *
from RandomForest import *
from DecisionTree import *

__all__ = ['Boost']


class Boost(object):

    def __init__(self, N=5, sample_factor=0.5, **kwargs):
        self.allmodels = [KNN, NaiveBayes, RandomForest, DecisionTree]
        self.models = [DecisionTree for _ in range(len(self.allmodels))]
        self.models = map(lambda m: m(), self.models)
        self.sample_factor = sample_factor

    def __bootstrap__(self, data):
        length = len(data)
        for t in data:
            t['weight'] = 1 / length
        return data

    def fit(self, train):
        num_samples = int(self.sample_factor*len(train))
        train = self.__bootstrap__(train)
        bootstrap_train = choice(train, num_samples, map(lambda t: t['weight'], train))
        print bootstrap_train

    def transform(self, test):
        pass

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)
