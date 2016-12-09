from DecisionTree import *
from helper.utils import most_common
from random import sample, choice
from copy import deepcopy

__all__ = ['RandomForest']


class RandomForest(object):

    def __init__(self, T=10, M=30, bagging=False):
        self.t = T
        self.m = M
        self.bagging = bagging
        self.forest = map(lambda i: DecisionTree(), range(T))
        self.shape = None
        self.selected_attributes = list()

    def __mapper__(self, data, i):
        data = deepcopy(data)
        length = len(data)
        for t in data:
            t['attributes'] = map(t['attributes'].__getitem__, self.selected_attributes[i])
        if self.bagging:
            return [choice(data) for _ in range(length)]
        return data

    def fit(self, train):
        self.shape = len(train), len(train[0]['attributes'])
        self.selected_attributes = map(lambda i: sample(range(self.shape[1]), self.m), range(self.t))
        map(lambda (i, tree): tree.fit(self.__mapper__(train, i)), enumerate(self.forest))

    def transform(self, test):
        results = map(lambda (i, tree): map(lambda r: r['assigned'], tree.transform(self.__mapper__(test, i))), enumerate(self.forest))
        results = map(lambda i: most_common(map(lambda r: r[i], results)), range(len(results[0])))
        for i, t in enumerate(test):
            t['assigned'] = results[i]
        return test

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)
