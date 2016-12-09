from helper.utils import distance, normalize
from helper.constants import *
import heapq

__all__ = ['KNN']


class KNN(object):

    def __init__(self, K):
        self.model = None
        self.K = K
        self.normalization_factor = None

    def __mapper__(self, data):
        X = map(lambda d: d['attributes'], data)
        if self.normalization_factor is None:
            X = map(lambda i: normalize(map(lambda x: x[i], X)), range(len(X[0])))
            self.normalization_factor = map(lambda (a, b): b,  X)
        else:
            X = map(lambda i: normalize(map(lambda x: x[i], X), self.normalization_factor[i]), range(len(X[0])))
        X = map(lambda (a, b): a,  X)
        for i, d in enumerate(data):
            d['attributes'] = map(lambda j: X[j][i], xrange(len(X)))
        return data

    def fit(self, train):
        self.model = self.__mapper__(train)

    def transform(self, test):
        test = self.__mapper__(test)
        model = self.model
        if model is None:
            raise Exception(MODEL_NOT_TRAINED_ERROR)
        K = self.K
        for tt in test:
            heap = dict()
            for tr in model:
                heap[tr['id']] = distance(tt, tr)

            heap = [(value, key) for key, value in heap.items()]
            smallest = heapq.nsmallest(K, heap)
            smallest = [(key, value) for value, key in smallest]

            filter_list = []
            for point in smallest:
                ids = map(lambda l: point[0], point)
                filtered = filter(lambda d: d['id'] in ids, model)
                filter_list.append(filtered)

            labels = map(lambda x: x[0]['class'], filter_list)

            if labels.count('0') > labels.count('1'):
                tt['assigned'] = '0'
            else:
                tt['assigned'] = '1'
        return test

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)
