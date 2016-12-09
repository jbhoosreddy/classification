from __future__ import division
from collections import Counter


class Performance(object):

    def __init__(self, actual, predicted, labels=('0', '1')):
        prior = Counter(actual)
        if len(prior) == 1:
            pos = prior.keys()[0]
            if pos == labels[0]:
                neg = labels[1]
            else:
                neg = labels[0]
        else:
            pos, neg = map(lambda (k, v): k, prior.most_common(len(prior)))
        self.actual = actual
        self.predicted = predicted
        tp = tn = fp = fn = 0
        self.total = len(actual)
        for i in xrange(self.total):
            if actual[i] == predicted[i] == pos:
                tp += 1
            elif actual[i] == predicted[i] == neg:
                tn += 1
            elif actual[i] == pos and predicted[i] == neg:
                fn += 1
            elif actual[i] == neg and predicted[i] == pos:
                fp += 1
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def accuracy(self):
        return (self.tp + self.tn) / self.total

    def precision(self):
        if self.tp == self.fp == 0:
            return 100
        return self.tp/(self.tp + self.fp)

    def recall(self):
        if self.tp == self.fn == 0:
            return 100
        return self.tp/(self.tp + self.fn)

    def f1(self):
        recall = self.recall()
        precision = self.precision()
        return 2 * recall * precision / (recall + precision)

    def __render__(self, value):
        if value >= 100:
            return "Inf"
        return str(round(value * 100, 2))

    def __str__(self):
        return "Accuracy: " + self.__render__(self.accuracy()) + " %" + \
               "\nPrecision: " + self.__render__(self.precision()) + " %" + \
               "\nRecall: " + self.__render__(self.recall()) + " %" + \
               "\nF1-Measure: " + self.__render__(self.f1()) + " %"


