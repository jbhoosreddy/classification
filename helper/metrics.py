from __future__ import division


class Performance(object):

    def __init__(self, actual, predicted):
        pos, neg = '0', '1'
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
        return self.tp/(self.tp + self.fp)

    def recall(self):
        return self.tp/(self.tp + self.fn)

    def f1(self):
        recall = self.recall()
        precision = self.precision()
        return 2 * recall * precision / (recall + precision)

    def __str__(self):
        return "Accuracy: " + str(round(self.accuracy()*100, 2)) + \
               "\nPrecision: " + str(round(self.precision()*100, 2)) + \
               "\nRecall: " + str(round(self.recall()*100, 2)) + \
               "\nF1-Measure: " + str(round(self.f1()*100, 2))
