from __future__ import division


class NaiveBayes(object):

    def __init__(self):
        pass

    def fit(self, train):
        raise NotImplemented

    def transform(self, test):
        raise NotImplemented

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)
