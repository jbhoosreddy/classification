from random import choice
from KNearestNeighbour import *
from NaiveBayes import *
from RandomForest import *
from DecisionTree import *

__all__ = ['Boost']


class Boost(object):

    def __init__(self, N=5, **kwargs):
        self.allmodels = [KNN, NaiveBayes, RandomForest, DecisionTree]
        self.models = [choice(self.allmodels) for _ in choice(range(self.allmodels))]
        self.models = map(lambda m: m(), self.models)
