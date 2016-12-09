from helper.utils import split, load_data
from helper.metrics import Performance
from helper.statistics import mean
from time import time
from KNearestNeighbour import *
from NaiveBayes import *
from RandomForest import *
from DecisionTree import *
from Boost import *
range = xrange

METHOD = "DT"
P = 3
N = 10
K = 8
M = 10
T = 10


def main(data, method, P, **kwargs):
    if method == "DT":
        model = DecisionTree()
    elif method == "RF":
        model = RandomForest(kwargs['T'], kwargs['M'], kwargs['bagging'])
    elif method == "KNN":
        model = KNN(kwargs['K'], kwargs['scaling'])
    elif method == "NB":
        model = NaiveBayes()
    elif method == "BST":
        model = Boost(kwargs['T'], kwargs['M'], kwargs['bagging'], kwargs['K'], kwargs['scaling'])
    partitions = list(split(data, P)) if P > 1 else [data]

    metrics = list()
    for i in range(P):

        test = partitions[i]
        train = []

        for j in range(0, i):
            train += partitions[j]

        for j in range(i+1, P):
            train += partitions[j]

        result = model.fit_transform(train, test)

        actual = map(lambda t: t['class'], result)
        predicted = map(lambda t: t['assigned'], result)

        metrics.append(Performance(actual, predicted))
        print metrics[-1]

    if not P == 1:
        print '-----------------------------'
        print "Accuracy:", round(mean(filter(lambda v: v <= 1, map(lambda m: m.accuracy(), metrics)))*100, 2), "%"
        print "Precision:", round(mean(filter(lambda v: v <= 1, map(lambda m: m.precision(), metrics)))*100, 2), "%"
        print "Recall:", round(mean(filter(lambda v: v <= 1, map(lambda m: m.recall(), metrics)))*100, 2), "%"
        print "F1-Measure:", round(mean(filter(lambda v: v <= 1, map(lambda m: m.f1(), metrics)))*100, 2), "%"

if __name__ == '__main__':
    filename = 'project3_dataset4'
    data = load_data('data/' + filename + '.txt', map_to_int=True if METHOD == "KNN" else False)
    start_time = time()
    main(data=data, method=METHOD, P=P, K=K, T=T, M=M, N=N, scaling=False, bagging=False)
    print("--- %s seconds ---" % (time() - start_time))
