from helper.utils import split, load_data
from helper.metrics import Performance
from KNearestNeighbour import *
from DecisionTree import *
from NaiveBayes import *

METHOD = "DT"
N = 10
K = 8


def main(data, method, n, **kwargs):
    if method == "DT":
        model = DecisionTree()
    elif method == "KNN":
        model = KNN(kwargs['K'])
    elif method == "NB":
        model = NaiveBayes()
    partitions = list(split(data, n))

    for i in range(n):

        test = partitions[i]
        train = []

        print 'i:', i
        for j in range(0, i):
            train += partitions[j]

        for j in range(i+1, n):
            train += partitions[j]

        result = model.fit_transform(train, test)

        actual = map(lambda t: t['class'], result)
        predicted = map(lambda t: t['assigned'], result)

        metrics = Performance(actual, predicted)
        print metrics

filename = 'project3_dataset1'
data = load_data('data/' + filename + '.txt',  map_to_int=False)
main(data=data, method=METHOD, n=N, K=K)

