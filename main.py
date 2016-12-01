from helper.utils import split
from KNearestNeighbour import *
from DecisionTree import *

METHOD = "DT"
N = 10
K = 8


def main(data, method, n, **kwargs):
    if method == "DT":
        model = DecisionTree()
    elif method == "KNN":
        model = KNN(kwargs['K'])
    partitions = list(split(data, n))

    for i in range(n):

        test = partitions[i]
        train = []

        print 'i:', i
        for j in range(0,i):
            train += partitions[j]

        for j in range(i+1, n):
            train += partitions[j]

        model.fit(train)
        result = model.transform(test)

        count = 0
        for res in result:
            if res['assigned'] != res['class']:
                count += 1
        print count, len(test)-count

filename = 'project3_dataset1'
data = load_data('data/' + filename + '.txt')
main(data=data, method=METHOD, n=N, K=K)

