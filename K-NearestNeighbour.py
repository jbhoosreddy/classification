
from helper.utils import load_data, distance
import heapq

filename = '/Users/vinaygoyal/DataMining/Project3/data/project3_dataset1.txt'
data = load_data(filename)
nFold = 10
K = 8


def split(a, n):
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


def kNearest(test, train):

    for tt in test:
        heap = dict()
        for tr in train:
            heap[tr['id']] = distance(tt, tr)

        heap = [(value, key) for key,value in heap.items()]
        smallest = heapq.nsmallest(K, heap)
        smallest = [(key, value) for value, key in smallest]

        filterList = []
        for point in smallest:
            ids = map(lambda l: point[0], point)
            filtered = filter(lambda d: d['id'] in ids, train)
            filterList.append(filtered)

        labels = map(lambda x: x[0]['class'], filterList)

        if labels.count('0') > labels.count('1'):
            tt['assigned'] = '0'
        else:
            tt['assigned'] = '1'

    return test, train


def main():

    partitions = list(split(data, nFold))

    for i in range(nFold):

        test = partitions[i]
        train = []

        print 'i:', i
        for j in range(0,i):
            train += partitions[j]

        for j in range(i+1, nFold):
            train += partitions[j]

        # Call the kNearest clustering method
        test, train = kNearest(test, train)

        for res in test:
            if res['assigned'] != res['class']:
                print 'unequal', res

        print len(test)
        print len(train)


main()