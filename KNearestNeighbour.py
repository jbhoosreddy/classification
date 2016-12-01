from helper.utils import load_data, distance
import heapq


class KNN(object):

    def __init__(self, K):
        self.model = None
        self.K = K

    def fit(self, train):
        self.model = train

    def transform(self, test):
        K = self.K
        model = self.model
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
