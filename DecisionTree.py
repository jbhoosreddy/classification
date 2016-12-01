from __future__ import division
from helper.statistics import mean, median, mode
from copy import deepcopy


class DecisionTree(object):

    def __init__(self):
        self.length = None
        self.model = None
        self.dominant_label = None

    def fit(self, train):
        central = median
        central_type = 'median'

        y = map(lambda t: t['class'], train)
        labels = set(y)
        X = map(lambda t: t['attributes'], train)
        length = len(X[0])
        self.length = length
        model = list()
        for i in xrange(length):
            model.append([])
            features = dict()
            dominant_size = -1
            for label in labels:
                filtered = filter(lambda t: t['class'] == label, train)
                x = map(lambda t: t['attributes'][i], filtered)
                size = len(filtered)
                value = central(x)
                if size > dominant_size:
                    dominant_size = size
                    dominant_label = label
                features[label] = {'size': len(filtered), 'min': min(x), 'max': max(x),
                                   'mean': mean(x), 'median': median(x), 'mode': mode(x)}
            self.dominant_label = dominant_label
            centrals = map(lambda v: v[central_type], features.values())

            for label in labels:
                other_features = deepcopy(features)
                del other_features[label]
                mins = map(lambda v: v['min'], other_features.values())
                maxs = map(lambda v: v['max'], other_features.values())
                if features[label][central_type] == max(centrals):
                    current_min = features[label]['min']
                    other_max = max(maxs)
                    if other_max > current_min:
                        model[-1].append({'threshold': other_max, 'label': label, 'region': '>'})
                    else:
                        model[-1].append({'threshold': current_min, 'label': label, 'region': '>'})
                elif features[label][central_type] == min(centrals):
                    current_max = features[label]['max']
                    other_min = min(mins)
                    if other_min < current_max:
                        model[-1].append({'threshold': other_min, 'label': label, 'region': '<'})
                    else:
                        model[-1].append({'threshold': current_max, 'label': label, 'region': '<'})
        self.model = model

    def transform(self, test):
        model = self.model
        if model is None:
            raise Exception("You need to train your model first. Call fit method with training data.")

        for t in test:
            x = t['attributes']
            for i in xrange(self.length):
                break_it = False
                value = x[i]
                for m in model[i]:
                    if m['region'] == '<' and value < m['threshold'] or m['region'] == '>' and value > m['threshold']:
                        t['assigned'] = m['label']
                        break_it = True
                if break_it:
                    break
            if 'assigned' not in t.keys():
                t['assigned'] = self.dominant_label
        return test
