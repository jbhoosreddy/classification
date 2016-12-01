from __future__ import division
from helper.statistics import *
from helper.constants import *
from collections import Counter
from copy import deepcopy


class DecisionTree(object):

    def __init__(self):
        self.length = None
        self.model = None
        self.dominant_label = None

    def fit(self, train):
        prune_set = set()
        central_type = 'median'

        y = map(lambda t: t['class'], train)
        labels = set(y)
        X = map(lambda t: t['attributes'], train)
        length = len(X[0])
        self.length = length
        model = list()
        for i in xrange(length):
            nature = infer_nature(map(lambda t: t['attributes'][i], train))
            if nature == CATEGORICAL:
                filtered_train = map(lambda (i, v): v, filter(lambda (j, v): j not in prune_set, enumerate(train)))
            else:
                filtered_train = train
            model.append([])
            features = dict()
            dominant_size = -1
            for label in labels:
                filtered = filter(lambda t: t['class'] == label, filtered_train)
                x = map(lambda t: t['attributes'][i], filtered)
                size = len(filtered)
                if size > dominant_size:
                    dominant_size = size
                    dominant_label = label
                if nature == CATEGORICAL:
                    levels = set(x)
                    for level in levels:
                        count = Counter(map(lambda f: f['class'], filter(lambda f: f['attributes'][i] == level, filtered_train)))
                        if len(count) == 1:
                            model[-1].append({'label': count.most_common(1)[0][0], 'threshold': level, 'region': "="})
                else:
                    features[label] = {'size': len(filtered), 'min': min(x), 'max': max(x),
                                       'mean': mean(x), 'median': median(x), 'mode': mode(x)}
            self.dominant_label = dominant_label

            if nature != CATEGORICAL:
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
                prune_set |= self.prune(x, model[-1])
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
                nature = infer_nature([value])
                if nature == CATEGORICAL:
                    for m in model[i]:
                        if m['threshold'] == value:
                            t['assigned'] = m['label']
                            break_it = True
                else:
                    for m in model[i]:
                        if m['region'] == '<' and value < m['threshold'] or m['region'] == '>' and value > m['threshold']:
                            t['assigned'] = m['label']
                            break_it = True
                if break_it:
                    break
            if 'assigned' not in t.keys():
                t['assigned'] = self.dominant_label
        return test

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)

    def prune(self, x, model):
        prune_list = list()
        for i,v in enumerate(x):
            for m in model:
                if m['region'] == '<' and v < m['threshold'] or m['region'] == '>' and v > m['threshold']:
                    prune_list.append(i)
        return set(prune_list)
