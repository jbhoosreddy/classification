from __future__ import division
from helper.statistics import *
from helper.constants import *
from helper.utils import categorize, generalization_error
from helper.structures import DecisionTreeNode
from helper.structures import Tree
from collections import Counter

__all__ = ['DecisionTree']


class DecisionTree(object):

    def __init__(self, RENDER_TREE=False, prune=True):
        self.shape = None
        self.model = None
        self.dominant_label = None
        self.tree = Tree(DecisionTreeNode)
        self.render_tree = RENDER_TREE

    def fit(self, train):
        def clean(v):
            del v['entropy']
            del v['total']
            del v['data']
        tree = self.tree
        y = map(lambda t: t['class'], train)
        prior = Counter(y)
        self.dominant_label = prior.most_common(1)[0][0]
        labels = set(y)
        self.shape = len(y), len(train[0]['attributes'])
        node_entropy = entropy([prior[label]/self.shape[0] for label in labels])
        filtered_train = map(lambda x: x, train)
        seen_attributes = list()
        new_range = None
        entropy_value = gain_value = None
        while True:
            features = dict()
            entropy_dict = dict()
            gain_dict = dict()
            max_gain = 0
            selected_attribute = -1
            X = map(lambda t: t['attributes'], filtered_train)
            X = map(lambda i: categorize(map(lambda x: x[i], X)), range(self.shape[1]))
            for i in range(self.shape[1]):
                if i in seen_attributes:
                    continue
                features[i] = dict()
                values = set(X[i])
                for value in values:
                    node_filtered_train = filter(lambda t: value[0] <= t['attributes'][i] < value[1] if isinstance(value, tuple) else t['attributes'][i] == value, filtered_train)
                    features[i][value] = Counter(map(lambda tt: tt['class'], node_filtered_train))
                    features[i][value]['total'] = sum(features[i][value].values())
                    features[i][value]['data'] = node_filtered_train
                    features[i][value]['entropy'] = features[i][value]['total'] / self.shape[0] * entropy([features[i][value][label]/features[i][value]['total'] for label in labels])
                entropy_dict[i] = sum([c['entropy'] for c in features[i].values()])
                gain_dict[i] = node_entropy - entropy_dict[i]
                if gain_dict[i] >= max_gain:
                    max_gain = gain_dict[i]
                    selected_attribute = i
            if new_range is None and infer_nature(X[selected_attribute]) == CATEGORICAL:
                min_value = min(map(lambda x: x[0], set(X[selected_attribute])))
                max_value = max(map(lambda x: x[1], set(X[selected_attribute])))
                new_range = min_value, max_value
            else:
                new_range = None
            if entropy_value is None:
                entropy_value = entropy_dict[selected_attribute]
                gain_value = gain_dict[selected_attribute]
            seen_attributes.append(selected_attribute)
            if selected_attribute == -1:
                break
            node_value = {'id': selected_attribute,
                          'gain': gain_value,
                          'entropy': entropy_value,
                          'size': len(filtered_train),
                          'data': filtered_train,
                          'range': new_range}
            tree.insert(node_value, map(lambda (k, v): {'range': k, 'entropy': v['entropy'], 'gain': v['gain'], 'count': v, 'seen': list(seen_attributes), 'size': v['total'], 'data': v['data']}, features[selected_attribute].items()))
            map(lambda (k, v): clean(v), features[selected_attribute].items())
            while True:
                current_tree_node, parent_tree_node = tree.get_current(tree.root)
                if current_tree_node is None:
                    break
                if len(current_tree_node['seen']) < self.shape[1]:
                    break
                else:
                    current_tree_node.leaf_node = True
                    current_tree_node['label'] = current_tree_node['count'].most_common(1)[0][0]
            if current_tree_node is None:
                break
            new_range = current_tree_node['range']
            entropy_value = current_tree_node['entropy']
            gain_value = current_tree_node['gain']
            seen_attributes = current_tree_node['seen']
            selected_attribute = seen_attributes[-1]
            filtered_train = filter(lambda t: new_range[0] <= t['attributes'][selected_attribute] < new_range[1] if isinstance(new_range, tuple) else t['attributes'][selected_attribute] == new_range, current_tree_node['data'])
        tree.prune_tree()
        tree.clean(['data', 'gain', 'count', 'size'])
        if self.render_tree:
            print tree

    def transform(self, test):
        tree = self.tree
        if tree is None:
            raise Exception(MODEL_NOT_TRAINED_ERROR)

        for i, t in enumerate(test):
            x = t['attributes']
            current_node = tree.root
            while True:
                current_id = current_node['id']
                val = x[current_id]
                new_child = filter(lambda child: child['range'][0] <= val < child['range'][1] if isinstance(child['range'], tuple) else val == child['range'],  current_node.children)
                if len(new_child) == 0:
                    break
                else:
                    new_child = new_child[0]
                if 'label' in new_child.value.keys():
                    t['assigned'] = new_child['label']
                    break
                current_node = new_child
            if 'assigned' not in t.keys():
                t['assigned'] = self.dominant_label
        return test

    def fit_transform(self, train, test):
        self.fit(train)
        return self.transform(test)


def prune_tree(self, node=None):
    if node is None:
        node = self.root
    if node['entropy'] == 0:
        node.leaf_node = True
    if 'count' in node.value.keys():
        error = generalization_error(node)
        if len(error) == 1:
            node['label'] = error.keys()[0]
    for child in node.children:
        self.prune_tree(child)

Tree.prune_tree = prune_tree
