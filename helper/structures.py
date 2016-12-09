class Node(object):

    def __init__(self, value=None):
        self.value = value

    def __getitem__(self, item):
        return self.value[item]

    def __delitem__(self, key):
        del self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value

    def __str__(self):
        return self.value


class TreeNode(Node):

    def __init__(self, value=None):
        super(TreeNode, self).__init__(value)
        self.parent = None
        self.children = list()
        self.leaf_node = None
        self.id = None

    def is_leaf_node(self):
        return self.leaf_node if self.leaf_node is not None else True if len(self.children) == 0 else False

    def is_real_leaf_node(self):
        return True if len(self.children) == 0 else False

    def insert_children(self, values):
        nodes = map(lambda v: DecisionTreeNode(v), values)
        self.children.extend(nodes)

    def __str__(self):
        return str(self.value) + ": " + str([str(child) for child in self.children])


class DecisionTreeNode(TreeNode):

    def __init__(self, value=None):
        super(DecisionTreeNode, self).__init__(value)

    def __str__(self, depth=0, spacing=25):
        output = ""
        length = len(self.children)
        if length == 0:
            output += "\n" + str((" " * spacing + '|') * depth)
            output += str(self['range']) + ': ' + self['label']
        else:
            midpoint = int(length / 2)
            for i, child in enumerate(self.children):
                output += child.__str__(depth + 1)
                if midpoint == i:
                    if 'id' in self.value.keys():
                        output += "\n" + str(" " * (spacing + len(str(self['id']))) * depth)
                        output += str(self['id']) + str("-" * (spacing-len(str(self['id']))-depth)) + '|'
        return output


class Tree(object):

    def __init__(self, node_type, value=None):
        self.__c__ = 0
        self.node_type = node_type
        self.root = node_type(value)
        self.current_node = self.root

    def insert(self, value, children_values):
        current, parent = self.get_current(self.root)
        if current is None:
            node = self.node_type(value)
            current = node
        current.value = value
        current.parent = parent
        current.insert_children(children_values)
        current.id = self.__c__
        self.__c__ += 1

    def get_current(self, node, parent=None):
        new_current = None
        if node.value is None:
            return node, parent
        if node.value['entropy'] > 0:
            if node.leaf_node:
                return None, None
            if node.is_real_leaf_node():
                return node, parent
            else:
                parent = node
                for child in node.children:
                    new_node, new_parent = self.get_current(child, parent)
                    if isinstance(new_node, self.node_type):
                        new_current = new_node
                        parent = new_parent
        else:
            node.leaf_node = True
        return new_current, parent


    def clean(self, keys, node=None):
        def __cleaner__(o, k):
            del o[k]
        if node is None:
            node = self.root
        node_keys = filter(lambda k: k in node.value.keys(), keys)
        map(lambda k: __cleaner__(node, k), node_keys)
        for child in node.children:
            self.clean(keys, child)

    def __str__(self):
        return str(self.root)
