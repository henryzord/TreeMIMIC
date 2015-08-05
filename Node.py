import copy


class Node(object):
    name = ''
    color = None
    neighbours = []

    def __init__(self, name, **kwargs):
        self.name = name
        self.color = "#ffffff" if 'color' not in kwargs else kwargs['color']
        self.neighbours = [] if 'neighbours' not in kwargs else kwargs['neighbours']

    def __deepcopy__(self, memo):
        _dict = {
            'name': copy.deepcopy(self.name),
            'color': copy.deepcopy(self.color),  # shallow copy
            'neighbours': copy.copy(self.neighbours)  # shallow copy
        }

        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in _dict.items():
            setattr(result, k, v)
        return result

    def __copy__(self):
        _dict = {
            'color': copy.copy(self.color),  # shallow copy
            'neighbours': copy.copy(self.neighbours)  # shallow copy
        }
        name = copy.copy(self.name)
        return Node(name, **_dict)

    def link(self, node):
        if node not in self.neighbours and self not in node.neighbours and node != self:
            self.neighbours.append(node)
            node.neighbours.append(self)

    def clear_linkage(self):
        self.neighbours = []
        return self

    def str_edges(self):
        return [self.name + x.name for x in self.neighbours]