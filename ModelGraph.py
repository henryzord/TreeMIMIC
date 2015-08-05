import copy
import operator
import numpy as np
from Node import Node
import graphviz


class ModelGraph(object):
    """
    Attributes:

    :type nodes: list
    :param nodes: a list of the nodes.

    :type edges: list
    :param edges: a list of the edges between the nodes.
    """
    nodes = dict()
    edges = []
    _fitness = 0.
    names = []

    def __init__(self, *args, **kwargs):
        """
        Constructor for MGraph class.

        :type node: Node
        :param node: optional -- The nodes of this MGraph.

        :type neighborhood: list
        :param neighborhood: optional -- To pass the nodes of this MGraph as a list.

        :type visual: graphviz.graph
        :param visual: optional - visual engine of the graph

        :type edges: list
        :param edges: optional - edges of the graph
        """

        self.edges = kwargs['edges'] if 'edges' in kwargs else list()

        map(self.__extend_node_collection__, args)

        if 'neighborhood' in kwargs:
            map(self.__extend_node_collection__, kwargs['neighborhood'])

        self.edges = list(set(self.edges))
        self._fitness = self.fitness

        self.names = map(lambda x: x.name, self.nodes.values())

    def __copy__(self):
        _dict = dict()
        _dict['edges'] = copy.copy(self.edges)
        _dict['neighborhood'] = copy.copy(self.nodes)

        # pass dictionary as kwargs:
        # http://stackoverflow.com/questions/5710391/converting-python-dict-to-kwargs
        return ModelGraph(**_dict)

    def __deepcopy__(self, memo):
        _dict = dict()
        _dict['edges'] = copy.deepcopy(self.edges)

        nodes = [Node(copy.deepcopy(x.name), color=copy.deepcopy(x.color)) for x in self.nodes.values()]
        _dict['names'] = map(lambda x: copy.deepcopy(x.name), nodes)

        neighborhood = dict(zip(_dict['names'], nodes))

        for edge in _dict['edges']:
            links = list(edge)
            neighborhood[links[0]].link(neighborhood[links[1]])

        _dict['nodes'] = neighborhood

        # pass dictionary as kwargs:
        # http://stackoverflow.com/questions/5710391/converting-python-dict-to-kwargs
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in _dict.items():
            setattr(result, k, v)
        return result

    def __extend_node_collection__(self, node, make_edges=True):
        self.nodes[node.name] = node
        if make_edges:
            self.__make__edges__(node)

    def __make__edges__(self, node):
        _sorted = [''.join(sorted(x)) for x in node.str_edges()]
        self.edges.extend(_sorted)

    def randomize_edges(self, chain=False):
        """
        Randomly creates edges between nodes.

        :type chain: bool
        :param chain: Whether to chain attributes or not.
        """
        selectable = set(self.nodes)
        node_instances = self.nodes.values()

        for i, node_name in enumerate(self.nodes.keys()):
            # prevents a node from linking with itself and with its parent
            selectable = set(self.nodes.keys()) - ({node_name} | {x if node_name not in list(x) else node_name for x in self.edges})
            to_connect = None
            if not chain:  # random linkage
                to_connect = self.nodes[np.random.choice(list(selectable))]
            elif (i + 1) < len(self.nodes):  # chain linkage
                to_connect = node_instances[i + 1]

            if to_connect is not None:
                node_instances[i].link(to_connect)
                self.__make__edges__(node_instances[i])

        self.edges = list(set(self.edges))

    @property
    def fitness(self):
        """
        The fitness of this graph - the number of edges that it's nodes
        doesn't have the same color, normalized to [0,1].

        :rtype: float
        :return: Ranges from 0 (worst fitness) to 1 (best fitness).
        """
        return self._fitness

    @property
    def count_nodes(self):
        """
        :rtype: int
        :return: Number of nodes in this graph.
        """
        return len(self.nodes.keys())

    @property
    def colors(self):
        return dict(map(lambda x: (x.name, x.color), self.nodes.values()))

    @colors.setter
    def colors(self, value):
        if isinstance(value, list) or isinstance(value, np.ndarray):
            for i, node in enumerate(self.nodes.itervalues()):
                node.color = value[i]
        elif isinstance(value, dict):
            for key in value.iterkeys():
                self.nodes[key].color = value[key]

        diff_colors = reduce(
            operator.add,
            map(
                lambda y: len(set(
                    map(
                        lambda x: self.nodes[x].color,
                        list(y)
                    )
                )) > 1,
                self.edges
            )
        )
        self._fitness = float(diff_colors) / len(self.edges)

    def export(self, filename):
        visual = graphviz.Graph()

        for node in self.nodes.values():
            visual.node(node.name, node.name, style='filled', fillcolor=node.color)

        visual.edges(self.edges)
        visual.render(filename + '.gv')