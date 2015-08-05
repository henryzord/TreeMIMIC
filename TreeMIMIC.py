__author__ = 'Henry'


import sys
import copy
import random
import graphviz
import operator
import itertools
import numpy as np
import networkx as nx
from collections import Counter
from string import ascii_lowercase
from matplotlib import pyplot as plt

from Node import Node
from Color import Color
from ModelGraph import ModelGraph


def declare_function(header, body):
    """
    Declares the header function using the body code.

    :type header: str
    :param header: The name of the function.

    :type body: str
    :param body: The code of the function.

    :rtype: function
    :return: A pointer to the function.
    """
    exec body
    return locals()['f_' + str(header)]


class TreeMIMIC(object):
    population_size = 0
    model_graph = None
    palette = []
    population = []
    bbn = None

    dependencies = []
    _bbn_functions = []
    node_names = []

    def __init__(self, population_size, model_graph, colors):
        """
        Initializes the MIMIC algorithm;

        :type population_size: int
        :param population_size: The size of the population.

        :type model_graph: ModelGraph
        :param model_graph: The ModelGraph which the population will be copied of.

        :type colors: list
        :param colors: a list of valid colors to choose from.
        """
        self.population_size = population_size
        self.model_graph = model_graph
        self.palette = colors

        self.node_names = self.model_graph.names

        self.population = np.array(
            map(
                lambda x: copy.deepcopy(self.model_graph),
                xrange(self.population_size)
            )
        )

        self.dependencies = {None: self.node_names}
        self.__build_bbn__(depends_on=None)

    def solve(self, max_iter=100):
        """
        Solves the k-max coloring problem. Exports the best individual
        along with the bayesian belief network to a pdf file.

        :rtype max_iter: int
        :param max_iter: Max number of iterations.
        """
        i = 1
        while i <= max_iter:
            sys.stdout.write('\r' + 'Iterations: ' + "%03d" % (i,))
            i += 1

            self.__sample__()
            fitness = map(lambda x: x.fitness, self.population)
            median = np.median(fitness)
            fittest = list(itertools.ifilter(lambda x: x.fitness >= median, self.population))

            # former depends on latter in the tuple
            self.dependencies = self.__search_dependencies__(fittest)
            self.__build_bbn__(depends_on=self.dependencies, fittest=fittest)

            if self.__has_converged__():
                break

        print '\n'
        self.__export__(i, screen=True, pdf=False, file=True)

    def __build_bbn__(self, depends_on=None, fittest=[]):
        """
        Build a bayesian belief network and sets it to self._bbn_functions.

        :type depends_on: dict
        :param depends_on: the dependency chain between attributes.

        :type fittest: list
        :param fittest: A list of the fittest individuals (denoted as ModelGraph's) for this generation.
        """
        functions = []
        if not depends_on:
            # will build a simple chain dependency
            for node in self.node_names:
                _str = "def f_" + node + "(" + node + "):\n    return " + str(1. / float(len(self.palette)))
                func = declare_function(node, _str)
                functions += [(node, func)]

        else:
            # delete former functions to avoid any overlap in the next executions
            if len(self._bbn_functions) > 0:
                for func in self._bbn_functions:
                    del func

            # fittest_dict is a collection of the fittest individuals, grouped by their attributes in a dictionary
            fittest_dict = TreeMIMIC.__rotate_dict__(map(lambda x: x.colors, fittest), dict_to_list=False)
            functions = []
            count_fittest = len(fittest)

            for attribute in self.node_names:
                parent = list(itertools.ifilter(lambda x: attribute in x[1], self.dependencies.iteritems()))[0][0]
                _str = self.__build_function__(attribute, count_fittest, fittest_dict, dependency=parent)
                func = declare_function(attribute, _str)
                functions += [(attribute, func)]

        self._bbn_functions = dict(functions)

    def __build_function__(self, drawn, count_fittest, fittest_dict, dependency=None):
        """
        Builds a function, which later will be used to infer values for attributes.

        :type drawn: str
        :param drawn: The attribute which function will be defined.

        :type count_fittest: int
        :param count_fittest: The number of fittest individuals for this generation.

        :type fittest_dict: dict
        :param fittest_dict: The fittest individuals grouped by attributes.

        :type dependency: str
        :param dependency: the attribute which the drawn attribute depends on, or None otherwise.

        :rtype: str
        :return: The function body.
        """

        carriage = "    "
        _str = "def f_" + drawn + "(" + drawn + (', ' + dependency if dependency else '') + "):\n"

        if not dependency:
            count_drawn = Counter(fittest_dict[drawn])

            for i, color in enumerate(self.palette):
                _str += carriage + "if " + drawn + " == '" + color + "':\n" + (carriage * 2) + "return " + \
                    str(float(count_drawn.get(color) if color in count_drawn else 0.) / float(count_fittest)) + "\n"

        else:
            iterated = itertools.product(self.palette, self.palette)
            count_dependency = Counter(zip(fittest_dict[drawn], fittest_dict[dependency]))

            for ct in iterated:
                denominator = np.sum(map(lambda x: count_dependency.get(x) if x in count_dependency else 0., itertools.product(self.palette, [ct[1]])))

                value = 0. if denominator == 0. else float(count_dependency.get((ct[0], ct[1])) if (ct[0], ct[1]) in count_dependency else 0.) / denominator

                _str += carriage + "if " + drawn + " == '" + ct[0] + "' and " + dependency + " == '" + ct[1] + "':\n"
                _str += (carriage * 2) + "return " + str(value) + "\n"

        return _str

    def __sample__(self):
        """
        Assigns colors to the population of graphs.
        """
        sample = dict()
        children = list(itertools.product(self.dependencies[None], [None]))

        while len(children) > 0:
            # current[1] is the parent; current[0] is the child
            current = children[0]  # first child in the list

            probs = []
            if current[1] not in sample:
                for color in self.palette:
                    probs.append(self._bbn_functions[current[0]](color))
            else:
                _product = itertools.product(self.palette, sample[current[1]])
                raise NameError('implement me!')

            sample[current[0]] = np.random.choice(self.palette, size=self.population_size, replace=True, p=probs)
            children.remove(current)

        # rotates the dictionary
        sample = TreeMIMIC.__rotate_dict__(sample, dict_to_list=True)

        for graph, colors in itertools.izip(self.population, sample):
            graph.colors = colors

    @staticmethod
    def __rotate_dict__(sample, dict_to_list):
        if dict_to_list:
            return map(
                lambda x: dict(itertools.izip(sample.iterkeys(), x)),
                itertools.izip(*sample.values())
            )
        else:
            keys = sample[0].keys()
            # initializes an empty sample_dict
            sample_dict = dict(map(lambda x: (x, []), keys))
            for individual in sample:  # iterates over individuals in the sample
                for key, color in individual.iteritems():  # iterates over colors in the individual
                    sample_dict[key].append(color)

            return sample_dict

    def __search_dependencies__(self, fittest):
        """
        Infers the dependencies between attributes.

        :type fittest: list
        :param fittest: A list of the fittest individuals for this generation.

        :rtype: dict
        :return: A list of tuples containing each pair of dependencies.
        """

        dict_dependencies = dict()

        entropies = dict(map(lambda x: self.__entropy__(x, fittest), self.node_names))
        inner_node = list(itertools.ifilter(lambda x: x[1] == min(entropies.values()), entropies.items()))[0][0]
        dict_dependencies[None] = [inner_node]  # gets the first attribute with the least entropy
        not_in_the_tree = set(self.node_names) - {inner_node}

        in_the_tree = list() + [inner_node]

        while any(not_in_the_tree):
            possible_edges = filter(
                lambda x: x[0] != x[1] and (dict_dependencies[x[1]] != x[0] if x[1] in dict_dependencies else True),
                itertools.product(not_in_the_tree, in_the_tree)
            )

            entropies = dict(map(lambda x: self.__entropy__(x[0], fittest, x[1]), possible_edges))
            inner_node = list(itertools.ifilter(lambda x: x[1] == min(entropies.values()), entropies.items()))[0][0]
            if inner_node[1] in dict_dependencies:
                dict_dependencies[inner_node[1]] += [inner_node[0]]
            else:
                dict_dependencies[inner_node[1]] = [inner_node[0]]
            not_in_the_tree -= {inner_node[0]}
            in_the_tree += [inner_node[0]]

        return dict_dependencies

    @staticmethod
    def __marginalize__(dependencies, p_value):
        """
        Marginalizes the distribution given the p_value: P(p_value | any_q_value). Returns the frequency.

        :type dependencies: Counter
        :param dependencies: An instance of the class collections.Counter, with each configuration of values and
            its values. The tuples must be in the for of (p, q).

        :type p_value: Any
        :param p_value: The value of the conditioned variable.

        :rtype: float
        :return: The probability that p assumes the provided value given q values.
        """
        keys = []
        for item in dependencies.keys():
            if item[0] == p_value:
                keys.append(item)

        total = float(reduce(operator.add, dependencies.values()))
        _sum = np.sum(map(lambda x: dependencies[x], keys)) / total
        return _sum

    def __entropy__(self, attribute, sample, free=None):
        """
        Calculates the entropy of a given attribute.
        :type attribute: str
        :param attribute: The attribute name.

        :type free: str
        :param free: optional -- If the attribute is dependent on other attribute. In this case,
            it shall be provided the name of the free attribute.

        :rtype: tuple
        :return: A tuple containing the name of the attribute alongside its entropy.
        """
        if not free:
            return attribute, -1. * np.sum(
                map(
                    lambda x: (float(x) / len(sample)) * np.log2(float(x) / len(sample)),
                    Counter(map(lambda y: y.nodes[attribute].color, sample)).values()
                )
            )
        else:
            conditionals = Counter(map(lambda x: (x.nodes[attribute].color, x.nodes[free].color), sample))

            entropy = 0.
            for value in set(
                    map(lambda x: x[0], conditionals.keys())):  # iterates over the values of the conditioned attribute
                marginal = self.__marginalize__(conditionals, value)
                entropy += marginal * np.log2(marginal)

            return (attribute, free), -1. * entropy

    def __has_converged__(self):
        fitness = map(
            lambda x: x.fitness,
            self.population
        )
        median = np.median(fitness)
        result = np.all(fitness == median)
        return result

    def __best_individual__(self):
        fitness = map(lambda x: x.fitness, self.population)
        return self.population[np.argmax(fitness)]

    def __export__(self, iterations, screen=True, file=True, pdf=True):
        _str = 'Finished inference in ' + str(iterations) + ' iterations.\n'
        _str += 'Evaluations: ' + str(iterations * self.population_size) + '\n'
        _str += 'Population size: ' + str(self.population_size) + '\n'
        _str += 'Nodes: ' + str(self.model_graph.count_nodes) + '\n'
        _str += 'Colors: ' + str(len(self.palette)) + '\n'
        _str += 'Best individual fitness: ' + str(round(self.__best_individual__().fitness, 2)) + "\n"
        # _str += 'Marginalization matrix:\n'

        print _str

        _dict = copy.deepcopy(self.dependencies)
        del _dict[None]

        if pdf:
            bbn = graphviz.Digraph()
            [bbn.node(x) for x in self.node_names]

            for parent, children in _dict.iteritems():
                for child in children:
                    bbn.edge(parent, child)

            # bbn.edges(_dict.items())
            bbn.render('bbn.gv')

            self.__best_individual__().export('optimal')

        if file:
            with open('output.txt', 'w') as wfile:
                wfile.write(_str)

        if screen:
            def plot_bbn():
                plt.figure(1)
                G = nx.DiGraph()

                for parent, children in _dict.iteritems():
                    for child in children:
                        G.add_edge(parent, child)

                # layout = nx.fruchterman_reingold_layout(G)
                layout = nx.circular_layout(G)
                nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=list(itertools.repeat('cyan', len(self.node_names))))

            def plot_optimal():
                plt.figure(2)
                individual = self.__best_individual__()
                G = nx.Graph()
                G.add_nodes_from(individual.names)
                some_edges = [tuple(list(x)) for x in individual.edges]
                G.add_edges_from(some_edges)
                layout = nx.fruchterman_reingold_layout(G)
                nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=individual.colors.values())

            plot_bbn()
            plot_optimal()
            plt.show()


def main():
    _nodes = []
    count_nodes = 7  # max number of nodes = letters in the alphabet
    population_size = 200  # size of the population
    seed = None  # use None for random or any integer for predetermined randomization
    max_iter = 100  # max iterations to search for optima
    n_colors = 2  # number of colors to use

    random.seed(seed)
    np.random.seed(seed)

    for char in list(ascii_lowercase)[:count_nodes]:
        _nodes.append(Node(char))

    my_graph = ModelGraph(neighborhood=_nodes)
    my_graph.randomize_edges(chain=False)

    colors = Color.randomize_colors(n_colors=n_colors)

    mr_mime = TreeMIMIC(population_size, my_graph, colors)
    mr_mime.solve(max_iter=max_iter)


main()
