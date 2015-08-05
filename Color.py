__author__ = "Henry"

import numpy as np


class Color(object):
    hexadecimal_color_length = 6

    def __init__(self):
        pass

    @staticmethod
    def randomize_colors(n_colors, colors=[], p=[]):
        return [Color.randomize_color(colors, p) for x in range(n_colors)]

    @staticmethod
    def randomize_color(colors=[], p=[]):
        values = '0123456789ABCDEF'

        if not colors:  # invalid color array; may choose any color
            color = np.random.choice(list(values), size=Color.hexadecimal_color_length, replace=True)
            return '#' + ''.join(color)
        else:
            if not p:
                color = np.random.choice(colors, size=1)
            else:
                color = np.random.choice(a=colors, size=1, replace=True, p=p)
            return color