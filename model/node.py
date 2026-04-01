import numpy as np

class Node:
    def __init__(self, state, parent = None):
        self.state = state
        self.value = 0
        self.visits = 0
        self.avg_value = 0
        self.children = []
        self.parent = parent

    def visited(self):
       return self.visits != 0


def calc_ucb(node, state, net):
    for child in node.children:
        if child.state == state:
