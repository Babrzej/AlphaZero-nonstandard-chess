import numpy as np

# Tu prawie nic nie !!!

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

