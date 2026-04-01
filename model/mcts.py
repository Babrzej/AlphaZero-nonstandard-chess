import numpy as np
from node import *
from chess_env.chess_game import Chess

c_ucb = np.sqrt(2)

class MCTS:
    def __init__(self, root, net):
        self.root = root
        self.net = net

    def select(self, node):

        possible_moves = Chess.actions(node.state, node.player)
        ucb = []
        policy = self.net.get_policy(node.state)

        for move, pol in possible_moves, policy:
            next_state = Chess.next_state_and_reward(node.state, move)
            move_visited = False

            for child in node.children:
                if child.state == next_state:
                    new_ucb = calc_ucb(node.visits, child.visits, child.avg_value, pol)
                    ucb.append(new_ucb)
                    move_visited = True
                    break

            if not move_visited:
                new_ucb = calc_ucb(node.visits, 0, 0, pol)
                ucb.append(new_ucb)

        selected_move = possible_moves[np.argmax(ucb)]
        selected_state = Chess.next_state_and_reward(node.state, selected_move)

        for child in node.children:
            if child.state == selected_state:
                return self.select(child)

        return selected_state, node


    def expand(self, state, node):
        child = Node(state, node)
        node.children.append(child)
        child.visits += 1
        child.value += self.net.get_value(state)
        return child


    def backpropagate(self, node, value):
        if node == self.root:
            return None
        parent = node.parent
        parent.visits += 1
        parent.value += value
        return self.backpropagate(parent, value)

    def iter(self, num_of_iterations):
        for i in range(num_of_iterations):
            next_state, parent = self.select(self.root)
            new_node = self.expand(next_state, parent)
            self.backpropagate(new_node, new_node.value)






def calc_ucb(visits_parent, visits, value, policy, c = c_ucb):
    p = c * policy * np.sqrt(visits_parent) / (1 + visits)
    ucb = value + p
    return ucb

