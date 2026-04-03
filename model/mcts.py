import numpy as np
from node import *
from chess_env.chess_game import Chess

#---------------------------------------------
# !!! IMPORTANT !!!
# Nie obiecuje że to działa (raczej nie działa xD)
# i ma sens jakbyś chciał kontynuowac
# Więc możesz przede wszystkim "podpiąć" silnik szachowy
# i jako nn jakaś funckje która zwraca losowe wartości
#--------------------------------------------

# Constant for the UCB calculation
c_ucb = np.sqrt(2)

class MCTS:
    def __init__(self, game, root, net):
        #TODO: analyze all needed parameters
        self.game = game
        self.root = root
        self.net = net

    def select(self, node):
        #all allowed moves with their ucbs and their policies
        possible_moves = Chess.actions(node.state, node.player)
        ucb = []
        policy = self.net.get_policy(node.state)

        #we iterate over every move
        for move, pol in zip(possible_moves, policy):
            #TODO: make use of reward
            next_state, reward = Chess.next_state_and_reward(node.state, move)
            move_visited = False

            #we check if given move was already visited
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

        #If we already visited we go deeper
        for child in node.children:
            if child.state == selected_state:
                return self.select(child)

        #Ultimately return newly visited move(its state and parent node)
        return selected_state, node

    #Create node for state visited first time
    def expand(self, state, node):
        child = Node(state, node)
        node.children.append(child)
        child.visits += 1
        child.value += self.net.get_value(state)
        return child


    def backpropagate(self, node, value):
        if node == self.root:
            return None
        node.parent.visits += 1
        node.parent.value += value
        return self.backpropagate(node.parent, value)

    def iter(self, num_of_iterations):
        for i in range(num_of_iterations):
            next_state, parent = self.select(self.root)
            new_node = self.expand(next_state, parent)
            self.backpropagate(new_node, new_node.value)


def calc_ucb(visits_parent, visits, value, policy, c = c_ucb):
    p = c * policy * np.sqrt(visits_parent) / (1 + visits)
    ucb = value + p
    return ucb

