from mcts import MCTS
from chess_env.chess_game import Chess
import numpy as np

class Nett():
    def get_policy(self, state):
        return 0.5
    def get_value(self, state):
        return np.random.random() * 2 - 1

net = Nett()
game = Chess()

tree = MCTS(game, None,1, net)

if __name__ == "__main__":
    tree.iter(1000)
    tree.print_tree()