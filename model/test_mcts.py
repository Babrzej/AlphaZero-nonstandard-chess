import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

tree = MCTS(game, game.initial_state(), 1, net)

if __name__ == "__main__":
    tree.iter(100)

    print("\n🎯 DRZEWO MCTS (max głębokość 4):")
    tree.print_tree(max_depth=4)

    print("\n📊 STATYSTYKI RUCHU:")
    tree.print_tree_compact()