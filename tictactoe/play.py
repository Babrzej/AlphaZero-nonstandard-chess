import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import TicTacToe
from mcts import MCTS 
from neural_network import AlphaZeroNetwork
from interface import Interface_TicTacToe

MCTS_SIMS = 10000
MODEL_PATH = "alphazero_tic_tac_toe_gen100.pth"
class AlphaZeroStrategy:
    def __init__(self, game, net, mcts_simulations=10000):
        self.game = game
        self.net = net
        self.sims = mcts_simulations

    def choose_action(self, state, player):
        mcts = MCTS(self.game, state, player, self.net)
        mcts.iter(self.sims)
        best_move, _ = mcts.output()

        for i, act in enumerate(self.game.actions(state, player)):
            if list(act) == list(best_move): 
                return i
        return 0

def main():
    
    game = TicTacToe()
    net = AlphaZeroNetwork(2, 64, 3, 3, 9)

    net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    net.eval() # Ustawiamy sieć w tryb przewidywania (bardzo ważne!)
    
    ai_strategy = AlphaZeroStrategy(game, net, mcts_simulations=MCTS_SIMS)

    gui = Interface_TicTacToe(game)
    gui.play_with_strategy(ai_strategy, str_player=2)

if __name__ == "__main__":
    main()