import os
import sys
import torch
import numpy as np

# Ensure path is set up to find model and chess_env modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.neural_network import AlphaZeroNetwork
from model.mcts import MCTS
from chess_env.chess_game import Chess
from chess_env.chess_interface import play_with_strategy

MCTS_SIMS = 1000  # Number of search simulations (400-800 provides a strong, fast opponent)
MODEL_PATH = "model/alphazero_model_ep101.pth"


class AlphaZeroStrategy:
    def __init__(self, game, net, mcts_simulations=MCTS_SIMS):
        self.game = game
        self.net = net
        self.sims = mcts_simulations

    def choose_action(self, state, player):
        """
        Executes MCTS search and returns (action_idx, value, policy_distribution)
        as expected by chess_interface.py.
        """
        actions = self.game.actions(state, player)
        if len(actions) == 0:
            return None, 0.0, []

        mcts = MCTS(self.game, state, player, self.net)
        with torch.no_grad():
            mcts.iter(self.sims)

        best_move, policy = mcts.output()

        # Find the index of the chosen move in the current legal action list
        action_nr = None
        for idx, act in enumerate(actions):
            if act == best_move:
                action_nr = idx
                break

        # Fallback in case of mismatch
        if action_nr is None:
            action_nr = 0

        # We return (action_index, estimated_value, policy_distribution_for_debug)
        return action_nr, 0.0, policy


def main():
    print("Initializing 5x5 Chess game and neural network...")
    # Load 5x5 board configurations
    game = Chess("chess_env/boards/szachy_plansza_5x5")

    # Initialize neural network with 5x5 board specifications
    net = AlphaZeroNetwork(17, 32, 5, 5, 1225)

    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    else:
        print(f"WARNING: Checkpoint '{MODEL_PATH}' not found! AI will play using random initialized weights.")

    net.eval()  # Set network to evaluation mode

    # Build strategy wrapper for MCTS
    ai_strategy = AlphaZeroStrategy(game, net, mcts_simulations=MCTS_SIMS)

    # Launch Pygame chess interface
    print("Launching Graphic Interface... Enjoy the game!")
    play_with_strategy(game, ai_strategy, str_player=1)


if __name__ == "__main__":
    main()