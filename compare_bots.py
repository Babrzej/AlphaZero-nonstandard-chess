import os
import torch
import random
import numpy as np

# CRITICAL: Prevent PyTorch CPU thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"

from chess_env.chess_game import Chess
from model.mcts import MCTS
from model.neural_network import AlphaZeroNetwork

# ==========================================
# ARENA CONFIGURATION
# ==========================================
# Put your specific .pth files here, or type "random" to test against a baseline
BOT_1_FILE = "model/alphazero_model_ep18.pth"
BOT_2_FILE = "model/alphazero_model_ep9.pth"

NUM_GAMES = 10
MCTS_SIMS = 75  # Keep this matching your training simulations
MAX_GAME_STEPS = 100  # Prevent infinite games
BOARD_PATH = "/home/babrzej/Documents/AlphaZero-nonstandard-chess/chess_env/boards/szachy_plansza_5x5"


# ==========================================


class RandomStrategy:
    def choose_action(self, game, state, player):
        actions = game.actions(state, player)
        if not actions:
            return None
        return random.choice(actions)


class AlphaZeroStrategy:
    def __init__(self, game, model_path, sims):
        self.game = game
        self.sims = sims

        # Must match the 5x5 network architecture from your training script
        self.net = AlphaZeroNetwork(17, 32, 5, 5, 1225)

        # Load weights safely to CPU
        self.net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.net.eval()

    def choose_action(self, game, state, player):
        mcts = MCTS(self.game, state, player, self.net)

        with torch.no_grad():  # Prevent memory leaks during evaluation
            mcts.iter(self.sims)

        best_move, _ = mcts.output()
        return best_move


def play_match():
    print(f"Initializing Game Environment...")
    game = Chess(BOARD_PATH)

    print(f"Loading Bot 1: {BOT_1_FILE}")
    bot1 = AlphaZeroStrategy(game, BOT_1_FILE, MCTS_SIMS) if BOT_1_FILE != "random" else RandomStrategy()

    print(f"Loading Bot 2: {BOT_2_FILE}")
    bot2 = AlphaZeroStrategy(game, BOT_2_FILE, MCTS_SIMS) if BOT_2_FILE != "random" else RandomStrategy()

    bot1_wins, bot2_wins, draws = 0, 0, 0
    print(f"\n--- MATCH START: Bot 1 vs Bot 2 ({NUM_GAMES} games) ---\n")

    for i in range(NUM_GAMES):
        state = game.initial_state()
        player = 1
        step = 0

        # Alternate who plays White (Player 1)
        # Even games: Bot 1 is White. Odd games: Bot 2 is White.
        bot1_plays_as = 1 if i % 2 == 0 else 2

        color_str = "White" if bot1_plays_as == 1 else "Black"
        print(f"Game {i + 1}/{NUM_GAMES} (Bot 1 is playing {color_str})...", end=" ", flush=True)

        while True:
            step += 1

            # Check for Stalemate (No legal moves)
            legal_moves = game.actions(state, player)
            if len(legal_moves) == 0:
                draws += 1
                print(f"Draw (Stalemate at step {step})")
                break

            # Determine whose turn it is
            if player == bot1_plays_as:
                strategy = bot1
            else:
                strategy = bot2

            best_move = strategy.choose_action(game, state, player)
            state, reward = game.next_state_and_reward(player, state, best_move)

            # Check Terminal States (Checkmate)
            if reward != 0:
                if player == bot1_plays_as:
                    bot1_wins += 1
                    print(f"Bot 1 WINS (Checkmate at step {step})")
                else:
                    bot2_wins += 1
                    print(f"Bot 2 WINS (Checkmate at step {step})")
                break

            # Check Draw limits
            if step >= MAX_GAME_STEPS:
                draws += 1
                print(f"Draw (Turn limit reached at step {step})")
                break

            player = 3 - player  # Swap turn

    # --- PRINT FINAL RESULTS ---
    print(f"\n=======================")
    print(f"      FINAL SCORE      ")
    print(f"=======================")
    print(f"Bot 1 ({BOT_1_FILE}): {bot1_wins} wins ({(bot1_wins / NUM_GAMES) * 100:.1f}%)")
    print(f"Bot 2 ({BOT_2_FILE}): {bot2_wins} wins ({(bot2_wins / NUM_GAMES) * 100:.1f}%)")
    print(f"Draws: {draws} ({(draws / NUM_GAMES) * 100:.1f}%)")


if __name__ == "__main__":
    play_match()