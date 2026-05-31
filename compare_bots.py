import os
import torch
import random
import glob

os.environ["OMP_NUM_THREADS"] = "1"

from src.game_logic.chess_game import Chess
from src.model.mcts import MCTS
from src.model.neural_network import AlphaZeroNetwork

# CONFIGURATION
NUM_GAMES = 10
MCTS_SIMS = 180
MAX_GAME_STEPS = 100
BOARD_PATH = "boards/szachy_plansza_5x5"


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
        self.net = AlphaZeroNetwork(17, 32, 5, 5, 1225)
        self.net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.net.eval()

    def choose_action(self, game, state, player):
        mcts = MCTS(self.game, state, player, self.net)

        with torch.no_grad():
            mcts.iter(self.sims)

        best_move, _ = mcts.output()
        return best_move

def select_bot_interactive(bot_label, game):
    print(f"\n--- Configure {bot_label} ---")
    print("1. AlphaZero")
    print("2. Random")
    while True:
        choice = input("Select strategy type (1 or 2): ").strip()
        if choice == "1":
            # Search for available checkpoints
            checkpoint_dirs = ["checkpoints", "."]
            pth_files = []
            for d in checkpoint_dirs:
                if os.path.exists(d):
                    pth_files.extend(glob.glob(os.path.join(d, "*.pth")))

            # Remove duplicates and format paths
            pth_files = list(set(os.path.abspath(f) for f in pth_files))
            pth_files_relative = [os.path.relpath(f) for f in pth_files]

            if pth_files_relative:
                print("\nAvailable checkpoints:")
                for i, file_path in enumerate(pth_files_relative):
                    print(f"  [{i + 1}] {file_path}")
            else:
                while True:
                    custom_path = input("\nNo .pth files auto-detected. Enter path manually: ").strip()
                    if os.path.exists(custom_path):
                        return AlphaZeroStrategy(game, custom_path, MCTS_SIMS), custom_path
                    print("File not found! Try again.")

        elif choice == "2":
            return RandomStrategy(), "Random Strategy"
        else:
            print("Invalid input. Select 1 or 2.")

def play_match():
    print(f"Initializing Game Environment...")
    game = Chess(BOARD_PATH)

    bot1, bot1_name = select_bot_interactive("Bot 1 (White on Game 1)", game)
    bot2, bot2_name = select_bot_interactive("Bot 2 (Black on Game 1)", game)

    bot1_wins, bot2_wins, draws = 0, 0, 0
    print(f"\n--- MATCH START: Bot 1 ({bot1_name}) vs Bot 2 ({bot2_name}) ---")
    print(f"Total games: {NUM_GAMES}\n")

    for i in range(NUM_GAMES):
        state = game.initial_state()
        player = 1
        step = 0

        bot1_plays_as = 1 if i % 2 == 0 else 2

        color_str = "White" if bot1_plays_as == 1 else "Black"
        print(f"Game {i + 1}/{NUM_GAMES} (Bot 1 plays as {color_str})...", end=" ", flush=True)

        while True:
            step += 1

            # Check for stalemate
            legal_moves = game.actions(state, player)
            if len(legal_moves) == 0:
                draws += 1
                print(f"Draw (Stalemate at step {step})")
                break

            # determine whose turn it is
            if player == bot1_plays_as:
                strategy = bot1
            else:
                strategy = bot2

            best_move = strategy.choose_action(game, state, player)
            state, reward = game.next_state_and_reward(player, state, best_move)

            # Check for checkmate
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

            # Swap player
            player = 3 - player

    # Print final results
    print(f"\n=======================")
    print(f"      FINAL SCORE      ")
    print(f"=======================")
    print(f"Bot 1 ({bot1_name}): {bot1_wins} wins ({(bot1_wins / NUM_GAMES) * 100:.1f}%)")
    print(f"Bot 2 ({bot2_name}): {bot2_wins} wins ({(bot2_wins / NUM_GAMES) * 100:.1f}%)")
    print(f"Draws: {draws} ({(draws / NUM_GAMES) * 100:.1f}%)")

if __name__ == "__main__":
    play_match()