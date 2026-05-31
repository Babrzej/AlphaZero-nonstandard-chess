import os
import subprocess
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
BOT_1_ALPHAZERO = [False, "model/alphazero_model_ep20.pth"]
BOT_1_FAIRY_STOCKFISH = [True, "./Fairy-Stockfish/src/stockfish"]
BOT_1_RANDOM = [False]

BOT_2_ALPHAZERO = [True, "model/alphazero_model_ep101.pth"]
BOT_2_FAIRY_STOCKFISH = [False, "./Fairy-Stockfish/src/stockfish"]
BOT_2_RANDOM = [False]

NUM_GAMES = 10
MCTS_SIMS = 180  # Keep this matching your training simulations
MAX_GAME_STEPS = 100  # Prevent infinite games
BOARD_PATH = "chess_env/boards/szachy_plansza_5x5"


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

class FairyStockfishStrategy:
    def __init__(self, engine_path, variant_name="my_5x5_chess", movetime=10000):
        self.process = subprocess.Popen(
            [engine_path],
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.movetime = movetime
        self.variant_name = variant_name
        
        self._send_command("uci")
        
        while True:
            line = self.process.stdout.readline().strip()
            print(f"[Engine Init]: {line}")
            if line == "uciok":
                break
        variant_path = os.path.abspath("Fairy-Stockfish/src/variants.ini")
        self._send_command(f"setoption name VariantPath value {variant_path}")
        
        self._send_command(f"setoption name UCI_Variant value {self.variant_name}")
        self._send_command("setoption name UCI_LimitStrength value true")
        self._send_command("setoption name UCI_Elo value 2500")
        
        self._send_command("isready")
        while True:
            line = self.process.stdout.readline().strip()
            if line == "readyok":
                break
        print(f"Fairy-Stockfish ready! Variant: {self.variant_name}, ELO: 2500")

    def _send_command(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def state_to_fen(self, game, state, player):
        board = state.Board
        
        piece_map = {
            1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
            1001: 'p', 1002: 'n', 1003: 'b', 1004: 'r', 1005: 'q', 1006: 'k'
        }
        
        fen_rows = []
        for r in range(game.num_of_rows):
            empty = 0
            row_str = ""
            for c in range(game.num_of_columns):
                piece = board[r][c]
                if piece == 0:
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += piece_map[piece]
            if empty > 0:
                row_str += str(empty)
            fen_rows.append(row_str)
            
        fen_board = "/".join(fen_rows)
        active_color = 'w' if player == 1 else 'b'
        return f"{fen_board} {active_color} - - 0 1"

    def choose_action(self, game, state, player):
        fen = self.state_to_fen(game, state, player)
        
        self._send_command(f"position fen {fen}")
        self._send_command(f"go movetime {self.movetime}")
        
        best_move_str = None
        while True:
            line = self.process.stdout.readline().strip()
            #
            if line.startswith("bestmove"):
                parts = line.split(" ")
                if len(parts) > 1:
                    best_move_str = parts[1]
                break
                
        actions = game.actions(state, player)
        if not actions:
            return None
            
        if not best_move_str or best_move_str == "(none)":
            return random.choice(actions)
        
        c_from = ord(best_move_str[0]) - ord('a')
        r_from = game.num_of_rows - int(best_move_str[1])
        c_to = ord(best_move_str[2]) - ord('a')
        r_to = game.num_of_rows - int(best_move_str[3])

        promo_char = best_move_str[4] if len(best_move_str) == 5 else None
        promo_piece = None
        if promo_char:
            promo_map = {'q': 5, 'r': 4, 'b': 3, 'n': 2}
            base = promo_map.get(promo_char, 5)
            promo_piece = base if player == 1 else base + game.BlackShift

        actions = game.actions(state, player)
        for action in actions:
            if action[1] == r_from and action[2] == c_from and action[3] == r_to and action[4] == c_to:
                if promo_piece:
                    if len(action) > 5 and action[5] == promo_piece:
                        return action
                else:
                    return action
                
        print(f"[FATAL ERROR] Stockfish wyrzucił '{best_move_str}'. FEN był nieprawidłowy! Gram losowo.")
        return random.choice(actions)

    def close(self):
        self._send_command("quit")
        self.process.terminate()

def play_match():
    print(f"Initializing Game Environment...")
    game = Chess(BOARD_PATH)
    
    BOT_1_FILE = "Random Strategy"
    
    if BOT_1_RANDOM[0]:
        bot1 = RandomStrategy()
    elif BOT_1_FAIRY_STOCKFISH[0]:
        bot1 = FairyStockfishStrategy(BOT_1_FAIRY_STOCKFISH[1])
        BOT_1_FILE = BOT_1_FAIRY_STOCKFISH[1]
    elif BOT_1_ALPHAZERO[0]:
        bot1 = AlphaZeroStrategy(game, BOT_1_ALPHAZERO[1], MCTS_SIMS)
        BOT_1_FILE = BOT_1_ALPHAZERO[1]

    print(f"Loading Bot 1: {BOT_1_FILE}")
    
    BOT_2_FILE = "Random Strategy"
    
    if BOT_2_RANDOM[0]:
        bot2 = RandomStrategy()
        print(f"Loading Bot 2: random strategy")
    elif BOT_2_FAIRY_STOCKFISH[0]:
        bot2 = FairyStockfishStrategy(BOT_2_FAIRY_STOCKFISH[1])
        BOT_2_FILE = BOT_2_FAIRY_STOCKFISH[1]
    elif BOT_2_ALPHAZERO[0]:
        bot2 = AlphaZeroStrategy(game, BOT_2_ALPHAZERO[1], MCTS_SIMS)
        BOT_2_FILE = BOT_2_ALPHAZERO[1]
        
    print(f"Loading Bot 2: {BOT_2_FILE}")

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