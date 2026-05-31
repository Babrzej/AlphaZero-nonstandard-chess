import os
import sys
import torch
import subprocess
import random
import numpy as np

# Ensure path is set up to find model and chess_env modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.neural_network import AlphaZeroNetwork
from model.mcts import MCTS
from chess_env.chess_game import Chess
from chess_env.chess_interface import play_with_strategy

MCTS_SIMS = 1000
MODEL_PATH = "model/alphazero_model_ep101.pth"
STOCKFISH_PATH = "./Fairy-Stockfish/src/stockfish"


class AlphaZeroStrategy:
    def __init__(self, game, net, mcts_simulations=MCTS_SIMS):
        self.game = game
        self.net = net
        self.sims = mcts_simulations

    def choose_action(self, state, player):
        actions = self.game.actions(state, player)
        if len(actions) == 0:
            return None, 0.0, []

        mcts = MCTS(self.game, state, player, self.net)
        with torch.no_grad():
            mcts.iter(self.sims)

        best_move, policy = mcts.output()

        action_nr = None
        for idx, act in enumerate(actions):
            if act == best_move:
                action_nr = idx
                break

        if action_nr is None:
            action_nr = 0

        return action_nr, 0.0, policy


class FairyStockfishStrategy:
    def __init__(self, engine_path, game, variant_name="my_5x5_chess", movetime=1000):
        self.game = game
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
            if line == "uciok":
                break
                
        # Zależnie od tego, gdzie jest plik variants.ini. Tutaj zakłada główny folder.
        variant_path = os.path.abspath("variants.ini")
        self._send_command(f"setoption name VariantPath value {variant_path}")
        
        self._send_command(f"setoption name UCI_Variant value {self.variant_name}")
        # Wyłączamy ELO, niech gra z pełną mocą na pokaz!
        # self._send_command("setoption name UCI_LimitStrength value true")
        # self._send_command("setoption name UCI_Elo value 2500")
        
        self._send_command("isready")
        while True:
            line = self.process.stdout.readline().strip()
            if line == "readyok":
                break
        print(f"Fairy-Stockfish gotowy do gry w interfejsie graficznym!")

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

    def choose_action(self, state, player):
        """
        Zwraca (action_idx, ocena, polisa) pod interfejs Pygame
        """
        actions = self.game.actions(state, player)
        if len(actions) == 0:
            return None, 0.0, []

        fen = self.state_to_fen(self.game, state, player)
        self._send_command(f"position fen {fen}")
        self._send_command(f"go movetime {self.movetime}")
        
        best_move_str = None
        while True:
            line = self.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                parts = line.split(" ")
                if len(parts) > 1:
                    best_move_str = parts[1]
                break
                
        chosen_action = None
        if not best_move_str or best_move_str == "(none)":
            chosen_action = random.choice(actions)
        else:
            c_from = ord(best_move_str[0]) - ord('a')
            r_from = self.game.num_of_rows - int(best_move_str[1])
            c_to = ord(best_move_str[2]) - ord('a')
            r_to = self.game.num_of_rows - int(best_move_str[3])

            promo_char = best_move_str[4] if len(best_move_str) == 5 else None
            promo_piece = None
            if promo_char:
                promo_map = {'q': 5, 'r': 4, 'b': 3, 'n': 2}
                base = promo_map.get(promo_char, 5)
                promo_piece = base if player == 1 else base + self.game.BlackShift

            for action in actions:
                if action[1] == r_from and action[2] == c_from and action[3] == r_to and action[4] == c_to:
                    if promo_piece:
                        if len(action) > 5 and action[5] == promo_piece:
                            chosen_action = action
                            break
                    else:
                        chosen_action = action
                        break
        
        if chosen_action is None:
            chosen_action = random.choice(actions)

        # Znajdowanie INDEKSU ruchu (czego wymaga Pygame)
        action_nr = 0
        for idx, act in enumerate(actions):
            if act == chosen_action:
                action_nr = idx
                break

        # Zwracamy indeks_ruchu, ocena (0.0 bo jej nie zczytujemy), i pustą listę rozkładu MCTS
        fake_q_values = [0.0] * len(actions)
        
        return action_nr, 0.0, fake_q_values

    def close(self):
        self._send_command("quit")
        self.process.terminate()


def main():
    print("Initializing 5x5 Chess game...")
    game = Chess("chess_env/boards/szachy_plansza_5x5")

    # === WYBÓR PRZECIWNIKA ===
    # Ustaw na False, jeśli chcesz grać ze Stockfishem
    PLAY_VS_ALPHAZERO = False

    if PLAY_VS_ALPHAZERO:
        print("Initializing Neural Network...")
        net = AlphaZeroNetwork(17, 32, 5, 5, 1225)
        if os.path.exists(MODEL_PATH):
            print(f"Loading weights from {MODEL_PATH}...")
            net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        else:
            print(f"WARNING: Checkpoint '{MODEL_PATH}' not found!")
        net.eval()
        ai_strategy = AlphaZeroStrategy(game, net, mcts_simulations=MCTS_SIMS)
    else:
        print("Initializing Fairy-Stockfish...")
        ai_strategy = FairyStockfishStrategy(STOCKFISH_PATH, game, movetime=1000)

    # Uruchomienie Pygame
    print("Launching Graphic Interface... Enjoy the game!")
    try:
        # Możesz zmienić str_player=1 na str_player=2, aby AI grało białymi
        play_with_strategy(game, ai_strategy, str_player=1)
    finally:
        # Wymuszenie zamknięcia procesu Stockfisha, jeśli okno zostanie zamknięte
        if not PLAY_VS_ALPHAZERO:
            ai_strategy.close()


if __name__ == "__main__":
    main()