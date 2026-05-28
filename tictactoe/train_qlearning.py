#!/usr/bin/env python3
"""
Trenuj Q-learning dla tic-tac-toe identycznie jak szachy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import TicTacToe
from interface import Interface_TicTacToe
from chess_env.Qlearning import Strategy_Qdict, board_game_train_Q2


if __name__ == "__main__":
    game = TicTacToe()

    strategy_w = Strategy_Qdict(game)
    strategy_b = Strategy_Qdict(game)

    print("=" * 50)
    print("TRENING Q-LEARNING DLA TIC-TAC-TOE")
    print("=" * 50)
    strategy_w, strategy_b = board_game_train_Q2(
        game,
        players_to_train=[1, 2],
        strategy_w=strategy_w,
        strategy_b=strategy_b,
        number_of_games=1000
    )

    print("✓ Trening ukończony!")
    print(f"Player 1 strategie: {len(strategy_w.Q_dict)} stanów")
    print(f"Player 2 strategie: {len(strategy_b.Q_dict)} stanów")
    print("\n" + "=" * 50)
    print("URUCHAMIAM GRĘ")
    print("=" * 50)

    # Odpal grę z wytrenowanym modelem
    gui = Interface_TicTacToe(game)
    gui.play_with_strategy(strategy_b, str_player=2)

