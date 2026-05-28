import os
import torch
import numpy as np
from game import TicTacToe
from mcts import MCTS
from neural_network import AlphaZeroNetwork

BOT_1_FILE = "alphazero_tic_tac_toe_final.pth"
BOT_2_FILE = "alphazero_tic_tac_toe_gen80.pth"
NUM_GAMES = 5
MCTS_SIMS = 10000

class RandomStrategy:
    def choose_action(self, state, player):
        actions = TicTacToe().actions(state, player)
        return np.random.randint(len(actions))

class AlphaZeroStrategy:
    def __init__(self, game, model_path, sims):
        self.game = game
        self.sims = sims
        self.net = AlphaZeroNetwork(2, 64, 3, 3, 9)
        self.net.load_state_dict(torch.load(model_path, weights_only=True))
        self.net.eval()

    def choose_action(self, state, player):
        mcts = MCTS(self.game, state, player, self.net)
        mcts.iter(self.sims)
        best_move, _ = mcts.output()

        for i, act in enumerate(self.game.actions(state, player)):
            if list(act) == list(best_move): 
                return i
        return 0

def play_match():
    game = TicTacToe()
    
    bot1 = AlphaZeroStrategy(game, BOT_1_FILE, MCTS_SIMS) if BOT_1_FILE != "random" else RandomStrategy()
    bot2 = AlphaZeroStrategy(game, BOT_2_FILE, MCTS_SIMS) if BOT_2_FILE != "random" else RandomStrategy()
    
    wins1, wins2, draws = 0, 0, 0
    print(f"{BOT_1_FILE} vs {BOT_2_FILE} ({NUM_GAMES} games)\n")

    for i in range(NUM_GAMES):
        state = game.initial_state()
        if i % 2 == 0:
            player = 1
        else:
            player = 2
        step = 0
        
        while True:
            step += 1
            if player == 1:
                strategy = bot1
            else:
                strategy = bot2
            idx = strategy.choose_action(state, player)
            
            move = game.actions(state, player)[idx]
            state, reward = game.next_state_and_reward(player, state, move)
            
            if game.end_of_game(reward, step, state):
                if reward == 0: 
                    draws += 1
                elif player == 1: 
                    wins1 += 1
                else: 
                    wins2 += 1
                break
            player = 3 - player
        
        print(f"GAME {i+1}/{NUM_GAMES} FINISHED.", end="\r")

    print(f"\n\n=== RESULTS ===")
    print(f"Bot 1 ({BOT_1_FILE}): {wins1} wins ({(wins1/NUM_GAMES)*100:.1f}%)")
    print(f"Bot 2 ({BOT_2_FILE}): {wins2} wins ({(wins2/NUM_GAMES)*100:.1f}%)")
    print(f"Draws: {draws} ({(draws/NUM_GAMES)*100:.1f}%)")

if __name__ == "__main__":
    play_match()