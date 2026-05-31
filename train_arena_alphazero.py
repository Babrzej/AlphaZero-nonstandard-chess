import os
import random
import numpy as np
import glob
import re
import copy
from collections import deque
import concurrent.futures

# CRITICAL: Prevent PyTorch from spawning hidden threads and freezing the CPU
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from model.neural_network import AlphaZeroNetwork
from model.mcts import MCTS
from chess_env.chess_game import Chess

# ==========================================
# SYSTEM PARAMETERS
# ==========================================
EPISODES = 100
GAMES_PER_EPISODE = 64
MCTS_SIMS = 180
EPOCHS = 10
BATCH_SIZE = 128
MAX_GAME_STEPS = 100
BUFFER_SIZE = 50000
NUM_WORKERS = max(1, os.cpu_count() - 2)

# PIT SETTINGS (Arena)
ARENA_OPPONENTS_OFFSETS = [3, 6] 
ARENA_GAMES_PER_OPPONENT = 20
ARENA_MCTS_SIMS = 180 
WIN_RATE_THRESHOLD = 0.55
# ==========================================

def get_latest_checkpoint():
    checkpoints = glob.glob("model/alphazero_model_ep*.pth")
    if not checkpoints:
        return None, 0

    latest_ep = 0
    latest_file = ""
    for cp in checkpoints:
        match = re.search(r"alphazero_model_ep(\d+)\.pth", cp)
        if match:
            ep_num = int(match.group(1))
            if ep_num > latest_ep:
                latest_ep = ep_num
                latest_file = cp
    return latest_file, latest_ep

def get_historical_checkpoint(target_ep):
    paths = [f"model/alphazero_model_ep{target_ep}.pth", f"alphazero_model_ep{target_ep}.pth"]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def run_arena_match(game, new_weights, old_weights, games=10, sims=50):
    new_net = AlphaZeroNetwork(17, 32, 5, 5, 1225).to("cpu")
    old_net = AlphaZeroNetwork(17, 32, 5, 5, 1225).to("cpu")
    
    new_net.load_state_dict(new_weights)
    old_net.load_state_dict(old_weights)
    new_net.eval()
    old_net.eval()

    new_wins, old_wins, draws = 0, 0, 0

    for i in range(games):
        state = game.initial_state()
        player = 1
        step = 0
        new_plays_as = 1 if i % 2 == 0 else 2 
        color_str = "White" if new_plays_as == 1 else "Black"
        print(f"      [Match {i+1}/{games}] New Model playing as {color_str}...", end=" ", flush=True)

        while True:
            step += 1
            legal_moves = game.actions(state, player)
            if len(legal_moves) == 0:
                draws += 1
                print(f"Draw (Stalemate at step {step})")
                break
                
            active_net = new_net if player == new_plays_as else old_net
            mcts = MCTS(game, state, player, active_net)
            
            with torch.no_grad():
                mcts.iter(sims)
            
            best_move, _ = mcts.output()
            state, reward = game.next_state_and_reward(player, state, best_move)
            
            if reward != 0:
                if player == new_plays_as:
                    new_wins += 1
                    print(f"New Model WINS (Checkmate at step {step})")
                else:
                    old_wins += 1
                    print(f"Historical Model WINS (Checkmate at step {step})")
                break
                
            if step >= MAX_GAME_STEPS:
                draws += 1
                print(f"Draw (Turn limit at step {step})")
                break
            
            player = 3 - player
            
    return new_wins, old_wins, draws


def play_game(shared_net, game_path, worker_id):
    torch.set_num_threads(1)
    game = Chess(game_path)
    state = game.initial_state()
    player = 1
    dataset, step = [], 0
    shared_net.eval()
    
    while True:
        step += 1
        legal_moves = game.actions(state, player)
        if len(legal_moves) == 0: 
            print(f"  [Worker {worker_id}] Game finished at step {step} (Draw by stalemate).", flush=True)
            return assign_rewards(dataset, 0, player)
            
        mcts = MCTS(game, state, player, shared_net)
        with torch.no_grad(): 
            mcts.iter(MCTS_SIMS)
            
        _, local_policy = mcts.output()
        
        if step < 8:
            local_policy = local_policy / np.sum(local_policy)
            chosen_idx = np.random.choice(len(local_policy), p=local_policy)
            best_move = mcts.root.possible_moves[chosen_idx]
        else:
            chosen_idx = np.argmax(local_policy)
            best_move = mcts.root.possible_moves[chosen_idx]
            
        full_policy = np.zeros(1225, dtype=np.float32)
        for idx, m in enumerate(mcts.root.possible_moves):
            action_idx = game.action_to_index(m)
            full_policy[action_idx] = local_policy[idx]
            
        state_tensor = shared_net.state_to_tensor(state, player).squeeze(0)
        dataset.append([state_tensor, full_policy, player])
        
        state, reward = game.next_state_and_reward(player, state, best_move)
        
        if reward != 0: 
            print(f"  [Worker {worker_id}] Game finished at step {step} (Win/Loss).", flush=True)
            return assign_rewards(dataset, reward, player)
        if step >= MAX_GAME_STEPS: 
            print(f"  [Worker {worker_id}] Game finished at step {step} (Draw by limit).", flush=True)
            return assign_rewards(dataset, 0, player)
        player = 3 - player


def assign_rewards(dataset, final_reward, winning_player):
    training_data = []
    for state_tensor, full_policy, node_player in dataset:
        if final_reward == 0:
            value = -0.3
        else:
            value = 1.0 if node_player == winning_player else -1.0
        training_data.append((state_tensor, full_policy, np.array([value], dtype=np.float32)))
    return training_data


def train_network(net, replay_buffer, optimizer):
    if len(replay_buffer) < BATCH_SIZE: 
        print("    Not enough data in buffer to train yet.")
        return
        
    net.train()
    mse_loss = nn.MSELoss()
    steps_per_epoch = len(replay_buffer) // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            b_states, b_target_pol, b_target_val = zip(*batch)
            
            b_states = torch.stack(b_states)
            b_target_pol = torch.tensor(np.array(b_target_pol))
            b_target_val = torch.tensor(np.array(b_target_val))
            
            optimizer.zero_grad()
            out_val, out_pol = net(b_states)
            
            val_loss = mse_loss(out_val, b_target_val)
            pol_loss = -torch.sum(b_target_pol * torch.log(out_pol + 1e-8)) / BATCH_SIZE
            loss = val_loss + pol_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"    Epoch {epoch + 1}/{EPOCHS} | Avg Loss: {total_loss / max(1, steps_per_epoch):.4f}")


def main():
    print(f"Initializing Game and Network for CPU Multiprocessing ({NUM_WORKERS} workers)...")
    game_path = "chess_env/boards/szachy_plansza_5x5"
    game = Chess(game_path)
    
    net = AlphaZeroNetwork(17, 32, 5, 5, 1225)
    net.to("cpu")

    latest_file, start_episode = get_latest_checkpoint()
    if latest_file:
        print(f"\n[!] Found existing checkpoint: {latest_file}")
        print(f"[!] Resuming training from Episode {start_episode}")
        net.load_state_dict(torch.load(latest_file, map_location="cpu", weights_only=True))
    else:
        print("\n[!] No existing checkpoints found. Starting from scratch.")

    net.share_memory()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    target_total_episodes = start_episode + EPISODES

    for current_ep in range(start_episode + 1, target_total_episodes + 1):
        print(f"\n=== EPISODE {current_ep}/{target_total_episodes} ===")
        
        boss_weights = copy.deepcopy(net.state_dict())

        print(f"Phase 1: Self-Play (Generating {GAMES_PER_EPISODE} games across {NUM_WORKERS} cores)")
        new_data_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(play_game, net, game_path, i + 1) for i in range(GAMES_PER_EPISODE)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                replay_buffer.extend(future.result())
                new_data_count += len(future.result())
                print(f"  [Game {i + 1}/{GAMES_PER_EPISODE} complete] Buffer size: {len(replay_buffer)}/{BUFFER_SIZE}", end="\r")

        print(f"\n  Added {new_data_count} new moves to the replay buffer.")

        print("Phase 2: Training Network on Replay Buffer")
        train_network(net, replay_buffer, optimizer)

        print("\nPhase 3: Arena Evaluation (The Pit)")
        is_accepted = True
        
        for offset in ARENA_OPPONENTS_OFFSETS:
            target_opp = current_ep - offset
            hist_file = get_historical_checkpoint(target_opp)
            
            if not hist_file:
                target_opp = current_ep - 1
                hist_file = get_historical_checkpoint(target_opp)
            
            if hist_file:
                print(f"  [Evaluating against historical Boss from Episode {target_opp}]")
                old_weights = torch.load(hist_file, map_location="cpu", weights_only=True)
                wins, losses, draws = run_arena_match(game, net.state_dict(), old_weights, games=ARENA_GAMES_PER_OPPONENT, sims=ARENA_MCTS_SIMS)
                
                win_rate = 0.5 if (wins + losses) == 0 else wins / (wins + losses)
                print(f"  => Score against ep{target_opp}:")
                print(f"     New Model WINS: {wins} | Historical Model WINS: {losses} | DRAWS: {draws}")
                print(f"     Calculated Win Rate: {win_rate*100:.1f}% (Threshold is {WIN_RATE_THRESHOLD*100:.1f}%)")
                
                if win_rate <= WIN_RATE_THRESHOLD:
                    print(f"  [!] Failed. New model cannot efficiently beat Episode {target_opp} tactics.")
                    is_accepted = False
                    break
            else:
                print(f"  [!] History for Episode {target_opp} not deep enough. Skipping Arena Match.")

        if is_accepted:
             print(f"\n🏆 ARENA RESULT: ACCEPTED. Model has successfully advanced!")
             save_path = f"model/alphazero_model_ep{current_ep}.pth"
             torch.save(net.state_dict(), save_path)
             print(f"Saved checkpoint: {save_path}")
        else:
             print(f"\n❌ ARENA RESULT: REJECTED. Model fell into Catastrophic Forgetting. Reverting to Episode {current_ep-1} weights.")
             net.load_state_dict(boss_weights)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()