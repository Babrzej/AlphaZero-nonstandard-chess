import os
import random
import numpy as np
import glob
import re
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
# CPU-OPTIMIZED HYPERPARAMETERS
# ==========================================
EPISODES = 50
GAMES_PER_EPISODE = 20
MCTS_SIMS = 75
EPOCHS = 5
BATCH_SIZE = 32
MAX_GAME_STEPS = 50
BUFFER_SIZE = 10000
NUM_WORKERS = max(1, os.cpu_count() - 2)


# ==========================================

def get_latest_checkpoint():
    """Scans the directory for the newest model checkpoint and returns its path and episode number."""
    checkpoints = glob.glob("alphazero_model_ep*.pth")
    if not checkpoints:
        return None, 0

    latest_ep = 0
    latest_file = ""

    for cp in checkpoints:
        # Extract the number from the filename using regex
        match = re.search(r"alphazero_model_ep(\d+)\.pth", cp)
        if match:
            ep_num = int(match.group(1))
            if ep_num > latest_ep:
                latest_ep = ep_num
                latest_file = cp

    return latest_file, latest_ep


def play_game(shared_net, game_path, worker_id):
    """Worker function to play a single game."""
    torch.set_num_threads(1)

    game = Chess(game_path)
    state = game.initial_state()
    player = 1
    dataset = []
    step = 0

    shared_net.eval()

    while True:
        step += 1

        # Heartbeat to show progress
        if step % 10 == 0:
            print(f"  [Worker {worker_id}] playing move {step}...", flush=True)

        legal_moves = game.actions(state, player)
        if len(legal_moves) == 0:
            return assign_rewards(dataset, 0, player)

        mcts = MCTS(game, state, player, shared_net)
        with torch.no_grad():
            mcts.iter(MCTS_SIMS)

        best_move, local_policy = mcts.output()

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
            value = 0.0
        else:
            value = 1.0 if node_player == winning_player else -1.0
        training_data.append((state_tensor, full_policy, np.array([value], dtype=np.float32)))
    return training_data


def train_network(net, replay_buffer, optimizer):
    """Trains the network by sampling random batches from the historical replay buffer."""
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
    game_path = "/home/babrzej/Documents/AlphaZero-nonstandard-chess/chess_env/boards/szachy_plansza_5x5"

    net = AlphaZeroNetwork(17, 32, 5, 5, 1225)
    net.to("cpu")

    # --- AUTO-RESUME LOGIC ---
    latest_file, start_episode = get_latest_checkpoint()
    if latest_file:
        print(f"\n[!] Found existing checkpoint: {latest_file}")
        print(f"[!] Resuming training from Episode {start_episode + 1}")
        # Map location to CPU to prevent CUDA mismatch errors
        net.load_state_dict(torch.load(latest_file, map_location="cpu", weights_only=True))
    else:
        print("\n[!] No existing checkpoints found. Starting from scratch (Episode 1).")

    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    # Shift the loop range to account for the loaded episode
    target_total_episodes = start_episode + EPISODES

    for current_ep in range(start_episode + 1, target_total_episodes + 1):
        print(f"\n=== EPISODE {current_ep}/{target_total_episodes} ===")

        print(f"Phase 1: Self-Play (Generating {GAMES_PER_EPISODE} games across {NUM_WORKERS} cores)")

        new_data_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(play_game, net, game_path, i + 1) for i in range(GAMES_PER_EPISODE)]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                game_data = future.result()
                replay_buffer.extend(game_data)
                new_data_count += len(game_data)
                print(f"  [Game {i + 1}/{GAMES_PER_EPISODE} complete] Buffer size: {len(replay_buffer)}/{BUFFER_SIZE}",
                      end="\r")

        print(f"\n  Added {new_data_count} new moves to the replay buffer.")

        print("Phase 2: Training Network on Replay Buffer")
        train_network(net, replay_buffer, optimizer)

        torch.save(net.state_dict(), f"alphazero_model_ep{current_ep}.pth")
        print(f"Saved checkpoint: alphazero_model_ep{current_ep}.pth")

    print("\nTRAINING COMPLETE!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()