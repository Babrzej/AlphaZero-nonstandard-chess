import os
import random
import numpy as np
import glob
import re
from collections import deque
import concurrent.futures

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from src.model.neural_network import AlphaZeroNetwork
from src.model.mcts import MCTS
from src.game_logic.chess_game import Chess

# HYPERPARAMETERS
EPISODES = 20
GAMES_PER_EPISODE = 64
MCTS_SIMS = 180
EPOCHS = 5
BATCH_SIZE = 64
MAX_GAME_STEPS = 100
BUFFER_SIZE = 25000
NUM_WORKERS = max(1, os.cpu_count() - 2)


def get_latest_checkpoint():
    checkpoints = glob.glob("model/alphazero_model_ep20.pth")
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


def play_game(net, game_path, worker_id):
    torch.set_num_threads(1)

    game = Chess(game_path)
    state = game.initial_state()
    player = 1
    dataset = []
    step = 0

    net.eval()

    while True:
        step += 1

        legal_moves = game.actions(state, player)
        if len(legal_moves) == 0:
            return assign_rewards(dataset, 0, player)

        mcts = MCTS(game, state, player, net)
        with torch.no_grad():
            mcts.iter(MCTS_SIMS)

        _, local_policy = mcts.output()

        if step < 8:
            # Temperature = 1
            local_policy = local_policy / np.sum(local_policy)
            chosen_id = np.random.choice(len(local_policy), p=local_policy)
            best_move = mcts.root.possible_moves[chosen_id]
        else:
            # Temperature = 0
            chosen_id = np.argmax(local_policy)
            best_move = mcts.root.possible_moves[chosen_id]

        full_policy = np.zeros(1225, dtype=np.float32)
        for id, m in enumerate(mcts.root.possible_moves):
            action_id = game.action_to_index(m)
            full_policy[action_id] = local_policy[id]

        state_tensor = net.state_to_tensor(state, player).squeeze(0)
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
        print("Not enough data in buffer to train yet.")
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
    game_path = "boards/szachy_plansza_5x5"

    net = AlphaZeroNetwork(17, 32, 5, 5, 1225)
    net.to("cpu")

    latest_file, start_episode = get_latest_checkpoint()
    if latest_file:
        print(f"\n[!] Found existing checkpoint: {latest_file}")
        print(f"[!] Resuming training from Episode {start_episode + 1}")
        net.load_state_dict(torch.load(latest_file, map_location="cpu", weights_only=True))
    else:
        print("\n[!] No existing checkpoints found. Starting from scratch (Episode 1).")

    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
    replay_buffer = deque(maxlen=BUFFER_SIZE)

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