import os
import random
import numpy as np
from collections import deque
import concurrent.futures

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
GAMES_PER_EPISODE = 20  # Lowered: We use a buffer now, so we need fewer new games per loop
MCTS_SIMS = 75  # Lowered: Saves CPU time. The buffer compensates for lower quality.
EPOCHS = 5  # Increased: Train harder on the data we have
BATCH_SIZE = 32  # Small batch size is perfect for CPU
MAX_GAME_STEPS = 100
BUFFER_SIZE = 10000  # The network will remember the last 10,000 moves
NUM_WORKERS = max(1, os.cpu_count() - 2)  # Leave 2 cores free so your PC doesn't freeze


# ==========================================

def play_game(shared_net, game_path):
    """
    Worker function to play a single game.
    It instantiates its own game environment to prevent multi-threading collisions.
    """
    game = Chess(game_path)
    state = game.initial_state()
    player = 1
    dataset = []
    step = 0

    shared_net.eval()

    while True:
        step += 1
        legal_moves = game.actions(state, player)
        if len(legal_moves) == 0:
            return assign_rewards(dataset, 0, player)

        # MCTS uses the shared network across CPU cores
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
            return assign_rewards(dataset, reward, player)
        if step >= MAX_GAME_STEPS:
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

    # Calculate how many batches make up one "epoch" of our current buffer size
    steps_per_epoch = len(replay_buffer) // BATCH_SIZE

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            # 1. Randomly sample a batch from the past 10,000 moves
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
    #change for relative path
    game_path = "/home/babrzej/Documents/AlphaZero-nonstandard-chess/chess_env/boards/szachy_plansza_5x5"

    # Initialize network and force it to CPU
    net = AlphaZeroNetwork(17, 32, 5, 5, 1225)
    net.to("cpu")

    # CRITICAL: Share memory so worker processes can read the weights without copying
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # The new Replay Buffer
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    for episode in range(EPISODES):
        print(f"\n=== EPISODE {episode + 1}/{EPISODES} ===")

        # 1. Parallel Self Play Phase
        print(f"Phase 1: Self-Play (Generating {GAMES_PER_EPISODE} games across {NUM_WORKERS} cores)")

        new_data_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all games to the worker pool
            futures = [executor.submit(play_game, net, game_path) for _ in range(GAMES_PER_EPISODE)]

            # As each game finishes, add its data to the replay buffer
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                game_data = future.result()
                replay_buffer.extend(game_data)
                new_data_count += len(game_data)
                print(f"  [Game {i + 1}/{GAMES_PER_EPISODE} complete] Buffer size: {len(replay_buffer)}/{BUFFER_SIZE}",
                      end="\r")

        print(f"\n  Added {new_data_count} new moves to the replay buffer.")

        # 2. Training Phase
        print("Phase 2: Training Network on Replay Buffer")
        train_network(net, replay_buffer, optimizer)

        # 3. Save Checkpoint
        torch.save(net.state_dict(), f"alphazero_model_ep{episode + 1}.pth")
        print(f"Saved checkpoint: alphazero_model_ep{episode + 1}.pth")

    print("\nTRAINING COMPLETE!")


if __name__ == "__main__":
    # Required for PyTorch multiprocessing to work correctly on Linux
    mp.set_start_method('spawn', force=True)
    main()