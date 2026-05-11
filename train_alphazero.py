import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model.neural_network import AlphaZeroNetwork
from model.mcts import MCTS
from chess_env.chess_game import Chess

# These are extremely low so the script finishes in ~1 minute on a CPU.
# Once you verify it works, increase these for real training!
EPISODES = 1
GAMES_PER_EPISODE = 2
MCTS_SIMS = 10
EPOCHS = 2
BATCH_SIZE = 32
MAX_GAME_STEPS = 100  # Prevents infinite games


def play_game(game, net):
    """Plays one full game of self-play and returns the generated data."""
    state = game.initial_state()
    player = 1
    dataset = []
    step = 0

    net.eval()  # CRITICAL: Eval mode for self-play

    while True:
        step += 1

        # 1. Run MCTS
        mcts = MCTS(game, state, player, net)
        with torch.no_grad():  # CRITICAL: Don't fill RAM with gradients
            mcts.iter(MCTS_SIMS)

        best_move, local_policy = mcts.output()

        # 2. Map local policy back to 1225 full policy
        full_policy = np.zeros(1225, dtype=np.float32)
        for idx, m in enumerate(mcts.root.possible_moves):
            action_idx = game.action_to_index(m)
            full_policy[action_idx] = local_policy[idx]

        # 3. Store the state tensor, the full target policy, and the current player
        state_tensor = net.state_to_tensor(state, player).squeeze(0)  # Remove batch dim for now
        dataset.append([state_tensor, full_policy, player])

        # 4. Make the move
        state, reward = game.next_state_and_reward(player, state, best_move)

        # 5. Check Terminal states (Win/Loss/Draw/Turn Limit)
        if reward != 0:
            return assign_rewards(dataset, reward, player)
        if step >= MAX_GAME_STEPS:
            return assign_rewards(dataset, 0, player)  # Draw by limit

        player = 3 - player  # Swap player


def assign_rewards(dataset, final_reward, winning_player):
    """Assigns the final +1, -1, or 0 back to every move in the game."""
    training_data = []
    for state_tensor, full_policy, node_player in dataset:
        # If it's a draw, value is 0. If this node's player won, +1. Else -1.
        if final_reward == 0:
            value = 0.0
        else:
            value = 1.0 if node_player == winning_player else -1.0

        training_data.append((state_tensor, full_policy, np.array([value], dtype=np.float32)))
    return training_data


def train_network(net, dataset, optimizer):
    """Trains the network on the gathered data."""
    net.train()  # CRITICAL: Switch to training mode
    device = next(net.parameters()).device

    # Unpack dataset
    states, policies, values = zip(*dataset)
    states = torch.stack(states).to(device)
    target_policies = torch.tensor(np.array(policies)).to(device)
    target_values = torch.tensor(np.array(values)).to(device)

    dataset_size = len(states)
    indices = np.arange(dataset_size)

    # Standard AlphaZero Loss functions
    mse_loss = nn.MSELoss()

    for epoch in range(EPOCHS):
        np.random.shuffle(indices)
        total_loss = 0.0

        for i in range(0, dataset_size, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]

            # --- THE BATCH NORM FIX ---
            if len(batch_idx) == 1:
                continue

            b_states = states[batch_idx]
            b_target_pol = target_policies[batch_idx]
            b_target_val = target_values[batch_idx]

            optimizer.zero_grad()

            # Forward pass
            out_val, out_pol = net(b_states)

            # Calculate Loss
            val_loss = mse_loss(out_val, b_target_val)
            # Custom Cross Entropy for probabilities
            pol_loss = -torch.sum(b_target_pol * torch.log(out_pol + 1e-8)) / len(batch_idx)

            loss = val_loss + pol_loss

            # Backpropagate
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"    Epoch {epoch + 1}/{EPOCHS} | Avg Loss: {total_loss / max(1, (dataset_size // BATCH_SIZE)):.4f}")


def main():
    print("Initializing Game and Network...")
    game = Chess("chess_env/boards/szachy_plansza_5x5")
    net = AlphaZeroNetwork(17, 32, 5, 5, 1225)

    # Try to use GPU if ROCm/CUDA is available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    for episode in range(EPISODES):
        print(f"\n=== EPISODE {episode + 1}/{EPISODES} ===")
        dataset = []

        # 1. Self Play Phase
        print("Phase 1: Self-Play (Data Generation)")
        for g in range(GAMES_PER_EPISODE):
            print(f"  Playing Game {g + 1}/{GAMES_PER_EPISODE}...", end="\r")
            game_data = play_game(game, net)
            dataset.extend(game_data)
        print(f"\n  Generated {len(dataset)} total moves of data.")

        # 2. Training Phase
        print("Phase 2: Training Network")
        train_network(net, dataset, optimizer)

        # 3. Save Checkpoint
        torch.save(net.state_dict(), f"alphazero_model_ep{episode + 1}.pth")
        print(f"Saved checkpoint: alphazero_model_ep{episode + 1}.pth")

    print("\nTRAINING COMPLETE!")


if __name__ == "__main__":
    main()