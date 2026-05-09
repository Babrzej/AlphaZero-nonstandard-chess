import os
import sys
import numpy as np
import torch
import torch.optim as optim
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_env.chess_game import Chess
from mcts import MCTS 
from neural_network import AlphaZeroNetwork


def select_move_from_policy(policy, step, temp_moves=3):
    if step <= temp_moves:
        return np.random.choice(len(policy), p=policy)
    return int(np.argmax(policy))


def play_game(game, net, mcts_simulations=50, temp_moves=3):
    state = game.initial_state()
    player = int(np.random.choice([1, 2]))
    history = []
    step = 0

    while True:
        step += 1
        mcts = MCTS(game, state, player, net)
        mcts.iter(mcts_simulations)

        best_move, policy = mcts.output()
        actions = mcts.root.possible_moves

        history.append([state, player, policy, actions])

        move_idx = select_move_from_policy(policy, step, temp_moves=temp_moves)
        chosen_move = actions[move_idx]

        state, reward = game.next_state_and_reward(player, state, chosen_move)

        if game.end_of_game(reward, step, state):
            training_data = []
            
            for ex_state, ex_player, ex_policy, ex_actions in history:
                if reward == 0:
                    val = 0.0
                else:
                    val = 1.0 if ex_player == player else -1.0
                
                full_policy = np.zeros(9, dtype=np.float32)
                for i, act in enumerate(ex_actions):
                    action_idx = act[0] * 3 + act[1]
                    full_policy[action_idx] = ex_policy[i]

                training_data.append((ex_state, ex_player, full_policy, val))
                
            return training_data
        
        player = 3 - player


def train_network(net, optimizer, training_data, batch_size=32, epochs=3):
    net.train()

    states = []
    target_policies = []
    target_values = []

    for state, player, policy, value in training_data:
        state_tensor = net.state_to_tensor(state, player)
        states.append(state_tensor[0])
        target_policies.append(policy)
        target_values.append([value])

    states = torch.stack(states)
    target_policies = torch.tensor(np.array(target_policies), dtype=torch.float32)
    target_values = torch.tensor(np.array(target_values), dtype=torch.float32)

    dataset_size = len(states)
    for _ in range(epochs):
        
        indices = np.random.permutation(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            b_states = states[batch_idx]
            b_target_pol = target_policies[batch_idx]
            b_target_val = target_values[batch_idx]

            optimizer.zero_grad()

            pred_values, pred_policies = net(b_states, player=None)

            policy_loss = -torch.sum(b_target_pol * torch.log(pred_policies + 1e-8)) / len(b_states)
            value_loss = torch.nn.functional.mse_loss(pred_values, b_target_val)

            total_loss = policy_loss + value_loss

            total_loss.backward()
            optimizer.step()
            
    print(f"Straty (Loss) - Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f}")


def main():
    game = TicTacToe()
    
    net = AlphaZeroNetwork(2, 64, 3, 3, 9)
    
    model_path = "alphazero_tic_tac_toe_gen150.pth"
    if os.path.exists(model_path):
        print(f"Znaleziono stary model: {model_path}. Wczytywanie dotychczasowej wiedzy...")
        net.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print("Nie znaleziono starego modelu. Zaczynamy trening od zera.")
    
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    episodes = 100 
    games_per_ep = 40
    mcts_sims = 500  
    replay_size = 20000
    train_sample_size = 4000
    temp_moves = 3
    replay_buffer = deque(maxlen=replay_size)

    print("="*50)
    print("Training start")
    print("="*50)

    for episode in range(episodes):
        print(f"\n--- Generating {episode+1}/{episodes} ---")
        
        net.eval()
        all_training_data = []

        print("Self-Play...", end="", flush=True)
        for g in range(games_per_ep):
            game_data = play_game(game, net, mcts_simulations=mcts_sims, temp_moves=temp_moves)
            all_training_data.extend(game_data)
            print(".", end="", flush=True)
        print(f" Gathered {len(all_training_data)} positions.")

        replay_buffer.extend(all_training_data)
        if len(replay_buffer) > train_sample_size:
            sample_idx = np.random.choice(len(replay_buffer), size=train_sample_size, replace=False)
            sampled_data = [replay_buffer[idx] for idx in sample_idx]
        else:
            sampled_data = list(replay_buffer)
        print(f"Bufor replay: {len(replay_buffer)} positions | Training on: {len(sampled_data)} positions")

        print("Training network...")
        train_network(net, optimizer, sampled_data, batch_size=32, epochs=5)
        
        if (episode + 1) % 10 == 0:
            torch.save(net.state_dict(), f"alphazero_tic_tac_toe_gen{episode+1}.pth")
            print(f"Saved checkpoint: alphazero_tic_tac_toe_gen{episode+1}.pth")

    print("\nTraining completed successfully!")
    torch.save(net.state_dict(), "alphazero_tic_tac_toe_final.pth")

if __name__ == "__main__":
    main()