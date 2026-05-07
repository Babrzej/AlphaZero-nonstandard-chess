from sre_parse import State

import torch
import numpy as np
class ConvBatchBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBatchBlock, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, input):
        return self.model(input)

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.model = torch.nn.Sequential(
            ConvBatchBlock(in_channels, out_channels),
            torch.nn.ReLU(),
            ConvBatchBlock(out_channels, in_channels),
        )

    def forward(self, input):
        output = self.model(input)
        output += input
        output = torch.nn.functional.relu(output)
        return output
    
class DropoutBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, p=0.5):
        super(DropoutBlock, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p)
        )

    def forward(self, input):
        return self.model(input)

class AlphaZeroNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels, board_height, board_width, num_actions, res_blocks=5, p=0.3):      
        super(AlphaZeroNetwork, self).__init__()
        self.flatt_size = out_channels * board_height * board_width
        self.model = torch.nn.Sequential(
            ConvBatchBlock(in_channels=in_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            * [ResNetBlock(in_channels=out_channels, out_channels=out_channels) for _ in range(res_blocks)],
            torch.nn.Flatten(),
            * [DropoutBlock(in_channels=self.flatt_size, out_channels=self.flatt_size, p=p) for _ in range(2)]
        )
        
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.flatt_size, self.flatt_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.flatt_size, 1),
            torch.nn.Tanh()
        )
        
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(self.flatt_size, self.flatt_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.flatt_size, num_actions),
            torch.nn.Softmax(dim=1)
        )
        
    def state_to_tensor(self, State, player):
        board = State.Board
        x = np.zeros((2, 3, 3), dtype=np.float32)
        x[0] = (board == player).astype(np.float32)
        x[1] = (board == (3 - player)).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0)
    
    def forward(self, state, player):
        if isinstance(state, torch.Tensor):
            input = state
        else:
            input = self.state_to_tensor(state, player)
        features = self.model(input)
        value = self.value_head(features)
        policy = self.policy_head(features)
        return value, policy
   
    
    


