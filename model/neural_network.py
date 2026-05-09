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

    def state_to_tensor(self, state, player):
        board = state.Board
        # 12 for figures, 1 for turn, 4 for castling, en pasą ?, 50 moves ??
        num_channels = 17
        num_rows = board.shape[0]
        num_cols = board.shape[1]
        tensor = np.zeros((num_channels, num_rows, num_cols))

        # white pieces
        for piece in range(1, 7):
            tensor[piece - 1] = (board == piece).astype(np.float32)

        # black pieces
        shift = 1000
        for piece in range(1, 7):
            tensor[piece + 5] = (board == piece + shift).astype(np.float32)

        # turn
        if player == 1:
            tensor[12] = np.ones((num_rows, num_cols))

        # castlings
        if state.white_small_castling_possible:
            tensor[13] = np.ones((num_rows, num_cols))
        if state.black_small_castling_possible:
            tensor[14] = np.ones((num_rows, num_cols))
        if state.white_big_castling_possible:
            tensor[15] = np.ones((num_rows, num_cols))
        if state.black_big_castling_possible:
            tensor[16] = np.ones((num_rows, num_cols))

        return tensor

    def forward(self, state, player):
        if isinstance(state, torch.Tensor):
            input = state
        else:
            input = self.state_to_tensor(state, player)
        features = self.model(input)
        value = self.value_head(features)
        policy = self.policy_head(features)
        return value, policy
   
    
    


