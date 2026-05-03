import torch
from neural_network import AlphaZeroNetwork

bot = AlphaZeroNetwork(in_channels=17, out_channels=64, board_height=8, board_width=8, num_actions=4096)

random_board = torch.randn(1, 17, 8, 8)

bot.eval()
with torch.no_grad():
    value, policy = bot(random_board)

print("Win probability:", value.item())
print("Number of evaluated moves:", policy.shape[1])
