import numpy as np
import chess_env.chess_game as bfun

def state_to_tensor(state, player, game):
    board = state.Board
    # 12 for figures, 1 for turn, 4 for castling, en pasą ?, 50 moves ??
    num_channels = 17
    num_rows = board.shape[0]
    num_cols = board.shape[1]
    tensor = np.zeros((num_channels, num_rows, num_cols))

    #white pieces
    for piece in range(1,7):
        tensor[piece - 1] = (board == piece).astype(np.float32)

    #black pieces
    shift = game.BlackShift
    for piece in range(1,7):
        tensor[piece + 5] = (board == piece+shift).astype(np.float32)

    #turn
    if player == 1:
        tensor[12] = np.ones((num_rows, num_cols))

    #castlings
    if state.white_small_castling_possible:
        tensor[13] = np.ones((num_rows, num_cols))
    if state.black_small_castling_possible:
        tensor[14] = np.ones((num_rows, num_cols))
    if state.white_big_castling_possible:
        tensor[15] = np.ones((num_rows, num_cols))
    if state.black_big_castling_possible:
        tensor[16] = np.ones((num_rows, num_cols))

    return tensor