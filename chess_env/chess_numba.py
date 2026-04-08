import numpy as np
import numba as nb
from numpy.typing import NDArray

Board2D = NDArray[np.int32]
Moves1D = NDArray[np.int32]

# --- 1. GAME CONSTANTS ---
EMPTY = 0 # Empty square is represented by 0
# We represent pieces as integers: 1-6 for White and 1001-1006 for Black
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

# Promotion: We add 100-300 to the move index to indicate a specific promotion
QUEEN_PROMO = 0
KNIGHT_PROMO = 100
ROOK_PROMO = 200
BISHOP_PROMO = 300

BLACK_SHIFT = 1000  # We add 1000 to identify a black piece
BOARD_SIZE = 8 

# Players are represented as 1 (White) and 2 (Black)
WHITE = 1
BLACK = 2

# --- 2. FAST HELPER FUNCTIONS ---
@nb.njit
def is_on_board(r: int, c: int) -> bool:
    """Checks if the (r, c) coordinates are inside the {BOARD_SIZE} x {BOARD_SIZE} board."""
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

# --- 3. PIECE-SPECIFIC MOVES ---
@nb.njit
def get_pawn_moves(board: Board2D, r: int, c: int, player: int) -> Moves1D:
    """
    Generates all legal moves for a pawn at (r, c).
    Includes double-step, captures, promotion and en passant.
    Board is a 2D NumPy array where pieces are integers. Player is 1 (White) or 2 (Black).
    """
    
    # Pawn have different move allowed depending on placement on the board and enemy pieces.
    # Maximum number of moves is 12 in promotion scenarios (4 types of promotion * 3 possible moves: forward, capture left, capture right)
    moves = np.zeros(12, dtype=np.int32)
    move_count = 0
    
    # Determine enemy piece range based on player and info important for pawn movement
    if player == WHITE: # White (moving UP the board, towards row 0)
        direction = -1
        start_row = BOARD_SIZE - 2
        promo_row = 0
        enemy_min, enemy_max = 1 + BLACK_SHIFT, 6 + BLACK_SHIFT
    else:           # Black (moving DOWN the board, towards row BOARD_SIZE-1)
        direction = 1
        start_row = 1
        promo_row = BOARD_SIZE - 1
        enemy_min, enemy_max = 1, 6

    # Forward type of movement (single step (with promotion or not) or double step)
    next_r = r + direction
    if is_on_board(next_r, c) and board[next_r, c] == EMPTY:
        target_idx = next_r * BOARD_SIZE + c
        
        if next_r == promo_row:
            # Promotion options for forward move
            moves[move_count]     = QUEEN_PROMO
            moves[move_count + 1] = KNIGHT_PROMO
            moves[move_count + 2] = ROOK_PROMO
            moves[move_count + 3] = BISHOP_PROMO
            move_count += 4
        else:
            # Standard single step
            moves[move_count] = target_idx
            move_count += 1
            
            # Double step from starting position (only if path is clear)
            if r == start_row:
                double_r = r + 2 * direction
                if board[double_r, c] == EMPTY:
                    target_idx = double_r * BOARD_SIZE + c
                    moves[move_count] = target_idx
                    move_count += 1

    # 4. Diagonal types of movement (captures and en passant)
    for dc in [-1, 1]:
        next_c = c + dc
        if is_on_board(next_r, next_c):
            target_piece = board[next_r, next_c]
            target_idx = next_r * BOARD_SIZE + next_c
            
            # Check for enemy piece OR En Passant target square
            if (enemy_min <= target_piece <= enemy_max) or (target_idx == ep_sq):
                
                if next_r == promo_row:
                    # Promotion options via capture
                    moves[move_count]     = QUEEN_PROMO
                    moves[move_count + 1] = KNIGHT_PROMO
                    moves[move_count + 2] = ROOK_PROMO
                    moves[move_count + 3] = BISHOP_PROMO
                    move_count += 4
                else:
                    # Standard diagonal capture
                    moves[move_count] = target_idx
                    move_count += 1

    # 5. Return only the populated portion of the array
    return moves[:move_count]

@nb.njit
def get_knight_moves(board: Board2D, r: int, c: int, player: int) -> Moves1D:
    """
    Generates all legal moves for a knight at (r, c).
    Board is a 2D NumPy array where pieces are integers. Player is 1 (White) or 2 (Black).
    """
    # Knight ALWAYS has a max of 8 moves, regardless of board size.
    moves = np.zeros(8, dtype=np.int32)
    move_count = 0
    
    # Determine enemy piece range based on player
    if player == WHITE: 
        enemy_min, enemy_max = 1 + BLACK_SHIFT, 6 + BLACK_SHIFT 
    else:           
        enemy_min, enemy_max = 1, 6

    # Possible knight jumps (row cordinates and column coordinates)
    jumps_r = np.array([-2, -2, -1, -1, 1, 1, 2, 2], dtype=np.int32)
    jumps_c = np.array([-1, 1, -2, 2, -2, 2, -1, 1], dtype=np.int32)
    
    # Check each of the 8 possible knight moves
    for i in range(8):
        new_r = r + jumps_r[i]
        new_c = c + jumps_c[i]
        
        if is_on_board(new_r, new_c):
            target_piece = board[new_r, new_c] # board[new_r, new_c] is the piece at the target square
            
            if target_piece == EMPTY or (enemy_min <= target_piece <= enemy_max):
                
                # Assign the move as a legal move in 1D format
                target_idx = new_r * BOARD_SIZE + new_c
                moves[move_count] = target_idx
                move_count += 1
    
    # Return only the valid moves found, [: move_count] to avoid trailing zeros
    return moves[:move_count] 

@nb.njit
def get_bishop_moves(board: Board2D, r: int, c: int, player: int) -> Moves1D:
    """
    Generates all pseudo-legal moves for a bishop at (r, c).
    Board is a 2D NumPy array where pieces are integers. Player is 1 (White) or 2 (Black).
    """
    
# --- 4. TESTING ---
if __name__ == "__main__":
    # Create an empty 5x5 board
    test_board = np.zeros((5, 5), dtype=np.int32)
    
    # Place a White Knight dead center (row 2, col 2)
    test_board[2, 2] = KNIGHT
    
    # Place a Black Pawn on (0, 1) so the Knight can capture it
    test_board[1] = PAWN + BLACK_SHIFT
    
    # Place our own (White) Pawn on (0, 3) - the Knight CANNOT jump here!
    test_board[3] = PAWN 
    
    # Run the function!
    # First run compiles it
    possible_moves = get_knight_moves(test_board, 2, 2, 1)
    
    print("Knight is at [2, 2].")
    print("Found legal moves to squares (encoded as RowCol):", possible_moves)