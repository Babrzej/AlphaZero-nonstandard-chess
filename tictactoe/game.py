import numpy as np

class TicTacToe():
    def __init__(self):
        # Board dimensions to match the interface attributes
        self.num_of_rows = 3
        self.num_of_columns = 3

    class TicTacToeState():
        def __init__(self, _board=None):
            # If no board is provided, create an empty 3x3 board
            if _board is None:
                self.Board = np.zeros([3, 3], dtype=int)
            else:
                self.Board = _board

    def initial_state(self):
        # Returns the initial empty board state
        return self.TicTacToeState()

    def state_key(self, State):
        # Returns a string representation of the board (useful for MCTS dictionaries)
        return str(State.Board)

    def actions(self, State, player):
        # Finds all empty squares (represented by 0) and returns them as legal moves
        moves_potential = []
        for r in range(self.num_of_rows):
            for c in range(self.num_of_columns):
                if State.Board[r, c] == 0:
                    moves_potential.append([r, c])
        return moves_potential

    def next_state_and_reward(self, player, State, action):
        # Create a deep copy of the board to prevent mutating the parent state
        NextState = self.TicTacToeState(np.copy(State.Board))
        
        # Apply the move
        r, c = action
        NextState.Board[r, c] = player

        Reward = 0
        if self.__check_win(NextState.Board, player):
            Reward = 1 

        return NextState, Reward

    def end_of_game(self, _R=0, _number_of_moves=0, State=None, _action_nr=0):
        if np.abs(_R) > 0.95:
            return True
        
        if not np.any(State.Board == 0):
            return True
            
        return False

    def action_to_string(self, action):
        return f"Row:{action} Col:{action[1]}"

    def __check_win(self, board, player):
        for i in range(3):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return True
        # Private method: check both diagonals
        if board[0,0] == player and board[1, 1] == player and board[2, 2] == player:
            return True
        if board[0,2] == player and board[1, 1] == player and board[2, 0] == player:
            return True
            
        return False
        
    def print_board(self, State):
        # Helper method to visualize the board in terminal
        symbols = {0: ".", 1: "X", 2: "O"}
        for r in range(self.num_of_rows):
            row_str = " ".join([symbols[State.Board[r, c]] for c in range(self.num_of_columns)])
            print(row_str)
        print("-" * 5)