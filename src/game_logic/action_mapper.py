class ActionMapper:
    def __init__(self, num_rows=5, num_cols=5):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_squares = num_rows * num_cols
        self.moves_per_square = 49
        self.total_actions = self.num_squares * self.moves_per_square

        self.KNIGHT = 2
        self.BISHOP = 3
        self.ROOK = 4
        self.QUEEN = 5
        self.BLACK_SHIFT = 1000

    def _get_base_piece(self, piece):
        return piece - self.BLACK_SHIFT if piece > self.BLACK_SHIFT else piece

    def action_to_index(self, move):
        piece, r1, c1, r2, c2 = move[0], move[1], move[2], move[3], move[4]
        promo = move[5] if len(move) == 6 else None

        square_idx = (r1 * self.num_cols) + c1
        dr, dc = r2 - r1, c2 - c1
        move_type = -1

        # 1. Underpromotions (40-48)
        if promo:
            base_promo = self._get_base_piece(promo)
            if base_promo != self.QUEEN:
                # Column offset: -1(left)=0, 0(straight)=1, 1(right)=2
                col_offset = dc + 1
                if base_promo == self.KNIGHT:
                    move_type = 40 + col_offset
                elif base_promo == self.BISHOP:
                    move_type = 43 + col_offset
                elif base_promo == self.ROOK:
                    move_type = 46 + col_offset

        # 2. Knight Moves (32-39)
        if move_type == -1 and abs(dr) * abs(dc) == 2:
            knight_map = {(-2, 1): 32, (-1, 2): 33, (1, 2): 34, (2, 1): 35, (2, -1): 36, (1, -2): 37, (-1, -2): 38,
                          (-2, -1): 39}
            move_type = knight_map.get((dr, dc), -1)

        # 3. Queen moves (0-31)
        if move_type == -1:
            dist = max(abs(dr), abs(dc))
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
            # Normalize direction to unit vector
            unit_dr = dr // dist if dr != 0 else 0
            unit_dc = dc // dist if dc != 0 else 0
            try:
                dir_idx = directions.index((unit_dr, unit_dc))
                move_type = (dir_idx * 4) + (dist - 1)
            except ValueError:
                return None

        return (square_idx * self.moves_per_square) + move_type

    def index_to_action(self, index, board_state_at_idx):
        square_idx = index // self.moves_per_square
        move_type = index % self.moves_per_square

        r1, c1 = square_idx // self.num_cols, square_idx % self.num_cols
        piece = board_state_at_idx[r1, c1]
        is_black = piece > self.BLACK_SHIFT
        shift = self.BLACK_SHIFT if is_black else 0

        # Reconstruct the move based on move_type
        if move_type < 32:  # Ray
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
            dir_idx = move_type // 4
            dist = (move_type % 4) + 1
            dr, dc = directions[dir_idx]
            r2, c2 = r1 + dr * dist, c1 + dc * dist
            # Check if this ray move is a queen promotion
            if self._get_base_piece(piece) == 1 and (r2 == 0 or r2 == 4):  # Pawn at edge
                return [piece, r1, c1, r2, c2, self.QUEEN + shift]
            return [piece, r1, c1, r2, c2]

        elif move_type < 40:  # Knight
            knight_map = {32: (-2, 1), 33: (-1, 2), 34: (1, 2), 35: (2, 1), 36: (2, -1), 37: (1, -2), 38: (-1, -2),
                          39: (-2, -1)}
            dr, dc = knight_map[move_type]
            return [piece, r1, c1, r1 + dr, c1 + dc]

        else:  # Underpromotion
            dr = -1 if not is_black else 1
            dc = ((move_type - 40) % 3) - 1
            promo_type = [self.KNIGHT, self.BISHOP, self.ROOK][(move_type - 40) // 3]
            return [piece, r1, c1, r1 + dr, c1 + dc, promo_type + shift]

