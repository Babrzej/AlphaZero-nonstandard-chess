# Tu prawie nic nie ma!!!

class Node:
    def __init__(self, state = None, player = None, action = None, parent = None):
        self.action = action
        self.state = state
        self.player = player
        self.value = 0
        self.visits = 0
        self.children = []
        self.parent = parent
        self.possible_moves = []
        self.move_id = 0
        self.terminal = False

    def __repr__(self):
        # Format the action for readability (e.g., if it's a list or tuple)
        action_str = str(self.action) if self.action is not None else "Root"

        # Calculate avg_value dynamically if it's not being updated elsewhere
        avg = self.value / self.visits if self.visits > 0 else 0.0

        return (f"<Node | Player: {self.player} | Action: {action_str} | "
                    f"Visits: {self.visits} | AvgVal: {avg:.3f} | Children: {len(self.children)}>")


    def visited(self):
       return self.visits != 0

