import numpy as np

# Constant for the UCB calculation
c_ucb = np.sqrt(2)

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


class MCTS:
    def __init__(self, game, state, player, net):
        #TODO: analyze all needed parameters
        self.game = game
        self.root = Node(state = state, player = player)
        self.root.possible_moves = self.game.actions(state, player)
        self.root.children = np.empty(len(self.root.possible_moves), dtype = object)
        self.net = net

    def __repr__(self):
        # Safely get the class names for the game and neural net,
        # falling back to "Unknown" if they are missing.
        game_name = self.game.__class__.__name__ if self.game else "Unknown"
        net_name = self.net.__class__.__name__ if self.net else "Unknown"

        # The total iterations the MCTS has run is tracked by the root's visits
        total_simulations = self.root.visits if self.root else 0

        return f"<MCTS Engine | Game: {game_name} | Net: {net_name} | Total Simulations: {total_simulations}>"

    def select(self, node):
        #all allowed moves with their ucbs and their policies
        if node.terminal:
            return -1, node

        #policy = self.net.get_policy(node.state)
        #---------------------------------------------------------------------------------
        raw_policy = np.random.random(len(node.possible_moves))
        policy = raw_policy / np.sum(raw_policy)
        #---------------------------------------------------------------------------------
        values = np.empty((len(node.possible_moves), 2), dtype = float)
        values[:] = np.array([
            (c.value, c.visits) if isinstance(c, Node) else (0.0, 0.0)
            for c in node.children
        ]).reshape(-1, 2)
        ucb = calc_ucb(values, node.visits, policy)
        best = np.argmax(ucb)

        if isinstance(node.children[best], Node):
            return self.select(node.children[best])
        else:
            return best, node

    #Create node for state visited first time
    def expand(self, move, node):
        state, reward = self.game.next_state_and_reward(node.player, node.state, node.possible_moves[move])
        child = Node(action=node.possible_moves[move], state=state, parent=node, player=3 - node.player)
        child.possible_moves = self.game.actions(state, 3 - node.player)
        child.children = np.empty(len(child.possible_moves), dtype = object)
        if reward != 0:
            child.terminal = True
        node.children[move] = child
        return child, reward

    def backpropagate(self, node, value):
        if node is None:
            return

        node.visits += 1
        node.value += value
        self.backpropagate(node.parent, -value)

    def print_tree(self, node=None, depth=0, max_depth=3):
        # Default to the root node if no node is provided
        if node is None:
            node = self.root
            print(f"\n{'='*80}")
            print(f"{'MCTS TREE VISUALIZATION':^80}")
            print(f"{'='*80}\n")

        # ANSI color codes
        COLORS = {
            'root': '\033[95m',      # Magenta
            'good': '\033[92m',      # Green
            'mid': '\033[93m',       # Yellow
            'poor': '\033[91m',      # Red
            'info': '\033[94m',      # Blue
            'reset': '\033[0m',
            'bold': '\033[1m',
        }

        # Add indentation based on depth
        indent = "│   " * depth
        prefix = "├── " if depth > 0 else "🌳 "

        # Calculate win rate
        win_rate = (node.value / node.visits * 100) if node.visits > 0 else 0

        # Color based on win rate
        if node.visits == 0:
            color = COLORS['poor']
        elif win_rate > 50:
            color = COLORS['good']
        elif win_rate > 0:
            color = COLORS['mid']
        else:
            color = COLORS['poor']

        # Format action
        if node.action is None:
            action_str = "ROOT"
            icon = "🎯"
        else:
            action_str = f"[{node.action[0]},{node.action[1]}]"
            icon = "🎲" if node.visits > 50 else "○"

        # Terminal node indicator
        terminal_str = " ✓ WIN" if node.terminal and node.value > 0 else " ✗ LOSS" if node.terminal else ""

        # Build the line
        line = (f"{indent}{prefix}{COLORS['bold']}{icon} {action_str:8}{COLORS['reset']} "
                f"{color}▮{COLORS['reset']} "
                f"{COLORS['info']}V:{node.visits:4d}{COLORS['reset']} "
                f"{color}Q:{win_rate:+6.1f}%{COLORS['reset']}"
                f"{terminal_str}")

        print(line)

        # Base case: Stop if we reach maximum depth
        if depth >= max_depth:
            return

        # Filter out unexpanded children
        valid_children = [c for c in node.children if isinstance(c, object) and hasattr(c, 'visits')]

        if not valid_children:
            return

        # Sort children by visits
        sorted_children = sorted(valid_children, key=lambda c: c.visits, reverse=True)

        for i, child in enumerate(sorted_children):
            is_last = (i == len(sorted_children) - 1)
            self.print_tree(child, depth + 1, max_depth)

    def print_tree_compact(self):
        """Wyświetl statystyki w formacie tabelarycznym"""
        print(f"\n{'='*70}")
        print(f"{'MCTS STATISTICS':^70}")
        print(f"{'='*70}")
        print(f"{'Move':<12} {'Visits':<12} {'Value':<12} {'Q-Value':<12} {'Best?':<12}")
        print(f"{'-'*70}")

        # Sortuj dzieci po wizytach
        valid_children = [c for c in self.root.children if isinstance(c, object) and hasattr(c, 'visits')]
        sorted_children = sorted(valid_children, key=lambda c: c.visits, reverse=True)

        max_visits = max([c.visits for c in sorted_children]) if sorted_children else 1

        for i, child in enumerate(sorted_children):
            q_value = (child.value / child.visits) if child.visits > 0 else 0
            is_best = "★ BEST" if i == 0 else ""
            bar = "█" * int(20 * child.visits / max_visits) if max_visits > 0 else ""

            print(f"{str(child.action):<12} {child.visits:<12} {child.value:<12.2f} {q_value:<12.3f} {is_best:<12}")
            print(f"  {bar:<20}")

        print(f"{'='*70}\n")

    def iter(self, num_of_iterations):
        for i in range(num_of_iterations):
            move, parent = self.select(self.root)

            if move == -1:
                value = parent.value
                self.backpropagate(parent, value)
            else:
                # nn sim
                value = np.random.random() * 2 - 1
                new_node, reward = self.expand(move, parent)
                if reward != 0:
                    self.backpropagate(new_node, reward)
                else:
                    self.backpropagate(new_node, value)

    def output(self):
        visits = np.zeros(len(self.root.children))
        visits[:] = [
            c.visits if isinstance(c, Node) else 0
            for c in self.root.children
        ]
        best = np.argmax(visits)
        policy = visits / np.sum(visits)
        best_move = self.root.possible_moves[best]
        return best_move, policy



def calc_ucb(values, visits_parent, policy, c = c_ucb):
    value = values[:,0]
    visits = values[:,1]
    q = np.divide(value, visits, out=np.zeros_like(value), where=visits != 0)
    p = c * policy * np.sqrt(visits_parent) / (1 + visits)
    ucb = q + p
    return ucb

