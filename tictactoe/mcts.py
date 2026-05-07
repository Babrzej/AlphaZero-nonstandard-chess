import numpy as np
import torch

# Constant for the UCB calculation
c_ucb = np.sqrt(2)

class Node:
    def __init__(self, state = None, player = None, action = None, parent = None):
        self.action = action
        self.state = state
        self.player = player
        self.parent = parent
        
        self.children = []
        self.possible_moves = []
        
        self.policy = None
        self.value = 0
        self.visits = 0
        self.move_id = 0
        self.terminal = False
        self.terminal_value = 0.0
        
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
        self.root.children = [None] * len(self.root.possible_moves)
        self.net = net
        self.evaluate(self.root)
        
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

        policy = node.policy
        assert policy is not None, f"Node without policy: {node}"
        
        values = np.empty((len(node.possible_moves), 2), dtype = float)
        values[:] = np.array([
            (c.value, c.visits) if isinstance(c, Node) else (0.0, 0.0)
            for c in node.children
        ]).reshape(-1, 2)

        # child.value is stored from the child's perspective, so the parent
        # must negate it when comparing move quality.
        values[:, 0] *= -1
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
        child.children = [None] * len(child.possible_moves)
        if reward != 0 or len(child.possible_moves) == 0:
            child.terminal = True
            # reward is from the mover's perspective (node.player). Child is the
            # opponent-to-move perspective, so winning move means -1 for child.
            child.terminal_value = -float(reward)
        node.children[move] = child
        return child, reward
    
    def evaluate(self, node):
        if node.terminal:
            return 0
        with torch.no_grad():
            value, policy = self.net(node.state, node.player)

        val_float = value.item() 
        raw_policy = policy[0].cpu().numpy() 

        legal_policy = np.zeros(len(node.possible_moves))
        board_width = node.state.Board.shape[1]

        for i, move in enumerate(node.possible_moves):
            action_idx = move[0] * board_width + move[1]
            legal_policy[i] = raw_policy[action_idx]

        policy_sum = np.sum(legal_policy)
        if policy_sum > 0:
            node.policy = legal_policy / policy_sum
        else:
            node.policy = np.ones(len(legal_policy)) / len(legal_policy)

        return val_float
    
    def backpropagate(self, node, value):
        if node is None:
            return

        node.visits += 1
        node.value += value
        self.backpropagate(node.parent, -value)
        
    def iter(self, num_of_iterations):
        for i in range(num_of_iterations):
            move, parent = self.select(self.root)

            if move == -1:
                # Do not reuse accumulated parent.value here. For terminal nodes
                # we must propagate the fixed game outcome.
                value = parent.terminal_value
                self.backpropagate(parent, value)
            else:
                # nn sim
                new_node, reward = self.expand(move, parent)
                value = self.evaluate(new_node)
                if reward != 0:
                    # reward is from the mover's perspective; the new node is
                    # the opponent to move, so flip the sign here.
                    self.backpropagate(new_node, -reward)
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