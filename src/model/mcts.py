import numpy as np

c_ucb = 1


class Node:
    def __init__(self, state=None, player=None, action=None, parent=None):
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
        self.reward = 0
        self.policy = None

    def __repr__(self):
        action_str = str(self.action) if self.action is not None else "Root"
        avg = self.value / self.visits if self.visits > 0 else 0.0

        return (f"<Node | Player: {self.player} | Action: {action_str} | "
                f"Visits: {self.visits} | AvgVal: {avg:.3f} | Children: {len(self.children)}>")

    def visited(self):
        return self.visits != 0


class MCTS:
    def __init__(self, game, state, player, net):
        self.game = game
        self.root = Node(state=state, player=player)
        self.root.possible_moves = self.game.actions(state, player)
        self.root.children = np.empty(len(self.root.possible_moves), dtype=object)
        self.net = net

    def __repr__(self):
        game_name = self.game.__class__.__name__ if self.game else "Unknown"
        net_name = self.net.__class__.__name__ if self.net else "Unknown"
        total_simulations = self.root.visits if self.root else 0

        return f"<MCTS Engine | Game: {game_name} | Net: {net_name} | Total Simulations: {total_simulations}>"

    def mask_and_normalize(self, full_policy, node):
        fp = full_policy.detach().cpu().numpy().flatten()
        probs = np.array([fp[self.game.action_to_index(m)] for m in node.possible_moves])
        s = np.sum(probs)
        return probs / s if s > 0 else np.ones(len(probs)) / len(probs)

    def select(self, node):
        if node.terminal or len(node.possible_moves) == 0:
            return -1, node

        values = np.zeros((len(node.possible_moves), 2), dtype=float)
        values[:] = np.array([
            (-c.value, c.visits) if isinstance(c, Node) else (0.0, 0.0)
            for c in node.children
        ]).reshape(-1, 2)
        ucb = calc_ucb(values, node.visits, node.policy)
        best = np.argmax(ucb)

        if isinstance(node.children[best], Node):
            return self.select(node.children[best])
        else:
            return best, node

    def expand(self, move, node):
        state, reward = self.game.next_state_and_reward(node.player, node.state, node.possible_moves[move])
        child = Node(action=node.possible_moves[move], state=state, parent=node, player=3 - node.player)
        child.possible_moves = self.game.actions(state, 3 - node.player)
        child.children = np.empty(len(child.possible_moves), dtype=object)
        if reward != 0 or len(child.possible_moves) == 0:
            child.terminal = True
            child.reward = reward
        node.children[move] = child
        return child, reward

    def backpropagate(self, node, value):
        if node is None:
            return

        node.visits += 1
        node.value += value
        self.backpropagate(node.parent, -value * 0.95)

    def print_tree(self, node=None, depth=0, max_depth=3):
        if node is None:
            node = self.root
            print(f"\n--- MCTS Tree (Showing up to depth {max_depth}) ---")

        indent = "  " * depth * 2
        print(f"{indent}{node}")

        if depth >= max_depth:
            return

        valid_children = [c for c in node.children if isinstance(c, Node)]

        if not valid_children:
            return

        sorted_children = sorted(valid_children, key=lambda c: c.visits, reverse=True)

        for child in sorted_children:
            self.print_tree(child, depth + 1, max_depth)

    def iter(self, num_of_iterations):
        if self.root.policy is None:
            value_tensor, policy_tensor = self.net(self.root.state, self.root.player)
            self.root.policy = self.mask_and_normalize(policy_tensor, self.root)

            epsilon = 0.25
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(self.root.possible_moves))
            self.root.policy = (1 - epsilon) * self.root.policy + epsilon * noise

        for i in range(num_of_iterations):
            move, parent = self.select(self.root)

            if move == -1:
                if parent.reward != 0:
                    value = -1.0
                else:
                    value = -0.3 
                self.backpropagate(parent, value)
            else:
                new_node, reward = self.expand(move, parent)
                
                if reward != 0:
                    self.backpropagate(new_node, -1.0)
                elif len(new_node.possible_moves) == 0:
                    self.backpropagate(new_node, -0.3)
                else:
                    value_tensor, policy_tensor = self.net(new_node.state, new_node.player)
                    new_node.policy = self.mask_and_normalize(policy_tensor, new_node)
                    self.backpropagate(new_node, value_tensor.item())

    def output(self):
        if len(self.root.possible_moves) == 0:
            # No legal moves exist in this state
            return None, np.array([])

        visits = np.zeros(len(self.root.children))
        visits[:] = [
            c.visits if isinstance(c, Node) else 0
            for c in self.root.children
        ]

        sum_visits = np.sum(visits)
        if sum_visits > 0:
            best = np.argmax(visits)
            policy = visits / sum_visits
        else:
            # if no nodes were visited during search, use prior network policy
            best = np.argmax(self.root.policy) if self.root.policy is not None else 0
            policy = self.root.policy if self.root.policy is not None else np.ones(len(visits)) / len(visits)

        best_move = self.root.possible_moves[best]
        return best_move, policy


def calc_ucb(values, visits_parent, policy, c=c_ucb):
    value = values[:, 0]
    visits = values[:, 1]
    q = np.divide(value, visits, out=np.zeros_like(value), where=visits != 0)
    p = c * policy * np.sqrt(visits_parent) / (1 + visits)
    ucb = q + p
    return ucb