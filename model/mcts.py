import numpy as np
from model.node import Node
from chess_env.chess_game import Chess

#---------------------------------------------
# !!! IMPORTANT !!!
# Nie obiecuje że to działa (raczej nie działa xD)
# i ma sens jakbyś chciał kontynuowac
# Więc możesz przede wszystkim "podpiąć" silnik szachowy
# i jako nn jakaś funckje która zwraca losowe wartości
# prawdopodobnie dziala :)
#--------------------------------------------

# Constant for the UCB calculation
c_ucb = np.sqrt(2)

class MCTS:
    def __init__(self, game, player, net):
        #TODO: analyze all needed parameters
        self.game = game
        self.root = Node(state = game.initial_state(), player = player)
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
        possible_moves = self.game.actions(node.state, node.player)
        ucb = []

        #policy = self.net.get_policy(node.state)
        #---------------------------------------------------------------------------------
        raw_policy = np.random.random(len(possible_moves))
        policy = raw_policy / np.sum(raw_policy)
        #---------------------------------------------------------------------------------

        #we iterate over every move
        for move, pol in zip(possible_moves, policy):
            #TODO: make use of reward

            move_visited = False
            #we check if given move was already visited
            for child in node.children:
                if child.action == move:
                    new_ucb = calc_ucb(node.visits, child.visits, child.value, pol)
                    ucb.append(new_ucb)
                    move_visited = True
                    break

            if not move_visited:
                new_ucb = calc_ucb(node.visits, 0, 0, pol)
                ucb.append(new_ucb)

        selected_move = possible_moves[np.argmax(ucb)]
        selected_state, reward = self.game.next_state_and_reward(node.player, node.state, selected_move)

        #If we already visited we go deeper
        for child in node.children:
            if child.action == selected_move:
                return self.select(child)

        #Ultimately return newly visited move(its state and parent node)
        return selected_move, selected_state, node

    #Create node for state visited first time
    def expand(self, move, state, node):
        child = Node(action=move, state=state, parent=node, player=3 - node.player)
        node.children.append(child)
        return child

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
            print(f"\n--- MCTS Tree (Showing up to depth {max_depth}) ---")

        # Add indentation based on how deep we are in the tree
        indent = "  " * depth * 2  # 4 spaces per depth level

        # Use the __repr__ we just made!
        print(f"{indent}{node}")

        # Base case: Stop if we reach the maximum display depth or a leaf node
        if depth >= max_depth or not node.children:
            return

        # Pro-Tip: Sort the children by visits before printing!
        # This puts the moves the AI is actually considering at the top.
        sorted_children = sorted(node.children, key=lambda c: c.visits, reverse=True)

        for child in sorted_children:
            self.print_tree(child, depth + 1, max_depth)

    def iter(self, num_of_iterations):
        for i in range(num_of_iterations):
            move, next_state, parent = self.select(self.root)
            new_node = self.expand(move, next_state, parent)
            #nn sim
            value = np.random.random()*2 - 1
            self.backpropagate(new_node, value)




def calc_ucb(visits_parent, visits, value, policy, c = c_ucb):
    p = c * policy * np.sqrt(visits_parent) / (1 + visits)
    if value == 0:
        ucb = p
    else:
        ucb = value/visits + p
    return ucb

