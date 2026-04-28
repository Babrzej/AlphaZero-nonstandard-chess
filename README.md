# AlphaZero-nonstandard-chess

## Monte Carlo Tree Search:

### Legend of Variables:

* **Node:** represents a single, specific board layout at a given moment in the game. The decision tree consists of thousands of interconnected nodes. Each node remembers the history of how well it performed during the simulations.
* **N (`visits` - Number of Visits):** A counter indicating how many times the algorithm has checked or passed through this specific board state. The higher the N, the more the algorithm trusts this path. In the very end, the move with the highest number of visits (not the highest evaluation) is chosen to be played on the real board.
* **W (`value` - Cumulative Value): The total sum of all points (evaluations) gathered from simulations that passed through this node. W by itself doesn't tell us much until we divide it by the number of visits.**
* ****Q** **(`q ` - Average Move Quality):** The result of the simple equation** Q=W/N (in your code, you do this division dynamically inside the `calc_ucb` function). It tells the algorithm: *Based on my tests so far, this move leads to an average outcome of X"* . The scale tends to range from -1 (certain loss) to 1 (certain win).**
* ****P** (`policy`  **- Propability):** The output from the neural network (specifically the "Policy Head"). It is a hint from the network (e.g., 0.8) that a given move seems brilliant at first glance, before MCTS even starts analyzing it deeply. Currently, in your code, you simulate this by generating a random .**
* ****Game State (`state`):** The physical record of where the pieces currently stand (your board object).**
* ****Move (`action`):** The specific action (e.g., "Pawn from e2 to e4") that led to this particular board state.**
* ****The Relationships** (**`parent` and **`children`**). The** `parent` is the board state exactly one move prior. The `children` is an array storing new nodes—all possible board states that can occur in the next move.**
* ****End State** (`terminal`): A boolean variable (True/False) indicating whether this specific board layout is the end of the game (e.g., checkmate or stalemate where the Reward != 0). If it is, MCTS stops and does not search for further moves from this node.**
* ****Upper Confidence Bound (`ucb`):** The main formula driving the algorithm (represented in your code as** **`q + p`). It sums up what we already empirically know about the move (**Q**) with the hint provided by the network (**P**) to decide which node is worth exploring next.**

### **The 4 Core Phases**

****Single MCTS iteration** consists of 4 distinct phases. The algorithm loops through these 4 phases thousands of times before making a final move.**

#### **1. Selection**

#### **2. Expansion**

#### **3. Evaluation**

#### **4. Backpropagation**

### **The Final Decision**

## **Residual Network**
