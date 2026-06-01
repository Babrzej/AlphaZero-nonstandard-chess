# AlphaZero-nonstandard-chess

## Monte Carlo Tree Search (MCTS)

MCTS is responsible for **searching and decision making**. It simulates many possible move sequences by building a tree of game states. It decides which branches to explore using statistics from previous simulations. It does not know the game. It learns which paths are good by testing them.

**Its job is to:** explore possible future moves || collect results from simulations || choose the move with the highest number of visits

### Legend of Variables:

* **Node:** represents a single, specific board state at a given moment in the game. The decision tree consists of thousands of interconnected nodes. Each node remembers the history of how well it performed during the simulations.
* **N (`visits` - Number of Visits):** A counter indicating how many times the algorithm has checked or passed through this specific board state. The higher the N, the more the algorithm trusts this path. In the very end, the move with the highest number of visits (not the highest evaluation) is chosen to be played on the real board.
* **V (`value` - Cumulative Value):** The total sum of all points (evaluations) gathered from simulations that passed through this node. `Value` by itself doesn't tell us much until we divide it by the number of visits.
* **Q (`q ` - Average Move Quality):** The result of the simple equation q = value/visits. The scale tends to range from -1 (certain loss) to 1 (certain win).
* **P (`policy` - Propability):** The output from the neural network (Policy Head). It is a hint from the network that a given move seems brilliant at first glance, before MCTS even starts analyzing it deeply.
* **Game State (`state`):** The physical record of where the pieces currently stand.
* **Move (`action`):** The specific move (e.g., "Pawn from e2 to e4") that led to this particular board state.
* **The Relationships (`parent` **and `children`**):** The parent is the board state exactly one move ago. The children is an array stores future states, but it expands dynamically. new nodes are added one-by-one only when the algorithm decides to explore that specific move.
* **End State (`terminal`):** A boolean variable indicating whether this specific board layout is the end of the game. If it is, MCTS stops and does not search for further moves from this node.
* **Upper Confidence Bound (`ucb` = Q + c * P * [ √(N_parent) / (1 + N_child) ]):** The main formula driving the algorithm. It sums up what we already empirically know about the move (Q) with the hint provided by the network (P) to decide which node is worth exploring next.

### **The 4 Core Phases**

A **Single MCTS iteration**consists of 4 distinct phases. The algorithm loops through these 4 phases thousands of times before making a final move.

#### **1. Selection**

Starting from the **root** **Node** , the algorithm traverses down the existing tree to find the most promising path. At each step, it calculates the **`ucb`** for all available moves. It consistently selects the child with the highest **`ucb`**. This process continues until the algorithm reaches a **leaf node.**

#### **2. Expansion**

The decision tree grows strictly on demand. When the Selection phase stops at a leaf node, MCTS does not pick a move to explore at random. Instead, it identifies the **unexplored action with the highest `ucb`**. Since unvisited moves have no history (N=0, Q=0), the selection is driven by the `policy`  hint stored in the parent node. A new child Node is then created for this specific move.

#### **3. Evaluation**

Instead of playing the game to the end, the algorithm uses a **Neural Network** to instantly assess the new state. It returns two values:

* **Neural Network Value (**v**):** The predicted win probability for the current state (used to update the tree).
* **`Policy` (**P**):** A "roadmap" of probabilities for all legal moves, guiding which child to expand next.

#### **4. Backpropagation**

The neural network value is passed back up the tree, all the way to the root, following the exact path of parent nodes taken during the Selection phase. For every node on this path:

* **`Visits` (**N**):** Each node on the path increments its counter by 1.
* **Neural Network Value (V):** Each node adds the network value to its total score.
* **Sign Flip:** The value is multiplied by -1 at each step up, because a good position for the current player is a bad one for the opponent.

### **The Final Decision**

After completing all iterations, it makes its final choice based on `visits` (N) from the Root Node.

* **The Rule:** The move that was visited the most times (**max** **N**) is chosen as the final move.
* Why `visits`, Q value could be a "lucky" result from a path that hasn't been explored enough.

## Neural Network (CNN)

The neural network is **responsible for evaluation and guidance**. It looks at a single board position and predicts what is important. It does not search or simulate games. It only gives predictions based on learned patterns.

**Its job is to:** evaluate how good the current position is (Neural Network Value) || suggest which moves are most promising (Policy) || guide MCTS so it explores better paths

### **1. Tensor (Input Representation)**

The neural network receives the game state as a 3D tensor: `Channels × Height × Width`

**Channels:** Separate binary feature planes (one per piece type or attribute). Each channel acts as an independent switch:

* **1 · w = w** → to pass the signal
* **0 · w = 0** → fto mute it

This preventing the network from treating a Queen simply as "nine Pawns".

### **2. The Convolution layer (Feature Extraction)**

The convolution layer scans the input (like a board or image) using a small window (kernel) and looks for local patterns. It moves this window across the entire grid and checks what features appear in each location.

`torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding)` || **Conv1d:** Sequences, **Conv2d:** Grids, **Conv3d:** Volumes

* **in_channels:** The depth (number of layers) of your input data. For traditional images, this is 3 (RGB) or 1 (grayscale). For custom encoded data, it is the number of separate 2D grids (e.g., matrices of 0s and 1s) stacked together, where each layer represents a different attribute or binary state.
* **out_channels:** The number of different patterns the network searches for. If set to 64, the network scans your input 64 times simultaneously, each time looking for a different hidden pattern. This turns your input into a new stack of 64 grids, showing exactly where each specific pattern was found. Use powers of 2 or multiples of 8 to optimize calculations for GPU hardware.
* **kernel_size**: The network moves this focus area over every single part of your board (like a scanner). A value of 3 creates a 3x3 square. It dictates how large of a local area the network analyzes in a single step. Always use odd numbers to provide a clear center pixel
* **padding**: To maintain the original grid dimensions and to fix "edge blindness" we add a border of zeros using the formula **Padding = (Kernel_Size-1)/2**. For a 3x3 kernel, padding=1 creates a "safety buffer" that allows the window to center on corner squares, ensuring every spot is analyzed and the output size remains 3x3.

### **3. Batch Normalization (Normalizes values)**

Batch Normalization standardizes the output of a layer so that it has a mean close to 0 and variance close to 1. This makes training more stable and faster, because the network doesn’t have to constantly adjust to changing value ranges.

`torch.nn.BatchNorm2d(num_features)`

* **num_features:** number of channels (must match `out_channels` from Conv2d)

### **4. ReLU (Removes negative values)**

**Applies the function:**

$$
f(x) = \max(0, x)
$$

ReLU replaces all negative numbers with 0.This adds non-linearity, which lets the network learn complex patterns instead of just simple linear ones.

`torch.nn.ReLU()`

### **5. ResNet Block (Residual learning)**

A Residual Block introduces a **skip connection** that adds the input directly to the output of a few layers:

$output = F(x) + x$

Instead of learning a full transformation, the network learns a **residual (difference)** .
This makes training deep networks much easier and prevents vanishing gradients.

**Typical structure:**

* Conv2d
* BatchNorm2d
* ReLU
* Conv2d
* BatchNorm2d
* Skip connection (+ input)
* ReLU

### **6. Flatten (reshape to vector)**

Transforms a multi-dimensional tensor (C × H × W) into a 1D vector so it can be passed into fully connected layers.

`torch.nn.Flatten()`

### **7. Dropout Block (Randomly disables neurons)**

During training, randomly sets a fraction of inputs to zero. This prevents the network from relying too heavily on specific neurons and improves generalization (learning rules, not examples).

**Typical structure (it can be use many times):**

* Linear
* Batch1D
* relu
* Dropout

`torch.nn.Dropout(p)`

* **p:**  probability of dropping a neuron

### **8. Linear Layer (combines all features)**

Formula:


$$
y = Wx + b
$$

* **W:** Weights tell the network how important each input is.
* **b:** bias is just a small adjustment

`torch.nn.Linear(in_features, out_features)`

* **in_features:** size of input vector
* **out_features:** size of output vector

`Input: [feature_1, feature_2, feature_3]`

`Output:  feature_1 x w_1 + feature_2 x w_2  + feature_3 x w_3 + b`

### **9. Value Head (Predicts outcome in MCTS)**

Outputs representing how good the current state is (e.g., probability of winning). Fixed range [-1, 1]

**Typical structure:**

* Linear
* ReLU
* Linear
* Tanh: takes any number (even very large) and compresses it into the range [-1, 1] .

### **10. Policy Head (Chooses actions in MCTS)**

Outputs a probability distribution over all possible actions. Each value represents how likely a move is.

**Typical structure:**

* Linear
* ReLU
* Linear
* Softmax

**Softmax:** Transforms raw scores into probabilities.

$$
P_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

`torch.nn.Softmax(dim)`

* dim[batch_size, num_actions] = choose direction to normalize[0: across samples 1: across actions]

## **Self-Play**

Self-play generates training examples by letting the current network play both sides of the game.

### Game generation flow (`play_game`)

For each move:

1. Build an MCTS rooted at the current state.
2. Run `MCTS_SIMS` simulations (`180` by default).
3. Read the search policy from the root.
4. Choose a move:
   * first 7 plies: sample stochastically (temperature = 1),
   * later plies: choose argmax (temperature = 0).
5. Save one training sample:
   * encoded state tensor (`state_to_tensor`),
   * full action policy vector of size `1225`,
   * player-to-move.
6. Apply the selected move and continue until terminal state or step limit.

Stopping conditions:

* no legal moves,
* environment reward is non-zero (win/loss),
* `MAX_GAME_STEPS` reached (`100`, treated as draw by limit).

### Reward assignment (`assign_rewards`)

After game end, every stored position gets a value target:

* win/loss: `+1` for positions from the winner's perspective, `-1` otherwise,
* draw: fixed value `-0.3`.

Each replay item is: `(state_tensor, policy_target, value_target)`.

## **Training Loop (`train_alphazero.py`)**

The script alternates between data generation and network optimization.

### Hyperparameters

* `EPISODES = 20`
* `GAMES_PER_EPISODE = 64`
* `MCTS_SIMS = 180`
* `EPOCHS = 5`
* `BATCH_SIZE = 64`
* `MAX_GAME_STEPS = 100`
* `BUFFER_SIZE = 25000`
* `NUM_WORKERS = max(1, os.cpu_count() - 2)`

### Episode structure

For each episode:

1. **Self-play phase**
   * Run `GAMES_PER_EPISODE` games in parallel with `ProcessPoolExecutor`.
   * Extend a replay buffer (`deque`) with all generated move samples.
2. **Training phase**
   * Train for `EPOCHS` over random mini-batches from replay buffer.
   * Loss = value MSE + policy cross-entropy style loss  
     (`-sum(target_policy * log(pred_policy)) / BATCH_SIZE`).
3. **Checkpointing**
   * Save model weights as `alphazero_model_ep{episode}.pth`.

### Checkpoint resume behavior

At startup, training tries to load the latest available checkpoint pattern used in the script and resumes from the detected episode index; otherwise it starts from episode 1.

### Multiprocessing note

The entrypoint sets `torch.multiprocessing` start method to `spawn` and shares network memory before self-play workers are launched.

## Tournament Results

### Head-to-Head Tournament Matrix

![Head-to-Head Tournament Matrix](https://github.com/user-attachments/assets/6a767cab-1cc3-4d89-87cc-831f6b691cb7)

### Tournament vs Random Visualization

![Tournament Visualization](https://github.com/user-attachments/assets/7366f372-e819-4860-9393-8140139075f7)

We can observe that model is indeed becoming better. Even though it got worse for a while.