# The Dragon Hatchling (BDH)

This repository contains an educational PyTorch implementation of the BDH-GPU architecture proposed in the paper:

> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).

BDH is a novel Large Language Model architecture based on a scale-free, biologically-inspired network of locally-interacting neurons. It aims to bridge the gap between the tensor-based operations of modern Transformers and the graph-based, distributed dynamics of the human brain.

I find the paper particularly fascinating for its elegant synthesis of concepts from neuroscience, dynamical systems, and formal logic into a single, GPU-friendly architecture.

## Demo: Pathfinding and Visualizing the Model's "Brain"

The model is trained on a pathfinding task: given an N×N board with obstacles, find the shortest path from START to END.

Because of BDH's unique architecture, we can directly visualize its internal state during inference. The animation below shows the model solving a board puzzle. On the left is the model's output as it refines the path across its internal layers. On the right is a real-time visualization of its "brain" - the emergent communication network between its neurons. Red nodes and edges indicate active neurons and synapses, showing the model's "thought process" as it routes information to find the solution.

![Combined board and network visualization](combined_board_network.gif)

*Legend: `.` = Floor, `#` = Wall, `S` = Start, `E` = End, `*` = Path*

## Key Concepts of the BDH Architecture

The BDH architecture has several properties that distinguish it from conventional Transformers and enable its unique interpretability.

* **Neuron-Centric Scaling**: The architecture operates/scales primarily in the high-dimensional **Neuron** dimension (`N`), rather than the Transformer's dense latent dimension. State and parameters are primarily associated with these neurons, mirroring a biological structure.
* **Static Graph Topology (The Program)**: The learned parameter matrices (`E`, `Dx`, `Dy`) define a sparse, scale-free **Communication Graph** - the model's long-term learned knowledge and reasoning rules. This emergent structure is observable in the fixed gray background of the network visualization.
* **Dynamic Synaptic State (The Memory)**: During inference (forward pass), the network maintains a fast-changing state that acts as **in-context memory** or **synaptic plasticity** (the $\rho$ or $\sigma$ matrix in the paper). In the visualization, this dynamic memory is visiblea as the active red edges.
* **Sparse & Positive Activations**: All internal activation vectors are enforced to be positive and empirically observed to be highly sparse (only a few percent of neurons fire per token). This is key to both computational efficiency and the monosemantic interpretability of individual neurons and synapses.

The visualizations in the demo are not a post-hoc approximation; they are a direct rendering of the model's state during computation.


## Usage

#### Installation
```bash
pip install -r requirements.txt
```

#### Training
To train a new model from scratch, run:
```bash
python3 boardpath.py --mode train
```
The trained model will be saved to `boardpath.pt`.

#### Inference & Visualization
To load a trained model and run it on a randomly generated board:
```bash
python3 boardpath.py --mode inference --model boardpath.pt
```
This will print the input, target, and predicted boards to the console and generate several visualization GIFs:
- `output_predictions.gif`: The model's board output evolving layer by layer.
- `graph_e_dx_full.gif`: The full neuron communication graph and its activations.
- `graph_e_dx_hub.gif`: A zoomed-in view of the most active "hub" of the network.
- `combined_board_network.gif`: The side-by-side visualization shown in the demo.

#### Configuration
To adjust the model architecture or task parameters (e.g., board size, number of neurons), edit the `get_config()` function in `boardpath.py`.

