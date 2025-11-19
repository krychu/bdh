# The Dragon Hatchling (BDH)

This repository contains an educational PyTorch implementation of the BDH-GPU architecture proposed in the paper:

> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).

BDH is a novel Large Language Model architecture based on a scale-free, biologically-inspired network of locally-interacting neurons. It aims to bridge the gap between the tensor-based operations of modern Transformers and the graph-based, distributed dynamics of the human brain.

I find the paper particularly fascinating for its elegant synthesis of concepts from neuroscience, dynamical systems, and formal logic into a single, GPU-friendly architecture.

## Demo: Pathfinding and Visualizing the Model's "Brain"

The model is trained on a pathfinding task: given an N×N board with obstacles, find the shortest path from START to END.

BDH's architecture enables direct visualization of its internal computation. However, the challenge is that inference involves the superposition of multiple learned circuits: two fixed topologies (`E @ Dx` for signal propagation and `Dy.T @ Dy` for attention decoding) plus dynamic synaptic state that serves as in-context memory. Each reasoning layer propagates signals through these overlapping networks to compute the next state.

The animation below shows the model solving a board puzzle across its reasoning layers. **Left:** the model's output board predictions being refined layer by layer. **Right:** a unified view of the model's dual-network computation. Blue edges highlight **co-activation patterns** - functionally-related neurons activated together by the attention summary (`y[i] * y[j]` over the `Dy.T @ Dy` topology). Red edges show **causal signal flow** - actual neuron-to-neuron communication (`y[i] * (E @ Dx)[i,j]` over the `E @ Dx` topology). Together, these reveal which conceptual modules are active and how information flows between them to solve the task.

![Combined board and network visualization](combined_board_network.gif)

*Legend: `.` = Floor, `#` = Wall, `S` = Start, `E` = End, `*` = Path*

## Key Concepts of the BDH Architecture

The BDH architecture has several properties that distinguish it from conventional Transformers and enable its unique interpretability.

* **Neuron-Centric Scaling**: The architecture operates/scales primarily in the high-dimensional **Neuron** dimension (`N`), rather than the Transformer's dense latent dimension. State and parameters are primarily associated with these neurons, mirroring a biological structure.
* **Dual Fixed Topologies (The Program)**: The learned parameter matrices define two sparse, scale-free topologies: `E @ Dx` (signal propagation paths) and `Dy.T @ Dy` (attention decoder relationships). These represent the model's long-term learned reasoning circuits. In the visualization, these appear as the gray structural scaffold.
* **Dynamic Synaptic State (The Memory)**: During inference, the network maintains fast-changing in-context memory through synaptic co-activation patterns (computed as `x.T @ y` per layer). This dynamic state, combined with the fixed topologies, determines which circuits are active for each reasoning step. In the visualization, active circuits appear as colored edges and nodes.
* **Sparse & Positive Activations**: All internal activation vectors are enforced to be positive and empirically observed to be highly sparse (only a few percent of neurons fire per token). This is key to both computational efficiency and the monosemantic interpretability of individual neurons and circuits.

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
- `output_predictions.gif`: The model's board predictions evolving layer by layer.
- `graph_e_dx_hub_flow.gif`: Signal flow through the `E @ Dx` communication topology.
- `graph_dy_coact_hub.gif`: Co-activation patterns in the `Dy.T @ Dy` attention decoder.
- `graph_interleaved_hub.gif`: Unified dual-network view (blue: Dy co-activation, red: Dx signal flow).
- `combined_board_interleaved.gif`: Side-by-side board predictions and dual-network visualization (shown in the demo).

#### Configuration
To adjust the model architecture or task parameters (e.g., board size, number of neurons), edit the `get_config()` function in `boardpath.py`.

