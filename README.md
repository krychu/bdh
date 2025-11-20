# The Dragon Hatchling (BDH)

This repository contains an educational PyTorch implementation of the BDH-GPU architecture proposed in the paper:

> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).

BDH is a novel Large Language Model architecture based on a scale-free, biologically-inspired network of locally-interacting neurons.

I find the paper particularly fascinating for its elegant synthesis of concepts from neuroscience, distributed computing, dynamical systems, and formal logic into a single, GPU-friendly architecture.

## Demo: Pathfinding and Visualizing Reasoning Logic

The model is trained on a pathfinding task: given an N×N board with obstacles, find the shortest path from START to END.

![combined_board_interleaved](https://github.com/user-attachments/assets/5ba344a1-a741-4b94-99f8-62a589bd7a86)

BDH's architecture enables direct visualization of its internal computation. However, visualizing signal flow is challenging because inference relies on the superposition of static learned circuits (the "wiring" or "program") and dynamic attention mechanisms (the "state").

**Visualization Note:** The model contains over 8,000 neurons, but I render only the **"Hub" subgraph** - the top strongest connections. Remarkably, the sparse, modular organization you see is emergent. The model was not hard-coded to have hubs, but spontaneously organized itself this way from random initialization. This replicates the paper's empirical findings.

The animation above shows the model solving a board puzzle:
*   **Left:** The model's output board predictions being refined layer by layer.
*   **Right:** A unified view of the reasoning process ($L-1 \to L$), separating Association from Causality.
    *   **Nodes:** Blue Nodes represent **Context** (neurons $y_{l-1}$ active in the previous step). Red Nodes represent **Inference** (neurons $x_l$ triggered in the current step).
    *   **Blue Edges (Association):** Highlight non-causal co-activation patterns of the previous context. They connect Blue nodes ($y_{l-1}$) that are functionally related and active together.
    *   **Red Edges (Causality):** Show the physical signal flow. They trace how the previous context propagates through the fixed topology to trigger the new Red nodes ($y_{l-1} \to x_l$).

Together, they visualize a logical chain: Blue establishes "what we know" and Red executes the logical implication "what follows from it."

## Key Concepts of the BDH Architecture

The BDH architecture introduces several design choices that distinguish it from conventional Transformers and enable the causal interpretability shown above.

* **Neuron-Centric Scaling**: The model scales primarily in the high-dimensional **Neuron** dimension, rather than the dense latent dimension of Transformers. Parameters and state are localized to specific neuron pairs, mirroring biological structure.
* **Fixed Topologies as "Learned Programs"**: The model weights define two sparse, scale-free graphs that act as the system's fixed ruleset:
    1. **The Causal Circuit (`E @ Dx`):** Implements a probabilistic form of **Modus Ponens** reasoning ("If concept A is active, trigger concept B"). This corresponds to the **Red** edges in the visualization.
    2. **The Semantic Circuit (`Dy.T @ Dy`):** Groups neurons representing similar concepts (clustering). This corresponds to the **Blue** edges in the visualization.
* **Dynamic Synaptic State (Hebbian Memory)**: During inference, the network maintains fast-changing memory in the form of synaptic weights (state). These are updated via a **Hebbian Learning** rule ("neurons that fire together, wire together"). This allows the model to dynamically re-weight its fixed program based on the current context.
* **Sparse & Positive Activations**: The architecture enforces all activation vectors to be positive and highly sparse. Empirically, only a small fraction of neurons fire per step. This sparsity is what makes the "Hub" visualization possible - it filters out noise and reveals the distinct logical paths taken by the model.

The visualizations in the demo are not a post-hoc approximation; they are a direct rendering of the model's state variables during the forward pass.

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

Optional: You can ensure reproducibility by setting a fixed random seed:

```bash
python3 boardpath.py --mode train --seed 42
```

The trained model will be saved to `boardpath.pt`.

#### Inference & Visualization
To load a trained model and run it on a randomly generated board:
```bash
python3 boardpath.py --mode inference
```

Optional: If you have a specific checkpoint file you wish to load:

```bash
python3 boardpath.py --mode inference --model my_model.pt
```

This will print the input, target, and predicted boards to the console and generate several visualization GIFs:
- `output_predictions.gif`: The model's board predictions evolving layer by layer.
- `graph_e_dx_hub_flow.gif`: Signal flow through the E @ Dx communication topology.
- `graph_dy_coact_hub.gif`: Co-activation patterns in the Dy.T @ Dy attention decoder.
- `graph_interleaved_hub.gif`: Unified dual-network view (blue: Dy co-activation, red: Dx signal flow).
- `combined_board_interleaved.gif`: Side-by-side board predictions and dual-network visualization (shown in the demo).

#### Configuration
To adjust the model architecture or task parameters (e.g., board size, number of neurons), edit the `get_config()` function in `boardpath.py`.
