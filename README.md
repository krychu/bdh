# The Dragon Hatchling (BDH)

This repository contains an educational PyTorch implementation of the BDH-GPU architecture proposed in the paper:

> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).

BDH is a novel Large Language Model architecture based on a scale-free, biologically-inspired network of locally-interacting neurons.

I find the paper particularly fascinating for its elegant synthesis of concepts from neuroscience, distributed computing, dynamical systems, and formal logic into a single, GPU-friendly architecture.

## Demo: Pathfinding and Visualizing Reasoning Logic

The model is trained on a pathfinding task: given an N×N board with obstacles, find the shortest path from START to END.

![combined_board_neuron](https://github.com/user-attachments/assets/a3b76ce7-b1cc-4824-89f6-d4c2e1528a7f)

<table width="100%">
    <thead>
    <tr>
      <th width="50%" style="text-align:left;">Left Panel: Board Predictions</th>
      <th width="50%" style="text-align:left;">Right Panel: Neuron Dynamics (Gx = E @ Dx)</th>
    </tr>
  </thead>
  <tr>
    <td valign="top" width="50%">
      The model's output refined layer by layer.<br><br>Legend: FLOOR (white), WALL (black), START (red), END (green), PATH (gold)<br><br>
    </td>
    <td valign="top" width="50%">
      Signal flow through the learned "causal circuit" - the neuron-to-neuron connectivity graph.<br><br>
      - Blue rings: Source neurons (y<sub>l−1</sub>)<br>
      - Red fill: Destination neurons (x<sub>l</sub>)<br>
      - Edge darkness: Signal flow, y<sub>l−1</sub> × Gx × x<sub>l</sub><br><br>
      Activations are averaged across all board cells to produce one value per neuron.
    </td>
  </tr>
</table>

BDH's architecture enables direct visualization of its internal computation. The challenge is that inference relies on multiple superimposed topologies: fixed learned circuits (the model weights) and dynamic activations that change at inference time.

The model has 8,000+ neurons but for clarity I render only the hub subgraph selected by connectivity degree. Specifically: neurons are ranked by their degree in Gx (counting edges where |Gx[i,j]| > threshold), top candidates are selected, and small disconnected components are pruned. Remarkably, the sparse, modular organization you see is emergent. The model was not hard-coded to have hubs, but spontaneously organized itself this way from random initialization. This replicates the paper's empirical findings.

---

![combined_attention_sparsity](https://github.com/user-attachments/assets/bb8176a3-b2c4-467a-824f-9835f576d8d0)

<table width="100%">
    <thead>
    <tr>
      <th width="50%" style="text-align:left;">Left Panel: Board Attention</th>
      <th width="50%" style="text-align:left;">Right Panel: Sparsity Dynamics</th>
    </tr>
  </thead>
  <tr>
    <td valign="top" width="50%">
      The model's output refined layer by layer, with extra detail.<br><br>
      - Blue arrows: top 30 strongest cell-to-cell attentions<br>
      - Red dots: proportion of active neurons (x) per cell<br>
      - PATH cells in gold, confidence shown via alpha
    </td>
    <td valign="top" width="70%">
      Percentage of neurons active per layer. Red (x): ~20%, Blue (y): ~3-5%
    </td>
  </tr>
</table>

Blue arrows show attention initially radiating from START and END toward neighboring cells. As the path extends from both endpoints, attention shifts to the newly predicted cells, flowing outward to discover the remaining route until the path connects in the middle.

Red dots show more neurons firing at START, END, and WALL, with PATH cells activating progressively as predictions solidify.

The chart confirms that y activations are indeed very sparse throughout inference.

## Key Concepts of the BDH Architecture

The BDH architecture introduces several design choices that distinguish it from conventional Transformers and enable the causal interpretability shown above.

* **Neuron-Centric Scaling**: The model scales primarily in the high-dimensional **Neuron** dimension (n), rather than the dense latent dimension of Transformers. Parameters and state are localized to specific neuron pairs, mirroring biological structure.
* **Fixed Topologies as "Learned Programs"**: The model weights define sparse, scale-free graphs that act as the system's fixed ruleset:
    1. **The Causal Circuit (`Gx = E @ Dx`):** Implements signal propagation from y to x - a probabilistic form of **Modus Ponens** reasoning ("If concept A is active, trigger concept B"). The paper calls these the "wires".
    2. **The Output Circuit (`Gy = Dy @ E`):** Determines which neurons (y) should fire based on the attention-weighted context. The paper calls these the "prods".
* **Dynamic Synaptic State (Edge-Reweighting)**: Instead of a vector-based KV-cache, the model maintains "fast weights" on the edges between neurons (matrix σ). This state is updated via a **Hebbian Learning** rule ("neurons that fire together, wire together"), allowing the model to dynamically re-weight its own reasoning circuits over the duration of the context.
* **Sparse & Positive Activations**: The architecture enforces all activation vectors to be strictly positive and sparse. As noted in the paper, y activations are observed to be "extremely sparse" in practice (~3-5%). This design prevents the polysemantic "superposition" common in dense models, effectively filtering noise and isolating distinct logical paths.

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

This will print the input, target, and predicted boards to the console and generate visualizations:
- `combined_board_neuron.gif`: Board predictions + Neuron dynamics (shown in demo above)
- `combined_attention_sparsity.gif`: Board attention + Sparsity animation (shown in demo above)
- `sparsity_chart.png`: Static sparsity summary

#### Configuration
To adjust the model architecture or task parameters (e.g., board size, number of neurons), edit the `get_config()` function in `boardpath.py`.
