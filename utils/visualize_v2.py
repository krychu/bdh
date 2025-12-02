"""
Two-Panel BDH Visualization Module

Panel 1 (Board):
- Dynamic communication between cells (attention edges)
- Evolving per-cell predictions with confidence heatmap
- Per-cell signal strength (border thickness from y activations)

Panel 2 (Neuron Map):
- Fixed anatomical layout from PCA of weight signatures
- Top-M neurons by weight participation
- Fixed gray edges from W_eff = E @ Dx
- Dynamic node coloring: red fill = x_l, blue ring = y_{l-1}
- Dynamic edge highlighting for causal flow
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import torch
from torch import nn
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
import io

from utils.build_boardpath_dataset import FLOOR, WALL, START, END, PATH


# ============================================================================
# Configuration Defaults
# ============================================================================

DEFAULT_CONFIG = {
    'M_neurons': 300,           # Candidate neurons (by weight participation)
    'k_attn': 2,                # Top-k attention edges per source token
    'w_min': 0.05,              # Minimum attention weight threshold
    'w_eff_threshold': 0.15,    # Minimum |W_eff| to show edge
    'max_edges': 2000,          # Cap on total edges
    'min_component_size': 5,    # Remove components with fewer neurons
    'layout_seed': 42,          # Random seed for layout
}


# ============================================================================
# Core Utilities
# ============================================================================

def fig_to_pil_image(fig) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return image


def add_watermark(fig, ax):
    """Add GitHub URL watermark to bottom-right corner of the plot."""
    ax.text(0.98, 0.02, 'https://github.com/krychu/bdh',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='bottom',
            horizontalalignment='right',
            color='black',
            alpha=1.0,
            family='monospace')


def save_gif(images: List[Image.Image], save_path: str, duration: int = 500):
    """Save a list of PIL images as an animated GIF."""
    if not images:
        raise ValueError("Cannot save empty image list")

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )


def normalize_robust(arr: np.ndarray, percentile_low: float = 5, percentile_high: float = 95) -> np.ndarray:
    """Normalize array using robust percentile-based normalization."""
    if arr.size == 0:
        return arr
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    if high - low > 1e-8:
        return np.clip((arr - low) / (high - low), 0, 1)
    return np.zeros_like(arr)


# ============================================================================
# Neuron Subset Selection
# ============================================================================

def select_neurons_by_degree(
    model: nn.Module,
    M: int,
    threshold: float = 0.1,
    weighted: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top M neurons by degree in Gx = E @ Dx.

    Args:
        model: BDH model
        M: Number of neurons to select
        threshold: Minimum |Gx[i,j]| to count as edge (only used when weighted=False)
        weighted: If True, use weighted degree (sum of all |Gx|), else count edges above threshold

    Returns:
        selected_indices: Array of original neuron indices
        scores: Degree scores for selected neurons
    """
    with torch.no_grad():
        H, D, Nh = model.Dx.shape
        N = H * Nh

        Dx_flat = model.Dx.permute(1, 0, 2).reshape(D, N)
        Gx = (model.E @ Dx_flat).cpu().numpy()  # (N, N)

        abs_Gx = np.abs(Gx)
        np.fill_diagonal(abs_Gx, 0)

        if weighted:
            # Weighted degree: sum of all |Gx| values (no threshold)
            in_degree = abs_Gx.sum(axis=0)   # sum over column (incoming)
            out_degree = abs_Gx.sum(axis=1)  # sum over row (outgoing)
        else:
            # Unweighted degree: count edges above threshold
            edges = abs_Gx > threshold
            in_degree = edges.sum(axis=0)
            out_degree = edges.sum(axis=1)

        # Total degree (undirected view)
        score = in_degree + out_degree

        M = min(M, N)
        selected_indices = np.argsort(score)[-M:][::-1]

        return selected_indices, score[selected_indices]


# Available selection methods
SELECTION_METHODS = {
    'degree': lambda m, M, **kw: select_neurons_by_degree(m, M, weighted=False, **kw),
    'weighted_degree': lambda m, M, **kw: select_neurons_by_degree(m, M, weighted=True, **kw),
}


def select_top_neurons(
    model: nn.Module,
    M: int,
    method: str = 'degree',
    **kwargs
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Select top M neurons using specified method.

    Args:
        model: BDH model
        M: Number of neurons to select
        method: One of 'degree', 'weighted_degree'
        **kwargs: Additional args passed to selection method (e.g., threshold)

    Returns:
        selected_indices: Array of original neuron indices (length M)
        index_map: Dict mapping original index -> new index (0 to M-1)
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from {list(SELECTION_METHODS.keys())}")

    selector = SELECTION_METHODS[method]
    selected_indices, _ = selector(model, M, **kwargs)

    index_map = {orig: new for new, orig in enumerate(selected_indices)}
    return selected_indices, index_map


def compute_neuron_signatures(model: nn.Module, selected_indices: np.ndarray) -> np.ndarray:
    """
    Compute signature vectors for selected neurons for PCA layout.

    Signature = concat(Dx[:, j], Dy[:, j], E[j, :])

    Returns:
        signatures: (M, signature_dim) array
    """
    with torch.no_grad():
        H, D, Nh = model.Dx.shape
        N = H * Nh

        Dx_flat = model.Dx.permute(1, 0, 2).reshape(D, N).cpu().numpy()  # (D, N)
        Dy_flat = model.Dy.permute(1, 0, 2).reshape(D, N).cpu().numpy()  # (D, N)
        E = model.E.cpu().numpy()  # (N, D)

        signatures = []
        for idx in selected_indices:
            sig = np.concatenate([
                Dx_flat[:, idx],
                Dy_flat[:, idx],
                E[idx, :]
            ])
            signatures.append(sig)

        return np.array(signatures)


def compute_pca_layout(signatures: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Compute 2D PCA layout for neurons.

    Returns:
        positions: (M, 2) array of 2D coordinates
    """
    if signatures.shape[0] < 2:
        return np.zeros((signatures.shape[0], 2))

    pca = PCA(n_components=2, random_state=seed)
    positions = pca.fit_transform(signatures)

    # Normalize to [-1, 1] range for plotting
    for dim in range(2):
        min_val = positions[:, dim].min()
        max_val = positions[:, dim].max()
        if max_val - min_val > 1e-8:
            positions[:, dim] = 2 * (positions[:, dim] - min_val) / (max_val - min_val) - 1

    return positions


def compute_force_layout(
    edges: List[Tuple[int, int]],
    weights: np.ndarray,
    M: int,
    seed: int = 42,
    iterations: int = 100
) -> np.ndarray:
    """
    Compute 2D force-directed layout using networkx.

    Args:
        edges: List of (src, tgt) edge tuples
        weights: Edge weights
        M: Number of nodes
        seed: Random seed
        iterations: Number of layout iterations

    Returns:
        positions: (M, 2) array of 2D coordinates
    """
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(M))

    # Add edges with weights (higher weight = stronger attraction)
    for idx, (src, tgt) in enumerate(edges):
        if idx < len(weights):
            # NetworkX spring layout uses 'weight' for edge strength
            G.add_edge(src, tgt, weight=weights[idx])

    # Compute spring layout
    pos_dict = nx.spring_layout(
        G,
        k=2.0 / np.sqrt(M),  # Optimal distance between nodes
        iterations=iterations,
        seed=seed,
        weight='weight'
    )

    # Convert to array
    positions = np.array([pos_dict[i] for i in range(M)])

    # Normalize to [-1, 1]
    for dim in range(2):
        min_val = positions[:, dim].min()
        max_val = positions[:, dim].max()
        if max_val - min_val > 1e-8:
            positions[:, dim] = 2 * (positions[:, dim] - min_val) / (max_val - min_val) - 1

    return positions


# ============================================================================
# Fixed Edge Topology
# ============================================================================

def compute_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for vectors.

    Args:
        vectors: (N, D) array where each row is a vector

    Returns:
        (N, N) cosine similarity matrix
    """
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    normalized = vectors / norms

    # Cosine similarity = dot product of normalized vectors
    return normalized @ normalized.T


def compute_w_eff(model: nn.Module) -> np.ndarray:
    """
    Compute W_eff = E @ Dx - the true neuron-to-neuron connectivity.

    This represents the signal flow: y_{l-1} -> E -> v* -> Dx -> x_l
    W_eff[i,j] = how much y[i] contributes to x[j]

    Returns:
        W_eff: (N, N) numpy array
    """
    with torch.no_grad():
        H, D, Nh = model.Dx.shape
        N = H * Nh

        # Dx: (H, D, N//H) -> (D, N)
        Dx_flat = model.Dx.permute(1, 0, 2).reshape(D, N)

        # E: (N, D)
        # W_eff = E @ Dx: (N, D) @ (D, N) = (N, N)
        W_eff = model.E @ Dx_flat

        return W_eff.detach().cpu().numpy()


def build_fixed_edges_from_w_eff(
    W_eff: np.ndarray,
    selected_indices: np.ndarray,
    threshold: float = 0.1,
    max_edges: int = 2000,
    min_component_size: int = 5
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Build fixed edge list from W_eff = E @ Dx (true connectivity).

    W_eff[i,j] represents how much y[i] contributes to x[j].
    We threshold to get sparse, meaningful edges, then filter out
    small connected components.

    Args:
        W_eff: (N, N) effective connectivity matrix
        selected_indices: Indices of neurons to include
        threshold: Minimum |W_eff| value to include edge
        max_edges: Cap on total edges to avoid clutter
        min_component_size: Remove components with fewer nodes than this

    Returns:
        edges: List of (src, tgt) in remapped indices (0 to M'-1)
        weights: Corresponding W_eff values
        kept_indices: Which of the original selected_indices were kept
    """
    import networkx as nx

    # Extract submatrix for selected neurons
    W_sub = W_eff[np.ix_(selected_indices, selected_indices)]
    M = len(selected_indices)

    # Find edges above threshold
    abs_W = np.abs(W_sub)
    np.fill_diagonal(abs_W, 0)  # No self-edges

    # Get all edges above threshold
    rows, cols = np.where(abs_W > threshold)
    edge_weights = W_sub[rows, cols]

    # If too many edges, take top by absolute value
    if len(rows) > max_edges:
        top_idx = np.argsort(np.abs(edge_weights))[-max_edges:]
        rows = rows[top_idx]
        cols = cols[top_idx]
        edge_weights = edge_weights[top_idx]

    # Build graph to find connected components
    G = nx.Graph()
    G.add_nodes_from(range(M))
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c))

    # Find connected components and filter small ones
    components = list(nx.connected_components(G))
    large_components = [c for c in components if len(c) >= min_component_size]

    if not large_components:
        # If no large components, keep the largest one
        if components:
            large_components = [max(components, key=len)]

    # Get nodes to keep
    kept_nodes = set()
    for comp in large_components:
        kept_nodes.update(comp)

    # Create mapping from old index to new index
    kept_nodes_sorted = sorted(kept_nodes)
    old_to_new = {old: new for new, old in enumerate(kept_nodes_sorted)}

    # Filter edges to only include kept nodes, remap indices
    final_edges = []
    final_weights = []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        if r in kept_nodes and c in kept_nodes:
            final_edges.append((old_to_new[r], old_to_new[c]))
            final_weights.append(edge_weights[idx])

    # Return which original selected_indices are kept
    kept_original_indices = selected_indices[kept_nodes_sorted]

    return final_edges, np.array(final_weights), kept_original_indices


# ============================================================================
# Panel 1: Board Visualization
# ============================================================================

def draw_board_panel(
    ax,
    logits: np.ndarray,
    y_activations: np.ndarray,
    attn_weights: np.ndarray,
    input_board: np.ndarray,
    board_size: int,
    k_attn: int,
    w_min: float,
    layer_idx: int
):
    """
    Draw Panel 1: Board with predictions, confidence, and attention edges.

    Args:
        ax: Matplotlib axis
        logits: (T, V) per-cell logits
        y_activations: (N,) averaged y activations per neuron
        attn_weights: (T, T) attention weights (averaged over heads)
        input_board: (board_size, board_size) original input
        board_size: Size of the board
        k_attn: Number of top attention edges per source token
        w_min: Minimum attention weight threshold
        layer_idx: Current layer index
    """
    T = board_size * board_size

    # Compute predictions and confidence
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = probs.argmax(axis=-1)
    confidence = probs.max(axis=-1)

    # Board cell colors based on prediction
    cmap = ListedColormap(['white', 'black', 'lime', 'red', 'gold'])
    board_pred = predictions.reshape(board_size, board_size)
    conf_map = confidence.reshape(board_size, board_size)

    # Draw cells with confidence-based alpha
    for row in range(board_size):
        for col in range(board_size):
            cell_val = board_pred[row, col]
            cell_conf = conf_map[row, col]

            # Cell colors
            colors = ['white', 'black', 'lime', 'red', 'gold']
            cell_color = colors[cell_val]

            # Draw rectangle with confidence-modulated alpha
            rect = Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                facecolor=cell_color,
                edgecolor='gray',
                linewidth=1,
                alpha=0.3 + 0.7 * cell_conf
            )
            ax.add_patch(rect)

            # Add glyph
            glyphs = ['.', '#', 'S', 'E', '*']
            glyph = glyphs[cell_val]
            text_color = 'white' if cell_val == 1 else 'black'
            ax.text(col, row, glyph, ha='center', va='center',
                   fontsize=10, fontweight='bold', color=text_color)

    # Draw attention edges (top-k per source)
    attn_np = attn_weights.copy()

    # Apply numerically stable softmax normalization per source token
    attn_max = attn_np.max(axis=1, keepdims=True)
    attn_exp = np.exp(attn_np - attn_max)  # Subtract max for stability
    attn_softmax = attn_exp / (attn_exp.sum(axis=1, keepdims=True) + 1e-8)

    for src_idx in range(T):
        src_row, src_col = src_idx // board_size, src_idx % board_size

        # Get attention weights from this source
        weights = attn_softmax[src_idx].copy()
        weights[src_idx] = 0  # Exclude self-attention

        # Top-k targets
        if k_attn < T - 1:
            top_k_idx = np.argpartition(weights, -k_attn)[-k_attn:]
        else:
            top_k_idx = np.where(weights > 0)[0]

        for tgt_idx in top_k_idx:
            if weights[tgt_idx] < w_min:
                continue

            tgt_row, tgt_col = tgt_idx // board_size, tgt_idx % board_size

            # Draw edge with weight-proportional alpha/width
            alpha = min(0.8, weights[tgt_idx] * 2)
            width = 0.5 + weights[tgt_idx] * 2

            ax.annotate(
                '', xy=(tgt_col, tgt_row), xytext=(src_col, src_row),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=(0.2, 0.4, 0.8, alpha),
                    lw=width,
                    connectionstyle='arc3,rad=0.1'
                )
            )

    ax.set_xlim(-0.6, board_size - 0.4)
    ax.set_ylim(board_size - 0.4, -0.6)  # Invert y-axis
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Board Predictions - Layer {layer_idx}', fontsize=12, fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', label='Floor'),
        mpatches.Patch(facecolor='black', edgecolor='gray', label='Wall'),
        mpatches.Patch(facecolor='lime', edgecolor='black', label='Start'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='End'),
        mpatches.Patch(facecolor='gold', edgecolor='black', label='Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=8, frameon=True)


# ============================================================================
# Panel 2: Neuron Map Visualization
# ============================================================================

def draw_neuron_panel(
    ax,
    positions: np.ndarray,
    fixed_edges: List[Tuple[int, int]],
    fixed_weights: np.ndarray,
    x_activations: np.ndarray,
    y_prev_activations: np.ndarray,
    layer_idx: int,
    M_neurons: int,
    N_total: int
):
    """
    Draw Panel 2: Neuron map with W_eff edges and dynamic activations.

    Edges from W_eff = E @ Dx represent true y_{l-1} -> x_l connectivity.
    Flow = y_prev[src] * W_eff[src,tgt] (signal actually sent through edge)

    Args:
        ax: Matplotlib axis
        positions: (M, 2) array of neuron positions
        fixed_edges: List of (src, tgt) tuples from W_eff thresholding
        fixed_weights: W_eff[src,tgt] values (can be negative)
        x_activations: (M,) current x activations (raw)
        y_prev_activations: (M,) previous y activations (raw)
        layer_idx: Current layer index
        M_neurons: Number of selected neurons
        N_total: Total neurons in model
    """
    M = len(positions)

    # Normalize activations for display
    x_norm = normalize_robust(x_activations)
    y_norm = normalize_robust(y_prev_activations)

    gray_base = np.array([0.8, 0.8, 0.8])
    red_color = np.array([1.0, 0.2, 0.2])
    blue_color = np.array([0.0, 0.4, 1.0])

    # Compute flow on each edge:
    # flow[i->j] = ReLU(y_prev[i] * Gx[i,j] * x[j])
    # Modulated by destination x activation to show actual received signal
    edge_flows = []
    for idx, (src, tgt) in enumerate(fixed_edges):
        flow = y_prev_activations[src] * fixed_weights[idx] * x_activations[tgt]
        edge_flows.append(max(0, flow))  # ReLU: discard negative

    edge_flows = np.array(edge_flows) if edge_flows else np.array([])
    flow_norm = normalize_robust(edge_flows) if len(edge_flows) > 0 else np.array([])

    # --- Draw edges: gray (no flow) to black (high flow) gradient ---
    # Base gray for Gx structure, darkens with signal flow
    active_edge_count = 0
    for idx, (src, tgt) in enumerate(fixed_edges):
        flow_val = flow_norm[idx] if idx < len(flow_norm) else 0

        # Gray level: 0.7 (light gray, no flow) -> 0.0 (black, max flow)
        gray_level = 0.7 * (1 - flow_val)

        # Width: thin (no flow) -> slightly thicker (high flow)
        width = 0.4 + 1.0 * flow_val

        # Alpha: more visible base, solid when active
        alpha = 0.5 + 0.5 * flow_val

        color = (gray_level, gray_level, gray_level, alpha)

        if flow_val > 0.05:
            active_edge_count += 1

        ax.plot(
            [positions[src, 0], positions[tgt, 0]],
            [positions[src, 1], positions[tgt, 1]],
            color=color,
            linewidth=width,
            zorder=1 + flow_val  # Higher flow = drawn on top
        )

    # Draw nodes
    node_colors = []
    ring_colors = []
    ring_widths = []

    for i in range(M):
        # Fill color based on x activation (red intensity)
        # When x=0, neuron is gray; when x>0, neuron becomes red
        x_val = x_norm[i]
        fill_color = (1 - x_val) * gray_base + x_val * red_color
        node_colors.append(fill_color)

        # Ring based on y_prev (blue)
        # When y_prev=0, no ring (alpha=0, width=0)
        ring_colors.append((*blue_color, y_norm[i]))
        ring_widths.append(2.5 * y_norm[i])

    # Draw nodes with rings - fixed size
    ax.scatter(
        positions[:, 0], positions[:, 1],
        c=node_colors,
        s=40,  # Fixed size
        edgecolors=[rc[:3] for rc in ring_colors],
        linewidths=ring_widths,
        zorder=3
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Neuron Dynamics - Layer {layer_idx}', fontsize=12, fontweight='bold')

    # Legend
    legend_text = (
        f'Blue ring: y_{{l-1}} (source)\n'
        f'Gray->Black: signal flow\n'
        f'Red fill: x_l (destination)\n'
        f'Active edges: {active_edge_count}\n'
        f'Neurons: {M_neurons}/{N_total}'
    )
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           family='monospace')


# ============================================================================
# Main Animation Generator
# ============================================================================

def generate_two_panel_animation(
    input_board: torch.Tensor,
    target_board: torch.Tensor,  # For path-only averaging
    output_frames: List[torch.Tensor],
    x_frames: List[torch.Tensor],
    y_frames: List[torch.Tensor],
    attn_frames: List[torch.Tensor],
    logits_frames: List[torch.Tensor],
    model: nn.Module,
    board_size: int,
    config: Optional[Dict] = None,
    selection_method: str = 'degree'
) -> List[Image.Image]:
    """
    Generate two-panel animation showing board reasoning and neuron dynamics.

    Args:
        input_board: (board_size, board_size) input board tensor
        output_frames: List of predicted token tensors per layer
        x_frames: List of x activation tensors per layer
        y_frames: List of y activation tensors per layer
        attn_frames: List of attention score tensors per layer
        logits_frames: List of logit tensors per layer
        model: BDH model instance
        board_size: Size of the board
        config: Visualization configuration dict
        selection_method: Neuron selection method (see SELECTION_METHODS)

    Returns:
        List of PIL images for animation
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    L = len(output_frames)
    N = model.N

    # Pre-compute fixed structures
    print(f"  Selecting candidate neurons using '{selection_method}'...")
    candidate_indices, _ = select_top_neurons(
        model, cfg['M_neurons'],
        method=selection_method,
        threshold=cfg['w_eff_threshold']
    )
    print(f"    {len(candidate_indices)} candidates out of {N}")

    # Compute W_eff = E @ Dx (true connectivity)
    print("  Computing W_eff = E @ Dx...")
    W_eff = compute_w_eff(model)

    # Build edges from W_eff with threshold, filter small components
    min_comp = cfg.get('min_component_size', 5)
    print(f"  Building edges from W_eff (threshold={cfg['w_eff_threshold']}, min_component={min_comp})...")
    fixed_edges, fixed_weights, selected_indices = build_fixed_edges_from_w_eff(
        W_eff,
        candidate_indices,
        threshold=cfg['w_eff_threshold'],
        max_edges=cfg['max_edges'],
        min_component_size=min_comp
    )
    M = len(selected_indices)
    print(f"    {len(fixed_edges)} edges, {M} neurons after filtering disconnected")

    # Compute force-directed layout based on edge structure
    print("  Computing force-directed layout...")
    positions = compute_force_layout(
        fixed_edges,
        np.abs(fixed_weights),  # Use absolute weights for layout
        M,
        seed=cfg['layout_seed'],
        iterations=150
    )

    # Generate frames
    print("  Generating frames...")
    images = []

    for layer_idx in range(L):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Get data for this layer
        logits = logits_frames[layer_idx][0].cpu().numpy()  # (T, V) first sample
        attn = attn_frames[layer_idx][0].cpu().numpy()  # (T, T) first sample

        # x_frames and y_frames are (T, N) - average over all tokens
        x_full = x_frames[layer_idx].cpu().numpy()  # (T, N)
        x_act = x_full.mean(axis=0)[selected_indices]  # (M,) averaged over all tokens

        if layer_idx > 0:
            y_full = y_frames[layer_idx - 1].cpu().numpy()  # (T, N)
            y_prev = y_full.mean(axis=0)[selected_indices]  # (M,)
        else:
            y_prev = np.zeros_like(x_act)

        # Panel 1: Board
        draw_board_panel(
            axes[0],
            logits,
            x_act,  # Use x for signal strength visualization
            attn,
            input_board.numpy(),
            board_size,
            cfg['k_attn'],
            cfg['w_min'],
            layer_idx
        )

        # Panel 2: Neuron Map (pass raw activations, normalization done inside)
        draw_neuron_panel(
            axes[1],
            positions,
            fixed_edges,
            fixed_weights,
            x_act,
            y_prev,
            layer_idx,
            M,
            N
        )

        add_watermark(fig, axes[1])
        plt.tight_layout()

        images.append(fig_to_pil_image(fig))

    print(f"  Generated {len(images)} frames")
    return images
