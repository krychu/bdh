"""
BDH Neuron Dynamics Visualization

Visualizes signal flow through the neuron graph Gx = E @ Dx:
- Fixed layout from force-directed algorithm based on Gx connectivity
- Neurons selected by degree in Gx
- Gray edges show Gx structure, darken with signal flow
- Red fill = x_l activation (destination)
- Blue ring = y_{l-1} activation (source)
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from typing import List, Dict, Tuple, Optional
import io


# ============================================================================
# Configuration Defaults
# ============================================================================

DEFAULT_CONFIG = {
    'M_neurons': 300,           # Candidate neurons to consider
    'w_eff_threshold': 0.15,    # Minimum |Gx| to show edge
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
# Neuron Selection
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
            in_degree = abs_Gx.sum(axis=0)
            out_degree = abs_Gx.sum(axis=1)
        else:
            # Unweighted degree: count edges above threshold
            edges = abs_Gx > threshold
            in_degree = edges.sum(axis=0)
            out_degree = edges.sum(axis=1)

        score = in_degree + out_degree

        M = min(M, N)
        selected_indices = np.argsort(score)[-M:][::-1]

        return selected_indices, score[selected_indices]


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


# ============================================================================
# Graph Computation
# ============================================================================

def compute_w_eff(model: nn.Module) -> np.ndarray:
    """
    Compute Gx = E @ Dx - the neuron-to-neuron connectivity graph.

    This represents the signal flow: y_{l-1} -> E -> v* -> Dx -> x_l
    Gx[i,j] = how much y[i] contributes to x[j]

    Returns:
        Gx: (N, N) numpy array
    """
    with torch.no_grad():
        H, D, Nh = model.Dx.shape
        N = H * Nh

        Dx_flat = model.Dx.permute(1, 0, 2).reshape(D, N)
        Gx = model.E @ Dx_flat  # (N, D) @ (D, N) = (N, N)

        return Gx.detach().cpu().numpy()


def build_fixed_edges(
    Gx: np.ndarray,
    selected_indices: np.ndarray,
    threshold: float = 0.1,
    max_edges: int = 2000,
    min_component_size: int = 5
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Build edge list from Gx connectivity graph.

    Args:
        Gx: (N, N) connectivity matrix
        selected_indices: Indices of neurons to include
        threshold: Minimum |Gx| value to include edge
        max_edges: Cap on total edges
        min_component_size: Remove components with fewer nodes

    Returns:
        edges: List of (src, tgt) in remapped indices
        weights: Corresponding Gx values
        kept_indices: Which neurons were kept after filtering
    """
    import networkx as nx

    # Extract submatrix for selected neurons
    Gx_sub = Gx[np.ix_(selected_indices, selected_indices)]
    M = len(selected_indices)

    # Find edges above threshold
    abs_Gx = np.abs(Gx_sub)
    np.fill_diagonal(abs_Gx, 0)

    rows, cols = np.where(abs_Gx > threshold)
    edge_weights = Gx_sub[rows, cols]

    # Cap edges if too many
    if len(rows) > max_edges:
        top_idx = np.argsort(np.abs(edge_weights))[-max_edges:]
        rows = rows[top_idx]
        cols = cols[top_idx]
        edge_weights = edge_weights[top_idx]

    # Build graph for connected components
    G = nx.Graph()
    G.add_nodes_from(range(M))
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c))

    # Filter small components
    components = list(nx.connected_components(G))
    large_components = [c for c in components if len(c) >= min_component_size]

    if not large_components and components:
        large_components = [max(components, key=len)]

    kept_nodes = set()
    for comp in large_components:
        kept_nodes.update(comp)

    # Remap indices
    kept_nodes_sorted = sorted(kept_nodes)
    old_to_new = {old: new for new, old in enumerate(kept_nodes_sorted)}

    final_edges = []
    final_weights = []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        if r in kept_nodes and c in kept_nodes:
            final_edges.append((old_to_new[r], old_to_new[c]))
            final_weights.append(edge_weights[idx])

    kept_original_indices = selected_indices[kept_nodes_sorted]

    return final_edges, np.array(final_weights), kept_original_indices


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

    for idx, (src, tgt) in enumerate(edges):
        if idx < len(weights):
            G.add_edge(src, tgt, weight=weights[idx])

    pos_dict = nx.spring_layout(
        G,
        k=2.0 / np.sqrt(M),
        iterations=iterations,
        seed=seed,
        weight='weight'
    )

    positions = np.array([pos_dict[i] for i in range(M)])

    # Normalize to [-1, 1]
    for dim in range(2):
        min_val = positions[:, dim].min()
        max_val = positions[:, dim].max()
        if max_val - min_val > 1e-8:
            positions[:, dim] = 2 * (positions[:, dim] - min_val) / (max_val - min_val) - 1

    return positions


# ============================================================================
# Neuron Panel Visualization
# ============================================================================

def draw_neuron_panel(
    ax,
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    edge_weights: np.ndarray,
    x_activations: np.ndarray,
    y_prev_activations: np.ndarray,
    layer_idx: int,
    M_neurons: int,
    N_total: int
):
    """
    Draw neuron map with Gx edges and dynamic activations.

    Args:
        ax: Matplotlib axis
        positions: (M, 2) array of neuron positions
        edges: List of (src, tgt) tuples
        edge_weights: Gx[src,tgt] values
        x_activations: (M,) current x activations
        y_prev_activations: (M,) previous y activations
        layer_idx: Current layer index
        M_neurons: Number of neurons shown
        N_total: Total neurons in model
    """
    M = len(positions)

    # Normalize activations
    x_norm = normalize_robust(x_activations)
    y_norm = normalize_robust(y_prev_activations)

    gray_base = np.array([0.8, 0.8, 0.8])
    red_color = np.array([1.0, 0.2, 0.2])
    blue_color = np.array([0.0, 0.4, 1.0])

    # Compute flow: y_prev[src] * Gx[src,tgt] * x[tgt]
    edge_flows = []
    for idx, (src, tgt) in enumerate(edges):
        flow = y_prev_activations[src] * edge_weights[idx] * x_activations[tgt]
        edge_flows.append(max(0, flow))

    edge_flows = np.array(edge_flows) if edge_flows else np.array([])
    flow_norm = normalize_robust(edge_flows) if len(edge_flows) > 0 else np.array([])

    # Draw edges
    active_edge_count = 0
    for idx, (src, tgt) in enumerate(edges):
        flow_val = flow_norm[idx] if idx < len(flow_norm) else 0

        gray_level = 0.7 * (1 - flow_val)
        width = 0.4 + 1.0 * flow_val
        alpha = 0.5 + 0.5 * flow_val

        color = (gray_level, gray_level, gray_level, alpha)

        if flow_val > 0.05:
            active_edge_count += 1

        ax.plot(
            [positions[src, 0], positions[tgt, 0]],
            [positions[src, 1], positions[tgt, 1]],
            color=color,
            linewidth=width,
            zorder=1 + flow_val
        )

    # Draw nodes
    node_colors = []
    ring_colors = []
    ring_widths = []

    for i in range(M):
        x_val = x_norm[i]
        fill_color = (1 - x_val) * gray_base + x_val * red_color
        node_colors.append(fill_color)

        ring_colors.append((*blue_color, y_norm[i]))
        ring_widths.append(2.5 * y_norm[i])

    ax.scatter(
        positions[:, 0], positions[:, 1],
        c=node_colors,
        s=40,
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
# Animation Generator
# ============================================================================

def generate_neuron_animation(
    x_frames: List[torch.Tensor],
    y_frames: List[torch.Tensor],
    model: nn.Module,
    config: Optional[Dict] = None,
    selection_method: str = 'degree'
) -> List[Image.Image]:
    """
    Generate neuron dynamics animation.

    Args:
        x_frames: List of x activation tensors per layer, shape (T, N)
        y_frames: List of y activation tensors per layer, shape (T, N)
        model: BDH model instance
        config: Visualization configuration dict
        selection_method: 'degree' or 'weighted_degree'

    Returns:
        List of PIL images for animation
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    L = len(x_frames)
    N = model.N

    # Select neurons
    print(f"  Selecting neurons using '{selection_method}'...")
    candidate_indices, _ = select_top_neurons(
        model, cfg['M_neurons'],
        method=selection_method,
        threshold=cfg['w_eff_threshold']
    )
    print(f"    {len(candidate_indices)} candidates out of {N}")

    # Compute Gx
    print("  Computing Gx = E @ Dx...")
    Gx = compute_w_eff(model)

    # Build edges
    min_comp = cfg.get('min_component_size', 5)
    print(f"  Building edges (threshold={cfg['w_eff_threshold']}, min_component={min_comp})...")
    edges, edge_weights, selected_indices = build_fixed_edges(
        Gx,
        candidate_indices,
        threshold=cfg['w_eff_threshold'],
        max_edges=cfg['max_edges'],
        min_component_size=min_comp
    )
    M = len(selected_indices)
    print(f"    {len(edges)} edges, {M} neurons after filtering")

    # Compute layout
    print("  Computing layout...")
    positions = compute_force_layout(
        edges,
        np.abs(edge_weights),
        M,
        seed=cfg['layout_seed'],
        iterations=150
    )

    # Generate frames
    print("  Generating frames...")
    images = []

    for layer_idx in range(L):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Get activations (average over tokens)
        x_full = x_frames[layer_idx].cpu().numpy()  # (T, N)
        x_act = x_full.mean(axis=0)[selected_indices]

        if layer_idx > 0:
            y_full = y_frames[layer_idx - 1].cpu().numpy()
            y_prev = y_full.mean(axis=0)[selected_indices]
        else:
            y_prev = np.zeros_like(x_act)

        draw_neuron_panel(
            ax,
            positions,
            edges,
            edge_weights,
            x_act,
            y_prev,
            layer_idx,
            M,
            N
        )

        plt.tight_layout()
        images.append(fig_to_pil_image(fig))

    print(f"  Generated {len(images)} frames")
    return images
