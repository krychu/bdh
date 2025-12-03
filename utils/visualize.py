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
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import torch
from torch import nn
from typing import List, Dict, Tuple, Optional
import io

from utils.build_boardpath_dataset import FLOOR, WALL, START, END, PATH


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


def add_watermark_to_frames(
    frames: List[Image.Image],
    text: str = "https://github.com/krychu/bdh",
    padding: int = 10,
    font_size: int = 14,
    opacity: float = 0.6
) -> List[Image.Image]:
    """
    Add watermark text to bottom-right corner of each frame.

    Args:
        frames: List of PIL Images
        text: Watermark text
        padding: Pixels from edge
        font_size: Font size (approximate, uses default font)
        opacity: Text opacity (0-1)

    Returns:
        List of watermarked PIL Images
    """
    from PIL import ImageDraw, ImageFont

    # Try to get a monospace font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    watermarked = []
    for frame in frames:
        # Convert to RGBA for transparency handling
        img = frame.convert("RGBA")

        # Create transparent overlay for text
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Position in bottom-right
        x = img.width - text_w - padding
        y = img.height - text_h - padding

        # Draw text with opacity
        alpha = int(255 * opacity)
        draw.text((x, y), text, font=font, fill=(80, 80, 80, alpha))

        # Composite and convert back to RGB
        img = Image.alpha_composite(img, overlay)
        watermarked.append(img.convert("RGB"))

    return watermarked


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


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    """Normalize positions to [-1, 1] range."""
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
    iterations: int = 100,
    **kwargs
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
    return normalize_positions(positions)




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
    selection_method: str = 'degree',
    token_mask: Optional[np.ndarray] = None
) -> List[Image.Image]:
    """
    Generate neuron dynamics animation.

    Args:
        x_frames: List of x activation tensors per layer, shape (T, N)
        y_frames: List of y activation tensors per layer, shape (T, N)
        model: BDH model instance
        config: Visualization configuration dict
        selection_method: 'degree' or 'weighted_degree'
        token_mask: Optional boolean array of shape (T,) to filter which tokens
                    to include when averaging activations. If None, all tokens used.
                    Use this to focus on path cells only.

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

    # Report token masking
    if token_mask is not None:
        n_masked = token_mask.sum()
        print(f"  Using token mask: {n_masked}/{len(token_mask)} tokens (path cells only)")
    else:
        print("  Using all tokens")

    # Compute layout (force-directed)
    print("  Computing force layout...")
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

        # Get activations (average over tokens, optionally masked)
        x_full = x_frames[layer_idx].cpu().numpy()  # (T, N)
        if token_mask is not None:
            x_act = x_full[token_mask].mean(axis=0)[selected_indices]
        else:
            x_act = x_full.mean(axis=0)[selected_indices]

        if layer_idx > 0:
            y_full = y_frames[layer_idx - 1].cpu().numpy()
            if token_mask is not None:
                y_prev = y_full[token_mask].mean(axis=0)[selected_indices]
            else:
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


# ============================================================================
# Board Attention Animation
# ============================================================================

def generate_board_attention_frames(
    output_frames: List[torch.Tensor],
    attn_frames: List[torch.Tensor],
    prob_frames: List[torch.Tensor],
    x_frames: List[torch.Tensor],
    board_size: int,
    input_board: Optional[torch.Tensor] = None
) -> List[Image.Image]:
    """
    Generate board animation with predictions, attention arrows, and x activation dots.

    Args:
        output_frames: List of (T,) tensors with predicted tokens per layer
        attn_frames: List of (T, T) tensors with attention scores per layer
        prob_frames: List of (T, V) tensors with class probabilities per layer
        x_frames: List of (T, N) tensors with x activations per layer
        board_size: Board size (e.g., 10)
        input_board: Optional (T,) input board for filtering wall-to-wall attention

    Returns:
        List of PIL Images
    """
    T = board_size * board_size

    # Cell colors
    colors = {
        FLOOR: np.array([0.95, 0.95, 0.95]),
        WALL: np.array([0.15, 0.15, 0.15]),
        START: np.array([0.2, 0.8, 0.2]),
        END: np.array([0.9, 0.2, 0.2]),
        PATH: np.array([1.0, 0.84, 0.0]),
    }

    images = []
    n_layers = len(output_frames)

    for layer_idx in range(n_layers):
        fig, ax = plt.subplots(figsize=(8, 8))

        predictions = output_frames[layer_idx]
        attn_scores = attn_frames[layer_idx]
        logits = prob_frames[layer_idx]
        x_act = x_frames[layer_idx].cpu().numpy()

        # Squeeze batch dimension if present
        if attn_scores.dim() == 3:
            attn_scores = attn_scores.squeeze(0)
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        # Get PATH probabilities for shading (apply softmax to logits)
        if torch.is_tensor(logits):
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        path_probs = probs[:, PATH]

        # Create board image from predictions
        pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        board_img = np.zeros((board_size, board_size, 3))
        for i in range(T):
            row, col = i // board_size, i % board_size
            cell_val = int(pred_np[i])
            base_color = colors.get(cell_val, colors[FLOOR])

            # Shade PATH cells by confidence
            if cell_val == PATH:
                confidence = path_probs[i] ** 2
                board_img[row, col] = base_color * confidence + colors[FLOOR] * (1 - confidence)
            else:
                board_img[row, col] = base_color

        ax.imshow(board_img, interpolation='nearest')

        # Add grid
        for i in range(board_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Draw x activation dots
        pct_active = (x_act > 0).mean(axis=1)  # (T,)
        act_min, act_max = pct_active.min(), pct_active.max()
        if act_max > act_min:
            activity_norm = (pct_active - act_min) / (act_max - act_min)
        else:
            activity_norm = np.zeros_like(pct_active)

        cols = np.arange(T) % board_size
        rows = np.arange(T) // board_size
        activity_scaled = activity_norm ** 1.5
        sizes = 180 * activity_scaled
        alphas = 0.85 * activity_scaled

        for i in range(T):
            ax.scatter(cols[i], rows[i], s=sizes[i], c='red', alpha=alphas[i],
                      edgecolors='none', zorder=5)

        # Draw attention arrows (top 30)
        attn_np = attn_scores.cpu().numpy() if torch.is_tensor(attn_scores) else attn_scores
        attn_copy = attn_np.copy()
        np.fill_diagonal(attn_copy, -np.inf)

        # Zero out wall-to-wall attention
        if input_board is not None:
            input_np = input_board.cpu().numpy() if torch.is_tensor(input_board) else input_board
            for i in range(T):
                for j in range(T):
                    if int(input_np[i]) == WALL and int(input_np[j]) == WALL:
                        attn_copy[i, j] = -np.inf

        # Find top-k attention values
        top_k = 30
        flat_attn = attn_copy.flatten()
        top_indices = np.argpartition(flat_attn, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-flat_attn[top_indices])]

        top_values = flat_attn[top_indices]
        val_min, val_max = top_values.min(), top_values.max()
        if val_max > val_min:
            top_normalized = (top_values - val_min) / (val_max - val_min)
        else:
            top_normalized = np.ones_like(top_values)

        for idx, flat_idx in enumerate(top_indices):
            src_idx = flat_idx // T
            tgt_idx = flat_idx % T
            src_row, src_col = src_idx // board_size, src_idx % board_size
            tgt_row, tgt_col = tgt_idx // board_size, tgt_idx % board_size
            alpha = 0.3 + 0.6 * top_normalized[idx]

            ax.annotate(
                '',
                xy=(tgt_col, tgt_row),
                xytext=(src_col, src_row),
                arrowprops=dict(
                    arrowstyle='->',
                    color='blue',
                    alpha=alpha,
                    linewidth=1.0,
                    shrinkA=3,
                    shrinkB=3,
                ),
                zorder=10
            )

        ax.set_xlim(-0.5, board_size - 0.5)
        ax.set_ylim(board_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Board Attention - Layer {layer_idx}', fontsize=12, fontweight='bold')

        # Legend
        legend_text = (
            f'Red dots: x activity\n'
            f'Blue arrows: attention\n'
            f'Gold: PATH prediction'
        )
        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               family='monospace')

        plt.tight_layout()
        images.append(fig_to_pil_image(fig))

    return images


# ============================================================================
# Neural Activity UMAP Animation
# ============================================================================

def compute_neuron_features_gy(model: nn.Module) -> np.ndarray:
    """
    Extract Gy = Dy·E features for UMAP layout.

    Gy captures neuron-to-neuron interaction in the signal (y) pathway.

    Returns:
        features: (N, N) array - each row is a neuron's interaction pattern
    """
    H, D, Nh = model.Dx.shape
    N = H * Nh

    Dy_flat = model.Dy.permute(1, 0, 2).reshape(D, -1)  # (D, N)
    E = model.E  # (N, D)
    Gy = Dy_flat.T @ E.T  # (N, D) @ (D, N) = (N, N)

    return Gy.detach().cpu().numpy()


def compute_umap_layout(
    neuron_features: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Compute 2D UMAP layout for neurons.

    Args:
        neuron_features: (N, F) feature matrix
        seed: Random seed

    Returns:
        positions: (N, 2) array of coordinates normalized to [0, 1]
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=seed
        )
        positions = reducer.fit_transform(neuron_features)
    except ImportError:
        print("  UMAP not available, falling back to PCA")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=seed)
        positions = pca.fit_transform(neuron_features)

    # Normalize to [0, 1]
    positions = positions - positions.min(axis=0)
    positions = positions / (positions.max() + 1e-8)

    return positions


def generate_neural_activity_frames(
    y_frames: List[torch.Tensor],
    model: nn.Module,
    target_board: Optional[torch.Tensor] = None,
    seed: int = 42
) -> List[Image.Image]:
    """
    Generate neural activity animation showing y activations on UMAP layout.

    Args:
        y_frames: List of (T, N) tensors with y activations per layer
        model: BDH model instance
        target_board: Optional (T,) target board - if provided, average y over PATH cells
        seed: Random seed for UMAP

    Returns:
        List of PIL Images
    """
    N = model.N

    # Compute UMAP layout (fixed across frames)
    print("  Computing Gy = Dy @ E for UMAP...")
    neuron_features = compute_neuron_features_gy(model)
    print(f"  Computing UMAP layout for {N} neurons...")
    positions = compute_umap_layout(neuron_features, seed)

    # Determine path mask
    path_mask = None
    n_path_cells = 0
    if target_board is not None:
        target_np = target_board.cpu().numpy() if torch.is_tensor(target_board) else target_board
        path_mask = (target_np == PATH)
        n_path_cells = path_mask.sum()
        print(f"  Averaging y over {n_path_cells} PATH cells")

    images = []
    n_layers = len(y_frames)

    for layer_idx in range(n_layers):
        fig, ax = plt.subplots(figsize=(8, 8))

        y_act_full = y_frames[layer_idx].cpu().numpy()  # (T, N)

        # Average over PATH cells or all cells
        if path_mask is not None and n_path_cells > 0:
            y_act = y_act_full[path_mask].mean(axis=0)  # (N,)
            label = f'PATH cells (n={n_path_cells})'
        else:
            y_act = y_act_full.mean(axis=0)
            label = 'All cells'

        # Normalize
        y_max = y_act.max() if y_act.max() > 0 else 1.0
        y_norm = y_act / y_max

        # Colors: gray for inactive, white for active
        gray = np.array([0.25, 0.25, 0.25])
        white = np.array([1.0, 1.0, 1.0])
        node_colors = np.zeros((N, 3))
        sizes = np.zeros(N)

        for i in range(N):
            y_val = min(y_norm[i], 1.0)
            node_colors[i] = gray * (1 - y_val) + white * y_val
            sizes[i] = 1.5 + 8.0 * (y_val ** 2)

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=sizes,
            c=node_colors,
            alpha=0.8,
            edgecolors='none'
        )

        # Stats
        if path_mask is not None and n_path_cells > 0:
            path_pcts = (y_act_full[path_mask] > 0).mean(axis=1) * 100
            avg_pct = path_pcts.mean()
            stats_text = f'y: {avg_pct:.1f}% avg/path\n{label}'
        else:
            per_cell_pct = (y_act_full > 0).mean(axis=1) * 100
            avg_pct = per_cell_pct.mean()
            stats_text = f'y: {avg_pct:.1f}% avg/cell\n{label}'

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                family='monospace')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Neural Activity (y) - Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_facecolor('#1a1a2e')

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='white', label='y active'),
            mpatches.Patch(facecolor='#404040', label='inactive'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        images.append(fig_to_pil_image(fig))

    return images


# ============================================================================
# Sparsity Chart (Static PNG)
# ============================================================================

# ============================================================================
# Simple Board Animation (no arrows/dots)
# ============================================================================

def generate_simple_board_frames(
    output_frames: List[torch.Tensor],
    board_size: int
) -> List[Image.Image]:
    """
    Generate simple board animation showing only predictions (no attention/dots).

    Args:
        output_frames: List of (T,) tensors with predicted tokens per layer
        board_size: Board size (e.g., 10)

    Returns:
        List of PIL Images
    """
    T = board_size * board_size

    colors = {
        FLOOR: np.array([0.95, 0.95, 0.95]),
        WALL: np.array([0.15, 0.15, 0.15]),
        START: np.array([0.2, 0.8, 0.2]),
        END: np.array([0.9, 0.2, 0.2]),
        PATH: np.array([1.0, 0.84, 0.0]),
    }

    images = []
    n_layers = len(output_frames)

    for layer_idx in range(n_layers):
        fig, ax = plt.subplots(figsize=(8, 8))

        predictions = output_frames[layer_idx]

        # Create board image from predictions (solid colors, no confidence shading)
        pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        board_img = np.zeros((board_size, board_size, 3))
        for i in range(T):
            row, col = i // board_size, i % board_size
            cell_val = int(pred_np[i])
            board_img[row, col] = colors.get(cell_val, colors[FLOOR])

        ax.imshow(board_img, interpolation='nearest')

        # Add grid
        for i in range(board_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        ax.set_xlim(-0.5, board_size - 0.5)
        ax.set_ylim(board_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Board Predictions - Layer {layer_idx}', fontsize=12, fontweight='bold')

        # Legend
        legend_text = 'Gold: PATH\nGreen: START\nRed: END'
        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               family='monospace')

        plt.tight_layout()
        images.append(fig_to_pil_image(fig))

    return images


# ============================================================================
# Animated Sparsity Chart
# ============================================================================

def generate_animated_sparsity_frames(
    x_frames: List[torch.Tensor],
    y_frames: List[torch.Tensor]
) -> List[Image.Image]:
    """
    Generate animated sparsity chart with layer indicator.

    Args:
        x_frames: List of (T, N) tensors with x activations per layer
        y_frames: List of (T, N) tensors with y activations per layer

    Returns:
        List of PIL Images (one per layer)
    """
    n_layers = len(y_frames)

    # Compute per-cell sparsity stats for y
    y_avg, y_min, y_max = [], [], []
    for layer_idx in range(n_layers):
        y_act = y_frames[layer_idx].cpu().numpy()
        per_cell = (y_act > 0).mean(axis=1) * 100
        y_avg.append(per_cell.mean())
        y_min.append(per_cell.min())
        y_max.append(per_cell.max())

    # Compute for x
    x_avg, x_min, x_max = [], [], []
    for layer_idx in range(n_layers):
        x_act = x_frames[layer_idx].cpu().numpy()
        per_cell = (x_act > 0).mean(axis=1) * 100
        x_avg.append(per_cell.mean())
        x_min.append(per_cell.min())
        x_max.append(per_cell.max())

    layers = list(range(n_layers))
    images = []

    for current_layer in range(n_layers):
        fig, ax = plt.subplots(figsize=(8, 8))

        # x activations (red)
        ax.fill_between(layers, x_min, x_max, color='red', alpha=0.15)
        ax.plot(layers, x_avg, 'r-', linewidth=2, label='x (gate)')
        ax.scatter(layers, x_avg, color='red', s=30, zorder=5)

        # y activations (blue)
        ax.fill_between(layers, y_min, y_max, color='blue', alpha=0.15)
        ax.plot(layers, y_avg, 'b-', linewidth=2, label='y (signal)')
        ax.scatter(layers, y_avg, color='blue', s=30, zorder=5)

        # Current layer indicator - vertical line
        ax.axvline(x=current_layer, color='black', linewidth=2, linestyle='--', alpha=0.7)

        # Highlight current points
        ax.scatter([current_layer], [x_avg[current_layer]], color='red', s=150,
                  edgecolors='black', linewidths=2, zorder=10)
        ax.scatter([current_layer], [y_avg[current_layer]], color='blue', s=150,
                  edgecolors='black', linewidths=2, zorder=10)

        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('% Neurons Active', fontsize=11)
        ax.set_title(f'Sparsity - Layer {current_layer}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_xticks(layers)
        ax.set_ylim(0, max(max(x_max), 100) * 1.05)

        # Stats box
        stats_text = (
            f'x: {x_avg[current_layer]:.1f}%\n'
            f'y: {y_avg[current_layer]:.1f}%'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               family='monospace')

        plt.tight_layout()
        images.append(fig_to_pil_image(fig))

    return images


# ============================================================================
# Side-by-Side GIF Combiner
# ============================================================================

def combine_frames_side_by_side(
    left_frames: List[Image.Image],
    right_frames: List[Image.Image],
    gap: int = 20,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> List[Image.Image]:
    """
    Combine two lists of frames side by side.

    Args:
        left_frames: List of PIL Images for left panel
        right_frames: List of PIL Images for right panel
        gap: Pixel gap between panels
        background_color: RGB tuple for gap color

    Returns:
        List of combined PIL Images
    """
    if len(left_frames) != len(right_frames):
        raise ValueError(f"Frame count mismatch: {len(left_frames)} vs {len(right_frames)}")

    combined = []
    for left, right in zip(left_frames, right_frames):
        # Resize to same height if needed
        left_w, left_h = left.size
        right_w, right_h = right.size

        if left_h != right_h:
            # Scale to match heights
            target_h = max(left_h, right_h)
            if left_h != target_h:
                scale = target_h / left_h
                left = left.resize((int(left_w * scale), target_h), Image.Resampling.LANCZOS)
                left_w, left_h = left.size
            if right_h != target_h:
                scale = target_h / right_h
                right = right.resize((int(right_w * scale), target_h), Image.Resampling.LANCZOS)
                right_w, right_h = right.size

        # Create combined image
        total_w = left_w + gap + right_w
        combined_img = Image.new('RGB', (total_w, left_h), background_color)
        combined_img.paste(left, (0, 0))
        combined_img.paste(right, (left_w + gap, 0))
        combined.append(combined_img)

    return combined


# ============================================================================
# Sparsity Chart (Static PNG)
# ============================================================================

def generate_sparsity_chart(
    x_frames: List[torch.Tensor],
    y_frames: List[torch.Tensor],
    save_path: str = 'sparsity_chart.png'
) -> Image.Image:
    """
    Generate static chart showing x and y sparsity across layers.

    Args:
        x_frames: List of (T, N) tensors with x activations per layer
        y_frames: List of (T, N) tensors with y activations per layer
        save_path: Path to save PNG

    Returns:
        PIL Image
    """
    n_layers = len(y_frames)

    # Compute per-cell sparsity stats for y
    y_avg, y_min, y_max = [], [], []
    for layer_idx in range(n_layers):
        y_act = y_frames[layer_idx].cpu().numpy()
        per_cell = (y_act > 0).mean(axis=1) * 100
        y_avg.append(per_cell.mean())
        y_min.append(per_cell.min())
        y_max.append(per_cell.max())

    # Compute for x
    x_avg, x_min, x_max = [], [], []
    for layer_idx in range(n_layers):
        x_act = x_frames[layer_idx].cpu().numpy()
        per_cell = (x_act > 0).mean(axis=1) * 100
        x_avg.append(per_cell.mean())
        x_min.append(per_cell.min())
        x_max.append(per_cell.max())

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(n_layers))

    # x activations (red)
    ax.fill_between(layers, x_min, x_max, color='red', alpha=0.15, label='x range')
    ax.plot(layers, x_avg, 'r-', linewidth=2, label='x')
    ax.scatter(layers, x_avg, color='red', s=10, zorder=5)

    # y activations (blue)
    ax.fill_between(layers, y_min, y_max, color='blue', alpha=0.15, label='y range')
    ax.plot(layers, y_avg, 'b-', linewidth=2, label='y')
    ax.scatter(layers, y_avg, color='blue', s=10, zorder=5)

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('% Neurons Active', fontsize=11)
    ax.set_title('Per-Cell Sparsity Across Layers', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))

    plt.tight_layout()

    # Save
    image = fig_to_pil_image(fig)
    image.save(save_path)

    return image
