import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
from PIL import Image
import torch
from typing import List, Optional, Tuple
import networkx as nx
from utils.build_boardpath_dataset import FLOOR, WALL, START, END, PATH
import io

# ============================================================================
# Core Utilities
# ============================================================================

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

def fig_to_pil_image(fig) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return image

def normalize_array(arr: np.ndarray, vmin: float = 0, vmax: float = None) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    if vmax is None:
        vmax = arr.max()
    if vmax > 0:
        norm = Normalize(vmin=vmin, vmax=vmax)
        return norm(arr)
    return np.zeros_like(arr)

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

# ============================================================================
# Color Computation
# ============================================================================
def compute_dual_network_node_colors(
    y_activations: np.ndarray,
    x_activations: np.ndarray,
    blue_color: np.ndarray = np.array([0.012, 0.376, 1.0]),
    red_color: np.ndarray = np.array([1.0, 0.164, 0.164]),
    gray_base: np.ndarray = np.array([0.75, 0.75, 0.75])
) -> List[Tuple[float, float, float]]:
    """
    Compute dual-network node colors: blend blue (y) and red (x).

    Args:
        y_activations: Normalized y activations [0, 1]
        x_activations: Normalized x activations [0, 1]
        blue_color: RGB for y network
        red_color: RGB for x network
        gray_base: RGB for inactive state

    Returns:
        List of RGB tuples
    """
    colors = []
    for y_val, x_val in zip(y_activations, x_activations):
        total = y_val + x_val
        if total > 0:
            # Weighted blend based on activation strengths
            weight_y = y_val / total
            weight_x = x_val / total
            blended_color = weight_y * blue_color + weight_x * red_color

            # Interpolate from gray to blended color
            intensity = max(y_val, x_val)
            final_color = gray_base + intensity * (blended_color - gray_base)
        else:
            final_color = gray_base

        colors.append(tuple(final_color))

    return colors

def compute_dual_network_edge_colors_and_widths(
    activations: np.ndarray,
    color: np.ndarray,
    gray_base: np.ndarray = np.array([0.75, 0.75, 0.75]),
    width_range: Tuple[float, float] = (0.3, 1.5),
    alpha: float = 0.8
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """
    Compute edge colors and widths for dual-network visualization.

    Args:
        activations: Normalized activation values [0, 1]
        color: Target RGB color (blue or red)
        gray_base: Base gray RGB
        width_range: (min_width, max_width)
        alpha: Alpha channel value

    Returns:
        Tuple of (colors, widths)
    """
    colors = []
    widths = []

    for act_val in activations:
        # Blend from gray to target color (linear interpolation)
        r = gray_base[0] + act_val * (color[0] - gray_base[0])
        g = gray_base[1] + act_val * (color[1] - gray_base[1])
        b = gray_base[2] + act_val * (color[2] - gray_base[2])
        colors.append((r, g, b, alpha))

        # Width proportional to activation
        width = width_range[0] + act_val * (width_range[1] - width_range[0])
        widths.append(width)

    return colors, widths

# ============================================================================
# Topology Extraction and Graph Building
# ============================================================================

def get_parameter_topology(model, topology_type: str = 'e_dx') -> torch.Tensor:
    """
    Extract N×N topology from model parameters.

    Args:
        model: BDH model instance
        topology_type: 'e_dx', 'dx_coact', or 'dy_coact'

    Returns:
        topology: (N, N) tensor with connection strengths
    """
    H, D, Nh = model.Dx.shape
    N = H * Nh

    # Reshape Dx and Dy from (H, D, N//H) to (D, N)
    Dx_reshaped = model.Dx.permute(1, 0, 2).reshape(D, N)
    Dy_reshaped = model.Dy.permute(1, 0, 2).reshape(D, N)

    if topology_type == 'e_dx':
        topology = model.E @ Dx_reshaped
    elif topology_type == 'dx_coact':
        topology = Dx_reshaped.T @ Dx_reshaped
    elif topology_type == 'dy_coact':
        topology = Dy_reshaped.T @ Dy_reshaped
    else:
        raise ValueError(f"Unknown topology_type: {topology_type}")

    return topology.abs().detach()

def build_topology_graph(
    topology_matrix: torch.Tensor,
    top_k_edges: int
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Build edge list from topology matrix using top-K threshold.

    Args:
        topology_matrix: (N, N) tensor
        top_k_edges: Number of strongest edges to keep

    Returns:
        Tuple of (edge_list, edge_weights)
    """
    topology_np = topology_matrix.cpu().numpy()
    N = topology_np.shape[0]

    # Use only upper triangle to avoid duplicates, then pick exact top_k edges
    iu, ju = np.triu_indices(N, k=1)
    weights = (topology_np[iu, ju] + topology_np[ju, iu]) / 2  # Symmetrize

    if top_k_edges < len(weights):
        top_idx = np.argpartition(weights, -top_k_edges)[-top_k_edges:]
        # Sort selected edges by weight descending for stability
        top_idx = top_idx[np.argsort(-weights[top_idx])]
    else:
        top_idx = np.arange(len(weights))

    edge_list = [(int(iu[k]), int(ju[k])) for k in top_idx]
    edge_weights = weights[top_idx]

    return edge_list, edge_weights

def extract_hub_subgraph(
    edge_list: List[Tuple[int, int]],
    N: int,
    min_component_size: int = 1
) -> Tuple[List[int], dict, List[Tuple[int, int]]]:
    """
    Extract connected neurons (hub) and remap edge list, filtering small components.

    Args:
        edge_list: List of edges with original neuron indices
        N: Total number of neurons
        min_component_size: Minimum size of connected components to keep (default: 1 = keep all)

    Returns:
        Tuple of (connected_neurons, neuron_map, remapped_edges)
    """
    # Build temporary graph to find connected components
    G_temp = nx.Graph()
    G_temp.add_edges_from(edge_list)

    # Find connected components
    components = list(nx.connected_components(G_temp))

    # Filter components by size
    large_components = [comp for comp in components if len(comp) >= min_component_size]
    small_components = [comp for comp in components if len(comp) < min_component_size]

    # Silently filter small components

    # Collect neurons from large components only
    connected_neurons = set()
    for comp in large_components:
        connected_neurons.update(comp)
    connected_neurons = sorted(connected_neurons)

    # Create mapping: old_idx → new_idx
    neuron_map = {old_idx: new_idx for new_idx, old_idx in enumerate(connected_neurons)}

    # Remap edges (only those where both endpoints are in large components)
    remapped_edges = []
    for i, j in edge_list:
        if i in neuron_map and j in neuron_map:
            remapped_edges.append((neuron_map[i], neuron_map[j]))

    return connected_neurons, neuron_map, remapped_edges

def compute_graph_layout(
    edges: List[Tuple[int, int]],
    N: int,
    seed: int = 42
) -> dict:
    """
    Compute force-directed layout for graph.

    Args:
        edges: List of edges
        N: Number of nodes
        seed: Random seed for reproducibility

    Returns:
        Position dictionary {node_id: (x, y)}
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, k=1/np.sqrt(N), iterations=50, seed=seed)
    return pos

# ============================================================================
# Edge Activation Computation
# ============================================================================

def compute_edge_activations_signal_flow(
    edge_list: List[Tuple[int, int]],
    source_activations: np.ndarray,
    topology_matrix: np.ndarray,
    target_activations: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute causal signal flow.
    If target_activations is provided: Flow = |Source * W * Target| (Successful transmission)
    If not: Flow = |Source * W| (Broadcast magnitude)
    """
    activations = []
    for i, j in edge_list:
        # i -> j
        flow_i_to_j = abs(source_activations[i] * topology_matrix[i, j])
        # j -> i
        flow_j_to_i = abs(source_activations[j] * topology_matrix[j, i])

        # Gate by target activation if provided
        if target_activations is not None:
            flow_i_to_j *= target_activations[j]  # Target is j
            flow_j_to_i *= target_activations[i]  # Target is i

        activations.append((flow_i_to_j + flow_j_to_i) / 2)
    return np.array(activations)

def compute_edge_activations_coactivation(
    edge_list: List[Tuple[int, int]],
    activations: np.ndarray
) -> np.ndarray:
    """Compute edge activations using co-activation (product of node activations)."""
    edge_activations = []
    for i, j in edge_list:
        co_activation = activations[i] * activations[j]
        edge_activations.append(co_activation)
    return np.array(edge_activations)

# ============================================================================
# Board Visualization
# ============================================================================

def generate_board_frames(
    output_frames: List[torch.Tensor],
    board_size: int
) -> List[Image.Image]:
    """
    Generate PIL images of board predictions through layers.

    Args:
        output_frames: List of tensors, each shape (T,) with predicted tokens
        board_size: Size of the board (e.g., 8 for 8x8)

    Returns:
        List of PIL Image objects
    """
    # Define colors: FLOOR=0, WALL=1, START=2, END=3, PATH=4
    cmap = ListedColormap(['white', 'black', 'lime', 'red', 'gold'])

    images = []
    for layer_idx, frame in enumerate(output_frames):
        fig, ax = plt.subplots(figsize=(8, 8))
        board = frame.cpu().numpy().reshape(board_size, board_size)

        # Display board
        im = ax.imshow(board, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')

        # Add grid
        ax.set_xticks(np.arange(-.5, board_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, board_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title
        ax.set_title(f'Predictions - layer: {layer_idx}', fontsize=18, fontweight='bold', pad=20)

        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='white', ec='black', label='Floor'),
            plt.Rectangle((0, 0), 1, 1, fc='black', label='Wall'),
            plt.Rectangle((0, 0), 1, 1, fc='lime', label='Start'),
            plt.Rectangle((0, 0), 1, 1, fc='red', label='End'),
            plt.Rectangle((0, 0), 1, 1, fc='gold', label='Path'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
                  fontsize=12, frameon=True)

        add_watermark(fig, ax)
        plt.tight_layout()

        images.append(fig_to_pil_image(fig))

    return images

# ============================================================================
# Combined Processing Visualization (board + dual graph)
# ============================================================================

def generate_processing_frames(
    output_frames: List[torch.Tensor],
    x_frames: List[torch.Tensor],
    y_frames: List[torch.Tensor],
    model,
    board_size: int,
    top_k_edges: int = 500,
    layout_seed: int = 42,
    min_component_size: int = 10
) -> List[Image.Image]:
    """
    Build per-layer panels showing:
    1) Board prediction
    2) Dual fixed graphs lit by current activations (Dy co-activation in blue, E@Dx flow in red)
    """
    cmap_board = ListedColormap(['white', 'black', 'lime', 'red', 'gold'])

    # Fixed topologies
    topology_dx = get_parameter_topology(model, topology_type='e_dx')
    topology_dy = get_parameter_topology(model, topology_type='dy_coact')
    N = topology_dx.shape[0]

    edges_dx, _ = build_topology_graph(topology_dx, top_k_edges)
    edges_dy, _ = build_topology_graph(topology_dy, top_k_edges)

    all_edges = edges_dx + edges_dy
    connected_neurons, neuron_map, _ = extract_hub_subgraph(all_edges, N, min_component_size)

    if len(connected_neurons) == 0:
        connected_neurons = list(range(N))
        neuron_map = {i: i for i in connected_neurons}

    edges_dx_hub = []
    for (i, j) in edges_dx:
        if i in neuron_map and j in neuron_map:
            edges_dx_hub.append((neuron_map[i], neuron_map[j]))

    edges_dy_hub = []
    for (i, j) in edges_dy:
        if i in neuron_map and j in neuron_map:
            edges_dy_hub.append((neuron_map[i], neuron_map[j]))

    N_viz = len(connected_neurons)
    all_edges_hub = edges_dx_hub + edges_dy_hub
    pos = compute_graph_layout(all_edges_hub, N_viz, layout_seed)

    topology_dx_np = topology_dx.cpu().numpy()
    topology_dx_subset = topology_dx_np[np.ix_(connected_neurons, connected_neurons)]

    red_color = np.array([1.0, 0.164, 0.164])   # #FF2A2A
    blue_color = np.array([0.012, 0.376, 1.0])  # #0360FF
    gray_base = np.array([0.75, 0.75, 0.75])

    images = []
    for layer_idx, board_tokens in enumerate(output_frames):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # ------------------------------------------------------------------
        # Panel 1: Board prediction
        # ------------------------------------------------------------------
        board = board_tokens.cpu().numpy().reshape(board_size, board_size)
        axes[0].imshow(board, cmap=cmap_board, vmin=0, vmax=4, interpolation='nearest')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(f'Board prediction - layer {layer_idx}', fontsize=14, fontweight='bold')

        # ------------------------------------------------------------------
        # Panel 2: Dual fixed graphs lit by activations
        # ------------------------------------------------------------------
        x_full = x_frames[layer_idx].cpu().numpy()
        x_act = x_full[connected_neurons]

        if layer_idx == 0:
            y_prev_act = np.zeros_like(x_act)
        else:
            y_prev_full = y_frames[layer_idx - 1].cpu().numpy()
            y_prev_act = y_prev_full[connected_neurons]

        # Blue edges: co-activation of y_{l-1}
        edge_act_dy = compute_edge_activations_coactivation(edges_dy_hub, y_prev_act) if len(edges_dy_hub) > 0 else np.array([])
        # Red edges: flow y_{l-1} -> x_l via E@Dx (broadcast magnitude; no gating by x)
        edge_act_dx = compute_edge_activations_signal_flow(
            edges_dx_hub,
            source_activations=y_prev_act,
            topology_matrix=topology_dx_subset,
            target_activations=None
        ) if len(edges_dx_hub) > 0 else np.array([])

        edge_act_dy_norm = normalize_array(edge_act_dy) if edge_act_dy.size > 0 else np.array([])
        edge_act_dx_norm = normalize_array(edge_act_dx) if edge_act_dx.size > 0 else np.array([])

        y_norm = normalize_array(y_prev_act)
        x_norm = normalize_array(x_act)

        node_colors = compute_dual_network_node_colors(y_norm, x_norm, blue_color, red_color, gray_base)
        # node_size_range = (30, 140)
        node_size_range = (5, 40)
        node_sizes = [node_size_range[0] + max_val * (node_size_range[1] - node_size_range[0])
                      for max_val in np.maximum(y_norm, x_norm)]

        G = nx.Graph()
        G.add_nodes_from(range(N_viz))
        G.add_edges_from(all_edges_hub)

        # Red edges (Dx flow)
        edge_colors_dx, edge_widths_dx = compute_dual_network_edge_colors_and_widths(
            edge_act_dx_norm, red_color, gray_base, width_range=(0.4, 2.0), alpha=0.9
        )
        if len(edges_dx_hub) > 0:
            nx.draw_networkx_edges(
                G, pos, ax=axes[1],
                edgelist=edges_dx_hub,
                edge_color=edge_colors_dx,
                width=edge_widths_dx
            )

        # Blue edges (Dy co-activation)
        edge_colors_dy, edge_widths_dy = compute_dual_network_edge_colors_and_widths(
            edge_act_dy_norm, blue_color, gray_base, width_range=(0.4, 2.0), alpha=0.9
        )
        if len(edges_dy_hub) > 0:
            nx.draw_networkx_edges(
                G, pos, ax=axes[1],
                edgelist=edges_dy_hub,
                edge_color=edge_colors_dy,
                width=edge_widths_dy
            )

        nx.draw_networkx_nodes(
            G, pos, ax=axes[1],
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors='none'
        )
        axes[1].set_title(f'Fixed graphs lit by activations - layer {layer_idx}', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        legend_text = 'Blue: y_{l-1} co-activation (Dy)\n'
        legend_text += 'Red: y_{l-1} → x_l flow (E@Dx)\n'
        legend_text += f'Hub neurons: {N_viz}/{N}'

        axes[1].text(0.02, 0.02, legend_text, transform=axes[1].transAxes,
                     fontsize=10, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                     family='monospace')

        add_watermark(fig, axes[1])
        plt.tight_layout()

        images.append(fig_to_pil_image(fig))

    return images
