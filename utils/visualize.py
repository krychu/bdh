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
    print(f"Saved GIF to: {save_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration per frame: {duration}ms")

def combine_image_lists(
    image_lists: List[List[Image.Image]],
    spacing: int = 20
) -> List[Image.Image]:
    """
    Combine multiple lists of images horizontally into a single list.

    Args:
        image_lists: List of image lists to combine
        spacing: Pixels of white space between images (default: 20)

    Returns:
        List of combined PIL images
    """
    if not image_lists or not all(image_lists):
        raise ValueError("All image lists must be non-empty")

    # Handle frame count mismatch by repeating last frame
    max_frames = max(len(imgs) for imgs in image_lists)

    # Extend shorter lists by repeating last frame
    extended_lists = []
    for imgs in image_lists:
        extended = imgs.copy()
        while len(extended) < max_frames:
            extended.append(extended[-1].copy())
        extended_lists.append(extended)

    # Get dimensions
    widths = [imgs[0].size[0] for imgs in extended_lists]
    heights = [imgs[0].size[1] for imgs in extended_lists]
    combined_width = sum(widths) + spacing * (len(widths) - 1)
    combined_height = max(heights)

    print(f"Combining {len(image_lists)} image lists horizontally:")
    print(f"  Frame counts: {[len(imgs) for imgs in image_lists]}")
    print(f"  Individual sizes: {list(zip(widths, heights))}")
    print(f"  Combined size: {combined_width}×{combined_height}")
    print(f"  Total frames: {max_frames}")

    # Create combined frames
    combined_frames = []
    for frame_idx in range(max_frames):
        combined = Image.new('RGB', (combined_width, combined_height), 'white')

        x_offset = 0
        for img_list, width, height in zip(extended_lists, widths, heights):
            frame = img_list[frame_idx]
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            y_offset = (combined_height - height) // 2
            combined.paste(frame, (x_offset, y_offset))
            x_offset += width + spacing

        combined_frames.append(combined)

    return combined_frames

# ============================================================================
# Color Computation
# ============================================================================

def compute_dual_layer_edge_colors(
    structural_weights: np.ndarray,
    activations: np.ndarray,
    base_range: Tuple[float, float] = (0.7, 0.9),
    active_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.9,
    width_range: Tuple[float, float] = (0.3, 1.5)
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """
    Compute dual-layer edge colors and widths: gray (structure) → color (activation).

    Args:
        structural_weights: Normalized structural weights [0, 1]
        activations: Normalized activation values [0, 1]
        base_range: (min, max) gray intensity for structure
        active_color: RGB color for full activation
        alpha: Alpha channel value
        width_range: (min_width, max_width) for edge thickness

    Returns:
        Tuple of (colors, widths)
    """
    colors = []
    widths = []

    for struct_val, act_val in zip(structural_weights, activations):
        # Base: light gray from structure (always visible)
        base_intensity = base_range[0] + struct_val * (base_range[1] - base_range[0])

        # Red overlay from activation
        # As activation increases: R→1, G&B→0 (pure red)
        r = base_intensity + act_val * (1.0 - base_intensity)  # Increases toward 1
        g = base_intensity * (1 - act_val * 0.9)  # Dims toward 0
        b = base_intensity * (1 - act_val * 0.9)  # Dims toward 0

        colors.append((r, g, b, alpha))

        # Width: dual-layer (structure + activation)
        # Base width from structure, increased by activation
        base_width = width_range[0] + struct_val * (width_range[1] - width_range[0]) * 0.4
        width = base_width + act_val * (width_range[1] - base_width)
        widths.append(width)

    return colors, widths

def compute_dual_layer_node_colors(
    activations: np.ndarray,
    base_gray: float = 0.85,
    active_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> List[Tuple[float, float, float]]:
    """
    Compute dual-layer node colors: gray (inactive) → color (active).

    Args:
        activations: Normalized activation values [0, 1]
        base_gray: Base gray intensity
        active_color: RGB color for full activation

    Returns:
        List of RGB tuples
    """
    colors = []
    for act_val in activations:
        # Red overlay from activation
        r = base_gray + (1 - base_gray) * act_val  # Increase red
        g = base_gray * (1 - act_val * 0.8)   # Dim green
        b = base_gray * (1 - act_val * 0.8)   # Dim blue
        colors.append((r, g, b))

    return colors

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
        # Blend from gray to target color
        # r = gray_base[0] + act_val * (color[0] - gray_base[0])
        # g = gray_base[1] + act_val * (color[1] - gray_base[1])
        # b = gray_base[2] + act_val * (color[2] - gray_base[2])
        r = gray_base[0] + 10000.0 * act_val * (color[0] - gray_base[0])
        g = gray_base[1] + 10000.0 * act_val * (color[1] - gray_base[1])
        b = gray_base[2] + 10000.0 * act_val * (color[2] - gray_base[2])
        colors.append((r, g, b, alpha))

        # Width proportional to activation
        # width = width_range[0] + act_val * (width_range[1] - width_range[0])
        width = width_range[0] + act_val * min(10.0*width_range[1] - width_range[0], 2.0)
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

    # Get threshold for top-K edges
    flat_topology = topology_np.flatten()
    threshold = np.partition(flat_topology, -top_k_edges)[-top_k_edges]

    # Build edge list (undirected, upper triangle only)
    edge_list = []
    edge_weights = []

    for i in range(N):
        for j in range(i+1, N):
            weight = (topology_np[i, j] + topology_np[j, i]) / 2  # Symmetrize
            if weight >= threshold:
                edge_list.append((i, j))
                edge_weights.append(weight)

    return edge_list, np.array(edge_weights)

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

    if min_component_size > 1 and small_components:
        small_sizes = sorted([len(comp) for comp in small_components], reverse=True)
        print(f"  Filtering out {len(small_components)} small components (sizes: {small_sizes[:10]}{'...' if len(small_sizes) > 10 else ''})")
        print(f"  Keeping {len(large_components)} large components (min size: {min_component_size})")

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

def compute_edge_activations_synapse(
    edge_list: List[Tuple[int, int]],
    synapse_matrix: np.ndarray
) -> np.ndarray:
    """Compute edge activations using synapse mode (Hebbian co-activation)."""
    activations = []
    for i, j in edge_list:
        syn_weight = (synapse_matrix[i, j] + synapse_matrix[j, i]) / 2
        activations.append(syn_weight)
    return np.array(activations)

def compute_edge_activations_signal_flow(
    edge_list: List[Tuple[int, int]],
    y_activations: np.ndarray,
    topology_matrix: np.ndarray
) -> np.ndarray:
    """Compute edge activations using signal flow mode (y * weight)."""
    activations = []
    for i, j in edge_list:
        flow_i_to_j = abs(y_activations[i] * topology_matrix[i, j])
        flow_j_to_i = abs(y_activations[j] * topology_matrix[j, i])
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
# Single Network Graph Visualization
# ============================================================================

def generate_graph_frames(
    x_frames: List[torch.Tensor],
    synapse_frames: List[torch.Tensor],
    model,
    top_k_edges: int = 5000,
    layout_seed: int = 42,
    topology_type: str = 'e_dx',
    y_frames: Optional[List[torch.Tensor]] = None,
    visualization_mode: str = 'synapse',
    min_component_size: int = 1
) -> List[Image.Image]:
    """
    Generate PIL images with dual-layer color encoding (hub-only view).

    Args:
        x_frames: List of L tensors, each shape (N,) with x neuron activations
        synapse_frames: List of L tensors, each shape (N, N) with synapse values
        model: BDH model (to extract topology)
        top_k_edges: Number of strongest connections to include
        layout_seed: Random seed for reproducible layout
        topology_type: 'e_dx', 'dx_coact', or 'dy_coact'
        y_frames: List of L tensors with y neuron activations (required for signal_flow and dy_coact)
        visualization_mode: 'synapse' or 'signal_flow'
        min_component_size: Minimum connected component size to include (default: 1 = all)

    Returns:
        List of PIL Image objects
    """
    # Validation
    if visualization_mode not in ['synapse', 'signal_flow']:
        raise ValueError(f"visualization_mode must be 'synapse' or 'signal_flow', got '{visualization_mode}'")
    if visualization_mode == 'signal_flow' and y_frames is None:
        raise ValueError("y_frames is required for 'signal_flow' visualization mode")
    if topology_type == 'dy_coact' and y_frames is None:
        raise ValueError("y_frames is required for 'dy_coact' topology type")

    # Get topology and build graph
    topology_matrix = get_parameter_topology(model, topology_type=topology_type)
    N = topology_matrix.shape[0]

    topology_desc = {
        'e_dx': 'E @ Dx (communication)',
        'dx_coact': 'Dx.T @ Dx (co-activation)',
        'dy_coact': 'Dy.T @ Dy (attention decoder)'
    }[topology_type]

    mode_desc = {
        'synapse': 'Hebbian co-activation (x.T @ y)',
        'signal_flow': 'Signal flow (y * weight)'
    }[visualization_mode]

    print(f"Building hub graph with top {top_k_edges} connections from {N} neurons...")
    print(f"Topology: {topology_desc}")
    print(f"Visualization mode: {mode_desc}")
    print(f"Edge colors: Light gray (structure) → Red (activation)")
    print(f"Node colors: Light gray (inactive) → Red (activation)")

    # Build graph structure
    edge_list, edge_weights_structural = build_topology_graph(topology_matrix, top_k_edges)
    connected_neurons, neuron_map, edge_list_hub = extract_hub_subgraph(edge_list, N, min_component_size)

    N_viz = len(connected_neurons)
    print(f"Graph built: {N} nodes, {len(edge_list)} edges")
    print(f"Connected neurons: {N_viz} ({N_viz/N*100:.1f}%)")
    print(f"Using {N_viz} connected neurons (hub view)")

    # Compute layout
    print(f"Computing force-directed layout...")
    pos = compute_graph_layout(edge_list_hub, N_viz, layout_seed)
    print(f"Layout computed. Creating animation...")

    # Build graph for drawing (with ALL hub nodes, not just those with edges)
    G_hub = nx.Graph()
    G_hub.add_nodes_from(range(N_viz))
    G_hub.add_edges_from(edge_list_hub)

    # Normalize structural weights
    edge_weights_struct_norm = normalize_array(edge_weights_structural, vmin=0, vmax=edge_weights_structural.max())

    # Get topology subset for signal flow mode
    topology_np = topology_matrix.cpu().numpy()
    topology_subset = topology_np[np.ix_(connected_neurons, connected_neurons)]

    # Generate frames
    images = []
    for layer_idx, (x_frame, synapse_frame) in enumerate(zip(x_frames, synapse_frames)):
        fig, ax = plt.subplots(figsize=(12, 12))

        # Extract activations for hub neurons
        if topology_type == 'dy_coact':
            activations_full = y_frames[layer_idx].cpu().numpy()
        else:
            activations_full = x_frame.cpu().numpy()

        synapse_np_full = synapse_frame.cpu().numpy()
        activations = activations_full[connected_neurons]
        synapse_np = synapse_np_full[np.ix_(connected_neurons, connected_neurons)]

        # Compute edge activations based on mode
        if topology_type == 'dy_coact':
            edge_activations = compute_edge_activations_coactivation(edge_list_hub, activations)
        elif visualization_mode == 'synapse':
            edge_activations = compute_edge_activations_synapse(edge_list_hub, synapse_np)
        elif visualization_mode == 'signal_flow':
            y_full = y_frames[layer_idx].cpu().numpy()
            y_activations = y_full[connected_neurons]
            edge_activations = compute_edge_activations_signal_flow(edge_list_hub, y_activations, topology_subset)

        # Normalize edge activations
        edge_activations_norm = normalize_array(edge_activations, vmin=0, vmax=edge_activations.max())

        # Compute edge colors and widths
        edge_colors, edge_widths = compute_dual_layer_edge_colors(
            edge_weights_struct_norm,
            edge_activations_norm,
            base_range=(0.7, 0.9),
            active_color=(1.0, 0.0, 0.0),
            alpha=0.9,
            width_range=(0.3, 1.5)
        )

        # Draw edges
        nx.draw_networkx_edges(
            G_hub, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths
        )

        # Normalize node activations and compute colors
        activations_norm = normalize_array(activations, vmin=0, vmax=activations.max())
        node_colors = compute_dual_layer_node_colors(
            activations_norm,
            base_gray=0.85,
            active_color=(1.0, 0.0, 0.0)
        )

        # Compute node sizes based on activation
        node_size_range = (20, 100)  # min to max size
        node_sizes = [node_size_range[0] + act * (node_size_range[1] - node_size_range[0])
                      for act in activations_norm]

        # Draw nodes
        nx.draw_networkx_nodes(
            G_hub, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors='none'
        )

        # Title
        layer_display = str(layer_idx)

        title = f'{topology_desc}'
        if topology_type != 'dy_coact':
            if visualization_mode == 'signal_flow':
                title += ' - signal flow'
            else:
                title += ' - synapse'
        title += f' - layer: {layer_display}'

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

        # Legend and stats
        active_neurons = (activations > 0.1 * activations.max()).sum() if activations.max() > 0 else 0
        active_synapses = (edge_activations > 0.1 * edge_activations.max()).sum() if edge_activations.max() > 0 else 0

        legend_text = 'Color: Inactive (gray) → Active (red)\n'
        legend_text += f'Hub neurons: {N_viz}/{N}\n'
        legend_text += f'Active neurons: {active_neurons}/{N_viz}\n'
        legend_text += f'Active edges: {active_synapses}/{len(edge_list)}'

        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                family='monospace')

        add_watermark(fig, ax)
        plt.tight_layout()

        images.append(fig_to_pil_image(fig))
        print(f"  Frame {layer_idx+1}/{len(x_frames)} completed")

    return images

# ============================================================================
# Dual-Network Interleaved Visualization
# ============================================================================

def generate_interleaved_graph_frames(
    x_frames: List[torch.Tensor],
    y_frames: List[torch.Tensor],
    synapse_frames: List[torch.Tensor],
    model,
    top_k_edges: int = 5000,
    layout_seed: int = 42,
    min_component_size: int = 1
) -> List[Image.Image]:
    """
    Generate interleaved dual-network visualization (hub-only view).

    Each frame shows the causal computation for a layer:
    - Blue nodes: y_{L-1} (previous layer output, source of signal)
    - Red nodes: x_L (current layer state, destination of signal)
    - Blue edges: y_{L-1} co-activation patterns (Dy topology)
    - Red edges: y_{L-1} -> x_L signal flow (Dx topology)

    Args:
        x_frames: List of L tensors, each shape (N,) with x neuron activations
        y_frames: List of L tensors, each shape (N,) with y neuron activations
        synapse_frames: List of L tensors, each shape (N, N) with synapse values
        model: BDH model (to extract topologies)
        top_k_edges: Number of strongest connections from each topology
        layout_seed: Random seed for reproducible layout
        min_component_size: Minimum connected component size to include (default: 1 = all)

    Returns:
        List of PIL images (one per layer)
    """
    print(f"Building interleaved dual-network visualization...")
    print(f"Blue: y_{{L-1}} (previous output) with Dy co-activation")
    print(f"Red: x_L (current state) with Dx signal flow from y_{{L-1}}")

    # Get both topologies
    topology_dy = get_parameter_topology(model, topology_type='dy_coact')
    topology_dx = get_parameter_topology(model, topology_type='e_dx')
    N = topology_dy.shape[0]

    # Build graphs from both topologies
    edges_dy, weights_dy = build_topology_graph(topology_dy, top_k_edges)
    edges_dx, weights_dx = build_topology_graph(topology_dx, top_k_edges)

    print(f"Master graph: {N} nodes, {len(edges_dy)} Dy edges, {len(edges_dx)} Dx edges")

    # Build unified hub subgraph containing edges from both topologies
    all_edges = edges_dy + edges_dx
    connected_neurons, neuron_map, _ = extract_hub_subgraph(all_edges, N, min_component_size)

    # Remap both edge lists (filtering out edges not in large components)
    edges_dy_hub = []
    for i, j in edges_dy:
        if i in neuron_map and j in neuron_map:
            edges_dy_hub.append((neuron_map[i], neuron_map[j]))

    edges_dx_hub = []
    for i, j in edges_dx:
        if i in neuron_map and j in neuron_map:
            edges_dx_hub.append((neuron_map[i], neuron_map[j]))

    N_viz = len(connected_neurons)
    print(f"Connected neurons: {N_viz} ({N_viz/N*100:.1f}%)")
    print(f"Using {N_viz} connected neurons (hub view)")

    # Compute unified layout ONCE
    all_edges_hub = edges_dy_hub + edges_dx_hub
    print(f"Computing unified layout for {N_viz} nodes...")
    pos = compute_graph_layout(all_edges_hub, N_viz, layout_seed)
    print(f"Layout computed. Generating {len(x_frames)} dual-network frames...")

    # Color definitions
    red_color = np.array([1.0, 0.164, 0.164])  # #FF2A2A
    blue_color = np.array([0.012, 0.376, 1.0])  # #0360FF
    gray_base = np.array([0.75, 0.75, 0.75])

    # Extract topology subset for Dx (needed for signal flow computation)
    topology_dx_np = topology_dx.cpu().numpy()
    topology_dx_subset = topology_dx_np[np.ix_(connected_neurons, connected_neurons)]

    # Generate frames
    images = []
    for layer_idx in range(len(x_frames)):
        # Extract current layer activations
        x_full = x_frames[layer_idx].cpu().numpy()
        x_act = x_full[connected_neurons]

        # Get y from PREVIOUS layer (or zeros for layer 0)
        if layer_idx == 0:
            # Layer 0: No previous y, show only x (input-driven)
            y_prev_act = np.zeros_like(x_act)
        else:
            y_prev_full = y_frames[layer_idx - 1].cpu().numpy()
            y_prev_act = y_prev_full[connected_neurons]

        # Blue (Dy) edges: Co-activation patterns of y_{l-1}
        # Shows which previous-layer neurons activated together
        edge_act_dy = compute_edge_activations_coactivation(edges_dy_hub, y_prev_act)

        # Red (Dx) edges: Signal flow from y_{l-1} to x_l
        # Shows how previous layer output propagates through E@Dx to produce current x
        if layer_idx == 0:
            # Layer 0: x is driven by input embeddings, not previous y
            edge_act_dx = np.zeros(len(edges_dx_hub))
        else:
            edge_act_dx = compute_edge_activations_signal_flow(
                edges_dx_hub,
                y_prev_act,
                topology_dx_subset
            )

        # Normalize
        edge_act_dy_norm = normalize_array(edge_act_dy) if len(edge_act_dy) > 0 else np.array([])
        edge_act_dx_norm = normalize_array(edge_act_dx) if len(edge_act_dx) > 0 else np.array([])
        y_prev_act_norm = normalize_array(y_prev_act)
        x_act_norm = normalize_array(x_act)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))

        # Build graph for drawing
        G_master = nx.Graph()
        G_master.add_nodes_from(range(N_viz))
        G_master.add_edges_from(edges_dy_hub + edges_dx_hub)

        # Draw Dx edges (red)
        edge_colors_dx, edge_widths_dx = compute_dual_network_edge_colors_and_widths(
            edge_act_dx_norm, red_color, gray_base, width_range=(0.3, 1.5), alpha=0.8
        )

        if len(edges_dx_hub) > 0:
            nx.draw_networkx_edges(
                G_master, pos, ax=ax,
                edgelist=edges_dx_hub,
                edge_color=edge_colors_dx,
                width=edge_widths_dx
            )

        # Draw Dy edges (blue) on top
        edge_colors_dy, edge_widths_dy = compute_dual_network_edge_colors_and_widths(
            edge_act_dy_norm, blue_color, gray_base, width_range=(0.3, 1.5), alpha=0.8
        )

        if len(edges_dy_hub) > 0:
            nx.draw_networkx_edges(
                G_master, pos, ax=ax,
                edgelist=edges_dy_hub,
                edge_color=edge_colors_dy,
                width=edge_widths_dy
            )

        # Compute node colors (blend blue and red)
        # Blue represents y_{l-1}, Red represents x_l
        node_colors = compute_dual_network_node_colors(y_prev_act_norm, x_act_norm, blue_color, red_color, gray_base)

        # Compute node sizes based on max activation (x or y_prev)
        node_size_range = (20, 100)  # min to max size
        max_activations = np.maximum(y_prev_act_norm, x_act_norm)
        node_sizes = [node_size_range[0] + act * (node_size_range[1] - node_size_range[0])
                      for act in max_activations]

        nx.draw_networkx_nodes(
            G_master, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors='none'
        )

        # Title
        title = f'Layer: {layer_idx} - Causal Flow (y_{{prev}} → x)'
        ax.set_title(title, fontsize=16, fontweight='bold', color='purple')
        ax.axis('off')

        # Legend
        legend_text = 'Blue: y_{L-1} (previous output)\n'
        legend_text += 'Red: x_L (current state)\n'
        legend_text += 'Blue edges: y_{L-1} co-activation\n'
        legend_text += 'Red edges: y_{L-1} → x_L signal flow'

        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                family='monospace')

        add_watermark(fig, ax)
        plt.tight_layout()

        images.append(fig_to_pil_image(fig))
        print(f"  Layer {layer_idx+1}/{len(x_frames)} completed (1 frame)")

    return images
