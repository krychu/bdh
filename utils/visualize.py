import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import torch
from typing import List, Optional
import networkx as nx
from utils.build_boardpath_dataset import FLOOR, WALL, START, END, PATH

# ============================================================================
# Helper Functions for Frame Management
# ============================================================================

def add_watermark(fig, ax):
    """
    Add GitHub URL watermark to bottom-right corner of the plot.

    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
    """
    # Add text in bottom-right corner of the axes
    ax.text(0.98, 0.02, 'https://github.com/krychu/bdh',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='bottom',
            horizontalalignment='right',
            color='black',
            alpha=1.0,
            family='monospace')

def save_gif(images: List[Image.Image], save_path: str, duration: int = 500):
    """
    Save a list of PIL images as an animated GIF.

    Args:
        images: List of PIL Image objects
        save_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
    """
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
        image_lists: List of image lists to combine (e.g., [[board_imgs], [hub_imgs], [full_imgs]])
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

    # Get dimensions of each image list
    widths = [imgs[0].size[0] for imgs in extended_lists]
    heights = [imgs[0].size[1] for imgs in extended_lists]

    # Calculate combined dimensions
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
        # Create white background
        combined = Image.new('RGB', (combined_width, combined_height), 'white')

        # Paste each image horizontally
        x_offset = 0
        for img_list, width, height in zip(extended_lists, widths, heights):
            frame = img_list[frame_idx]

            # Convert to RGB if needed
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            # Center vertically
            y_offset = (combined_height - height) // 2
            combined.paste(frame, (x_offset, y_offset))

            x_offset += width + spacing

        combined_frames.append(combined)

    return combined_frames

# ============================================================================
# Topology Extraction
# ============================================================================

def get_parameter_topology(model, topology_type: str = 'e_dx') -> torch.Tensor:
    """
    Extract N×N topology from model parameters.

    Args:
        model: BDH model instance
        topology_type: 'e_dx' (communication), 'dx_coact' (co-activation), or 'dy_coact' (attention decoder)

    Returns:
        topology: (N, N) tensor where topology[i,j] is connection strength
    """
    # E: (N, D)
    # Dx: (H, D, N//H)
    # Dy: (H, D, N//H)
    H, D, Nh = model.Dx.shape
    N = H * Nh

    # Reshape Dx from (H, D, N//H) to (D, N)
    Dx_reshaped = model.Dx.permute(1, 0, 2).reshape(D, N)
    # Reshape Dy from (H, D, N//H) to (D, N)
    Dy_reshaped = model.Dy.permute(1, 0, 2).reshape(D, N)

    if topology_type == 'e_dx':
        # E @ Dx: Communication structure
        # Shows how neuron i's output affects neuron j via embedding updates
        topology = model.E @ Dx_reshaped
    elif topology_type == 'dx_coact':
        # Dx.T @ Dx: Co-activation structure
        # Shows which neurons respond to similar embedding patterns
        topology = Dx_reshaped.T @ Dx_reshaped
    elif topology_type == 'dy_coact':
        # Dy.T @ Dy: Attention decoder co-activation structure
        # Shows which neurons respond to similar attention features
        topology = Dy_reshaped.T @ Dy_reshaped
    else:
        raise ValueError(f"Unknown topology_type: {topology_type}")

    # Return absolute values for undirected graph interpretation
    return topology.abs().detach()


def generate_board_frames(
    output_frames: List[torch.Tensor],
    board_size: int,
    interpolate_frames: int = 1
) -> List[Image.Image]:
    """
    Generate PIL images of board predictions through layers.

    Args:
        output_frames: List of tensors, each shape (T,) with predicted tokens
        board_size: Size of the board (e.g., 8 for 8x8)
        interpolate_frames: Number of frames between layers (1=no interpolation, simply repeat each frame)

    Returns:
        List of PIL Image objects
    """
    # Interpolate frames by repeating each frame
    if interpolate_frames > 1:
        output_frames_interp = []
        for i in range(len(output_frames) - 1):
            # Repeat current frame interpolate_frames times
            for _ in range(interpolate_frames):
                output_frames_interp.append(output_frames[i])
        # Add final frame
        output_frames_interp.append(output_frames[-1])
        output_frames = output_frames_interp
    # Define colors for each cell type
    # FLOOR=0: white, WALL=1: black, START=2: green, END=3: red, PATH=4: yellow
    cmap = ListedColormap(['white', 'black', 'lime', 'red', 'gold'])

    images = []
    for layer_idx, frame in enumerate(output_frames):
        fig, ax = plt.subplots(figsize=(8, 8))
        board = frame.cpu().numpy().reshape(board_size, board_size)

        # Display the board as a heatmap
        im = ax.imshow(board, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')

        # Add grid lines
        ax.set_xticks(np.arange(-.5, board_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, board_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)

        # Remove major ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title (match graph visualization format)
        ax.set_title(f'Predictions - layer: {layer_idx}', fontsize=18, fontweight='bold', pad=20)

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='white', ec='black', label='Floor'),
            plt.Rectangle((0, 0), 1, 1, fc='black', label='Wall'),
            plt.Rectangle((0, 0), 1, 1, fc='lime', label='Start'),
            plt.Rectangle((0, 0), 1, 1, fc='red', label='End'),
            plt.Rectangle((0, 0), 1, 1, fc='gold', label='Path'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
                  fontsize=12, frameon=True)

        # Add watermark
        add_watermark(fig, ax)

        plt.tight_layout()

        # Convert to PIL Image using buffer
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).copy()
        images.append(image)
        plt.close(fig)
        buf.close()

    return images


def generate_graph_frames(
    x_frames: List[torch.Tensor],
    synapse_frames: List[torch.Tensor],
    model,
    top_k_edges: int = 5000,
    layout_seed: int = 42,
    topology_type: str = 'e_dx',
    hub_only: bool = False,
    interpolate_frames: int = 1,
    y_frames: Optional[List[torch.Tensor]] = None,
    visualization_mode: str = 'synapse'
) -> List[Image.Image]:
    """
    Generate PIL images with dual-layer color encoding:

    - Graph structure (layout): Topology matrix (E @ Dx, Dx.T @ Dx, or Dy.T @ Dy)
    - Edge base color (gray): Structural weight (always visible if > threshold)
    - Edge overlay (red): Either synapse activation OR signal flow OR co-activation (depending on mode/topology)
    - Node base color: Light gray (always visible)
    - Node overlay (red): Neuron activation per layer

    Args:
        x_frames: List of L tensors, each shape (N,) with x neuron activations per layer (used for node coloring in e_dx/dx_coact)
        synapse_frames: List of L tensors, each shape (N, N) with synapse values per layer
        model: BDH model (to extract topology)
        top_k_edges: Number of strongest connections to include in graph (default: 5000)
        layout_seed: Random seed for reproducible layout (default: 42)
        topology_type: 'e_dx' (communication), 'dx_coact' (co-activation), or 'dy_coact' (attention decoder)
        hub_only: If True, show only connected neurons (zoomed view)
        interpolate_frames: Number of interpolated frames between each layer (1=no interpolation, 3=2 extra frames)
        y_frames: List of L tensors, each shape (N,) with y neuron activations per layer (required for 'signal_flow' mode and dy_coact topology)
        visualization_mode: 'synapse' (Hebbian co-activation) or 'signal_flow' (causal signal propagation) - ignored for dy_coact

    Note: For dy_coact topology, y_frames are used for node coloring (showing y activations),
          and edges show co-activation of y neurons (y[i] * y[j])
    """
    import io
    from matplotlib.colors import Normalize

    # Validate visualization mode
    if visualization_mode not in ['synapse', 'signal_flow']:
        raise ValueError(f"visualization_mode must be 'synapse' or 'signal_flow', got '{visualization_mode}'")

    if visualization_mode == 'signal_flow' and y_frames is None:
        raise ValueError("y_frames is required for 'signal_flow' visualization mode")

    if topology_type == 'dy_coact' and y_frames is None:
        raise ValueError("y_frames is required for 'dy_coact' topology type")

    # Get parameter topology
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

    print(f"Building graph with top {top_k_edges} connections from {N} neurons...")
    print(f"Topology: {topology_desc}")
    print(f"Visualization mode: {mode_desc}")
    print(f"Hub-only mode: {hub_only}")
    print(f"Edge colors: Light gray (structure) → Red (activation)")
    print(f"Node colors: Light gray (inactive) → Red (activation)")

    # Convert to numpy for graph building
    topology_np = topology_matrix.cpu().numpy()

    # Get top-K edges from structural topology (E @ Dx)
    flat_topology = topology_np.flatten()
    threshold = np.partition(flat_topology, -top_k_edges)[-top_k_edges]

    # Build graph structure (fixed across all layers)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    edge_list = []
    edge_weights_structural = []  # Store E @ Dx weights for each edge

    for i in range(N):
        for j in range(i+1, N):  # Undirected, so only upper triangle
            weight = (topology_np[i, j] + topology_np[j, i]) / 2  # Symmetrize
            if weight >= threshold:
                G.add_edge(i, j)
                edge_list.append((i, j))
                edge_weights_structural.append(weight)

    edge_weights_structural = np.array(edge_weights_structural)

    edge_count = len(edge_list)

    # Identify connected neurons for hub-only mode
    connected_neurons = set()
    for i, j in edge_list:
        connected_neurons.add(i)
        connected_neurons.add(j)
    connected_neurons = sorted(connected_neurons)

    print(f"Graph built: {N} nodes, {edge_count} edges")
    print(f"Connected neurons: {len(connected_neurons)} ({len(connected_neurons)/N*100:.1f}%)")

    # Handle hub-only mode
    if hub_only:
        # Create neuron index mapping (old → new)
        neuron_map = {old_idx: new_idx for new_idx, old_idx in enumerate(connected_neurons)}

        # Build subgraph with only connected neurons
        G_hub = nx.Graph()
        G_hub.add_nodes_from(range(len(connected_neurons)))

        edge_list_hub = []
        for i, j in edge_list:
            G_hub.add_edge(neuron_map[i], neuron_map[j])
            edge_list_hub.append((neuron_map[i], neuron_map[j]))

        G = G_hub
        edge_list = edge_list_hub
        N_viz = len(connected_neurons)
        print(f"Hub-only mode: Using {N_viz} connected neurons")
    else:
        N_viz = N
        neuron_map = None

    print(f"Computing force-directed layout...")

    # Compute layout once (this is slow for large graphs)
    pos = nx.spring_layout(G, k=1/np.sqrt(N_viz), iterations=50, seed=layout_seed)

    print(f"Layout computed. Creating animation...")

    # Interpolate frames if requested
    if interpolate_frames > 1:
        print(f"Interpolating {interpolate_frames}x frames between layers...")
        x_frames_interp = []
        synapse_frames_interp = []
        y_frames_interp = [] if y_frames is not None else None

        for i in range(len(x_frames) - 1):
            # Add current frame
            x_frames_interp.append(x_frames[i])
            synapse_frames_interp.append(synapse_frames[i])
            if y_frames is not None:
                y_frames_interp.append(y_frames[i])

            # Add interpolated frames
            for j in range(1, interpolate_frames):
                alpha = j / interpolate_frames  # 0 to 1
                # Linear interpolation
                x_interp = (1 - alpha) * x_frames[i] + alpha * x_frames[i + 1]
                synapse_interp = (1 - alpha) * synapse_frames[i] + alpha * synapse_frames[i + 1]
                x_frames_interp.append(x_interp)
                synapse_frames_interp.append(synapse_interp)

                if y_frames is not None:
                    y_interp = (1 - alpha) * y_frames[i] + alpha * y_frames[i + 1]
                    y_frames_interp.append(y_interp)

        # Add final frame
        x_frames_interp.append(x_frames[-1])
        synapse_frames_interp.append(synapse_frames[-1])
        if y_frames is not None:
            y_frames_interp.append(y_frames[-1])

        x_frames = x_frames_interp
        synapse_frames = synapse_frames_interp
        if y_frames is not None:
            y_frames = y_frames_interp
        print(f"Total frames after interpolation: {len(x_frames)}")

    # Normalize structural weights for gray base color (0 to 1)
    struct_norm = Normalize(vmin=0, vmax=edge_weights_structural.max())
    edge_weights_struct_norm = struct_norm(edge_weights_structural)

    images = []
    for layer_idx, (x_frame, synapse_frame) in enumerate(zip(x_frames, synapse_frames)):
        fig, ax = plt.subplots(figsize=(12, 12))

        # Node activations (full array)
        # For dy_coact, use y activations; otherwise use x activations
        if topology_type == 'dy_coact':
            activations_full = y_frames[layer_idx].cpu().numpy()
        else:
            activations_full = x_frame.cpu().numpy()

        # Edge synapse strengths for this layer (full array)
        synapse_np_full = synapse_frame.cpu().numpy()

        # Handle hub-only mode: extract only connected neurons
        if hub_only:
            activations = activations_full[connected_neurons]
            # Create submatrix for synapse
            synapse_np = synapse_np_full[np.ix_(connected_neurons, connected_neurons)]
        else:
            activations = activations_full
            synapse_np = synapse_np_full

        # Compute edge activations based on topology type and visualization mode
        if topology_type == 'dy_coact':
            # For Dy graph: visualize co-activation of output neurons (y)
            # When both neurons are active, the edge lights up
            # This shows which neurons are being activated together by the attention summary
            edge_activations = []
            for i, j in edge_list:
                # Co-activation strength: product of activations
                co_activation = activations[i] * activations[j]
                edge_activations.append(co_activation)
            edge_activations = np.array(edge_activations)

        elif visualization_mode == 'synapse':
            # Synapse mode: Use x.T @ y (Hebbian co-activation)
            edge_activations = []
            for i, j in edge_list:
                syn_weight = (synapse_np[i, j] + synapse_np[j, i]) / 2
                edge_activations.append(syn_weight)
            edge_activations = np.array(edge_activations)

        elif visualization_mode == 'signal_flow':
            # Signal flow mode: Use y[i] * (E @ Dx)[i,j] (causal signal propagation)
            # Get y activations for this layer
            y_full = y_frames[layer_idx].cpu().numpy()

            # Handle hub-only mode
            if hub_only:
                y_activations = y_full[connected_neurons]
                topology_subset = topology_np[np.ix_(connected_neurons, connected_neurons)]
            else:
                y_activations = y_full
                topology_subset = topology_np

            edge_activations = []
            for i, j in edge_list:
                # Signal flow from i to j: y[i] * weight[i,j]
                # Signal flow from j to i: y[j] * weight[j,i]
                # Take average for undirected visualization
                flow_i_to_j = abs(y_activations[i] * topology_subset[i, j])
                flow_j_to_i = abs(y_activations[j] * topology_subset[j, i])
                edge_activations.append((flow_i_to_j + flow_j_to_i) / 2)
            edge_activations = np.array(edge_activations)

        # Normalize activations (0 to 1)
        if edge_activations.max() > 0:
            act_norm = Normalize(vmin=0, vmax=edge_activations.max())
            edge_activations_norm = act_norm(edge_activations)
        else:
            edge_activations_norm = np.zeros_like(edge_activations)

        # Dual-layer edge coloring: Light gray (structure) → Bright red (activation)
        edge_colors = []
        for struct_val, act_val in zip(edge_weights_struct_norm, edge_activations_norm):
            # Base: light gray from structure (always visible)
            base_intensity = 0.7 + struct_val * 0.2  # Range: 0.7 to 0.9 (light gray)

            # Red overlay from activation
            # As activation increases: R→1, G&B→0 (pure red)
            r = base_intensity + act_val * (1.0 - base_intensity)  # Increases toward 1
            g = base_intensity * (1 - act_val * 0.9)  # Dims toward 0
            b = base_intensity * (1 - act_val * 0.9)  # Dims toward 0

            edge_colors.append((r, g, b, 0.9))  # RGBA

        # Draw edges with dual-layer colors (constant width)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=0.5,  # Constant width
        )

        # Normalize neuron activations (0 to 1)
        if activations.max() > 0:
            node_norm = Normalize(vmin=0, vmax=activations.max())
            activations_norm = node_norm(activations)
        else:
            activations_norm = np.zeros_like(activations)

        # Dual-layer node coloring: Light gray (base) + Red (activation)
        node_colors = []
        for act_val in activations_norm:
            base = 0.85  # Light gray

            # Red overlay from activation
            r = base + (1 - base) * act_val  # Increase red
            g = base * (1 - act_val * 0.8)   # Dim green
            b = base * (1 - act_val * 0.8)   # Dim blue

            node_colors.append((r, g, b))

        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=20,
            edgecolors='none'
        )

        # Show fractional layer index if interpolated
        if interpolate_frames > 1:
            layer_display = f"{layer_idx / interpolate_frames:.2f}"
        else:
            layer_display = str(layer_idx)

        # Build title: topology - mode - layer: X
        title = f'{topology_desc}'
        if topology_type != 'dy_coact':
            # For Dx graphs, show visualization mode
            if visualization_mode == 'signal_flow':
                title += ' - signal flow'
            else:
                title += ' - synapse'
        if hub_only:
            title += ' (hub only)'
        title += f' - layer: {layer_display}'

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add combined legend and stats (bottom left)
        active_neurons = (activations > 0.1 * activations.max()).sum() if activations.max() > 0 else 0
        active_synapses = (edge_activations > 0.1 * edge_activations.max()).sum() if edge_activations.max() > 0 else 0

        legend_text = 'Color: Inactive (gray) → Active (red)\n'

        if hub_only:
            legend_text += f'Hub neurons: {N_viz}/{N}\n'
            legend_text += f'Active neurons: {active_neurons}/{N_viz}\n'
        else:
            legend_text += f'Active neurons: {active_neurons}/{N_viz}\n'
        legend_text += f'Active edges: {active_synapses}/{edge_count}'

        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                family='monospace')

        # Add watermark
        add_watermark(fig, ax)

        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).copy()
        images.append(image)
        plt.close(fig)
        buf.close()

        print(f"  Frame {layer_idx+1}/{len(x_frames)} completed")

    return images


def generate_interleaved_graph_frames(
    x_frames: List[torch.Tensor],
    y_frames: List[torch.Tensor],
    synapse_frames: List[torch.Tensor],
    model,
    top_k_edges: int = 5000,
    layout_seed: int = 42,
    hub_only: bool = False,
    interpolate_frames: int = 1
) -> List[Image.Image]:
    """
    Generate interleaved visualization showing two-stage computation per layer:
    - Stage 1 (Blue/Dy): Attention decoding - which neurons activated by attention?
    - Stage 2 (Red/Dx): State propagation - how do those activations propagate?

    Creates 2 frames per layer showing the sequential computational flow.

    Args:
        x_frames: List of L tensors, each shape (N,) with x neuron activations
        y_frames: List of L tensors, each shape (N,) with y neuron activations
        synapse_frames: List of L tensors, each shape (N, N) with synapse values
        model: BDH model (to extract topologies)
        top_k_edges: Number of strongest connections to include from each topology
        layout_seed: Random seed for reproducible layout
        hub_only: If True, show only connected neurons (zoomed view)
        interpolate_frames: Number of interpolated frames between each stage (1=no interpolation)

    Returns:
        List of PIL images (2 * L frames total)
    """
    import io
    from matplotlib.colors import Normalize

    print(f"Building interleaved dual-network visualization...")
    print(f"Stage 1 (Blue): Dy - Attention Decoding")
    print(f"Stage 2 (Red): Dx - State Propagation")

    # Get both topologies
    topology_dy = get_parameter_topology(model, topology_type='dy_coact')
    topology_dx = get_parameter_topology(model, topology_type='e_dx')

    N = topology_dy.shape[0]

    # Convert to numpy
    topology_dy_np = topology_dy.cpu().numpy()
    topology_dx_np = topology_dx.cpu().numpy()

    # Get top-K edges from each topology
    flat_dy = topology_dy_np.flatten()
    flat_dx = topology_dx_np.flatten()

    threshold_dy = np.partition(flat_dy, -top_k_edges)[-top_k_edges]
    threshold_dx = np.partition(flat_dx, -top_k_edges)[-top_k_edges]

    # Build unified master graph containing edges from both topologies
    G_master = nx.Graph()
    G_master.add_nodes_from(range(N))

    edges_dy = []
    weights_dy = []
    edges_dx = []
    weights_dx = []

    # Collect Dy edges
    for i in range(N):
        for j in range(i+1, N):
            weight = (topology_dy_np[i, j] + topology_dy_np[j, i]) / 2
            if weight >= threshold_dy:
                edges_dy.append((i, j))
                weights_dy.append(weight)
                G_master.add_edge(i, j)

    # Collect Dx edges
    for i in range(N):
        for j in range(i+1, N):
            weight = (topology_dx_np[i, j] + topology_dx_np[j, i]) / 2
            if weight >= threshold_dx:
                edges_dx.append((i, j))
                weights_dx.append(weight)
                G_master.add_edge(i, j)

    weights_dy = np.array(weights_dy)
    weights_dx = np.array(weights_dx)

    print(f"Master graph: {N} nodes, {len(edges_dy)} Dy edges, {len(edges_dx)} Dx edges")

    # Identify connected neurons (for hub-only mode)
    connected_neurons = set()
    for i, j in list(edges_dy) + list(edges_dx):
        connected_neurons.add(i)
        connected_neurons.add(j)
    connected_neurons = sorted(connected_neurons)

    print(f"Connected neurons: {len(connected_neurons)} ({len(connected_neurons)/N*100:.1f}%)")

    # Handle hub-only mode
    if hub_only:
        neuron_map = {old_idx: new_idx for new_idx, old_idx in enumerate(connected_neurons)}

        # Build hub subgraph
        G_hub = nx.Graph()
        G_hub.add_nodes_from(range(len(connected_neurons)))

        edges_dy_hub = []
        edges_dx_hub = []

        for i, j in edges_dy:
            if i in connected_neurons and j in connected_neurons:
                G_hub.add_edge(neuron_map[i], neuron_map[j])
                edges_dy_hub.append((neuron_map[i], neuron_map[j]))

        for i, j in edges_dx:
            if i in connected_neurons and j in connected_neurons:
                G_hub.add_edge(neuron_map[i], neuron_map[j])
                edges_dx_hub.append((neuron_map[i], neuron_map[j]))

        G_master = G_hub
        edges_dy = edges_dy_hub
        edges_dx = edges_dx_hub
        N_viz = len(connected_neurons)
        print(f"Hub-only mode: Using {N_viz} connected neurons")
    else:
        N_viz = N
        neuron_map = None

    # Compute unified layout ONCE (critical for stability)
    print(f"Computing unified layout for {N_viz} nodes...")
    pos = nx.spring_layout(G_master, k=1/np.sqrt(N_viz), iterations=50, seed=layout_seed)
    print(f"Layout computed. Generating {len(x_frames)} dual-network frames...")

    # No interpolation for now - can add later if needed
    images = []

    for layer_idx in range(len(x_frames)):
        # Get activations for this layer
        x_full = x_frames[layer_idx].cpu().numpy()
        y_full = y_frames[layer_idx].cpu().numpy()
        synapse_full = synapse_frames[layer_idx].cpu().numpy()

        # Handle hub-only mode
        if hub_only:
            x_act = x_full[connected_neurons]
            y_act = y_full[connected_neurons]
            synapse_np = synapse_full[np.ix_(connected_neurons, connected_neurons)]
        else:
            x_act = x_full
            y_act = y_full
            synapse_np = synapse_full

        # Compute edge activations for both networks
        edge_act_dy = []
        for i, j in edges_dy:
            co_activation = y_act[i] * y_act[j]
            edge_act_dy.append(co_activation)
        edge_act_dy = np.array(edge_act_dy) if len(edge_act_dy) > 0 else np.array([])

        edge_act_dx = []
        for i, j in edges_dx:
            syn_weight = (synapse_np[i, j] + synapse_np[j, i]) / 2
            edge_act_dx.append(syn_weight)
        edge_act_dx = np.array(edge_act_dx) if len(edge_act_dx) > 0 else np.array([])

        # Normalize activations
        if len(edge_act_dy) > 0 and edge_act_dy.max() > 0:
            edge_act_dy_norm = edge_act_dy / edge_act_dy.max()
        else:
            edge_act_dy_norm = np.zeros_like(edge_act_dy) if len(edge_act_dy) > 0 else np.array([])

        if len(edge_act_dx) > 0 and edge_act_dx.max() > 0:
            edge_act_dx_norm = edge_act_dx / edge_act_dx.max()
        else:
            edge_act_dx_norm = np.zeros_like(edge_act_dx) if len(edge_act_dx) > 0 else np.array([])

        if y_act.max() > 0:
            y_act_norm = y_act / y_act.max()
        else:
            y_act_norm = np.zeros_like(y_act)

        if x_act.max() > 0:
            x_act_norm = x_act / x_act.max()
        else:
            x_act_norm = np.zeros_like(x_act)

        # ============================================================
        # SINGLE FRAME: Both networks shown simultaneously
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 12))

        # Draw Dx edges (red)
        edge_colors_dx = []
        for act_val in edge_act_dx_norm:
            base = 0.75
            r = base + act_val * (1.0 - base)
            g = base * (1 - act_val * 0.85)
            b = base * (1 - act_val * 0.85)
            edge_colors_dx.append((r, g, b, 0.7))

        nx.draw_networkx_edges(
            G_master, pos, ax=ax,
            edgelist=edges_dx,
            edge_color=edge_colors_dx if len(edge_colors_dx) > 0 else 'lightgray',
            width=0.5
        )

        # Draw Dy edges (blue) on top
        edge_colors_dy = []
        for act_val in edge_act_dy_norm:
            base = 0.75
            r = base * (1 - act_val * 0.85)
            g = base * (1 - act_val * 0.75)
            b = base + act_val * (1.0 - base)
            edge_colors_dy.append((r, g, b, 0.7))

        nx.draw_networkx_edges(
            G_master, pos, ax=ax,
            edgelist=edges_dy,
            edge_color=edge_colors_dy if len(edge_colors_dy) > 0 else 'lightgray',
            width=0.5
        )

        # Color nodes: red (x) has priority, then blue (y)
        # Smooth gradient from gray (inactive) to full color (active)
        node_colors = []

        for y_val, x_val in zip(y_act_norm, x_act_norm):
            base = 0.85

            # Priority 1: Red for x (Dx propagation)
            if x_val > y_val:
                # Red gradient: gray → red as x_val: 0 → 1
                r = base + x_val * (1 - base)
                g = base * (1 - x_val * 0.8)
                b = base * (1 - x_val * 0.8)
                node_colors.append((r, g, b))
            else:
                # Blue gradient: gray → blue as y_val: 0 → 1
                r = base * (1 - y_val * 0.8)
                g = base * (1 - y_val * 0.7)
                b = base + y_val * (1 - base)
                node_colors.append((r, g, b))

        nx.draw_networkx_nodes(
            G_master, pos, ax=ax,
            node_color=node_colors,
            node_size=20,
            edgecolors='none'
        )

        title = f'Layer: {layer_idx} - Dual-Network (Dy+Dx)'
        if hub_only:
            title += ' (hub only)'
        ax.set_title(title, fontsize=16, fontweight='bold', color='purple')
        ax.axis('off')

        # Add legend with color indicators
        # Count neurons by which is dominant (x vs y)
        red_count = (x_act_norm > y_act_norm).sum()
        blue_count = (y_act_norm >= x_act_norm).sum()

        legend_text = 'Dual-Network Visualization\n'
        legend_text += 'Blue: y dominant (Dy attention)\n'
        legend_text += 'Red: x dominant (Dx propagation)\n'
        legend_text += 'Intensity: gray (low) → color (high)\n'
        legend_text += f'Neurons: blue={blue_count}, red={red_count}'

        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9),
                family='monospace')

        add_watermark(fig, ax)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        images.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()

        print(f"  Layer {layer_idx+1}/{len(x_frames)} completed (1 frame)")

    return images
