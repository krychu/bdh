import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import torch
from typing import List, Optional
import networkx as nx
from datasets.build_boardpath_dataset import FLOOR, WALL, START, END, PATH

# ============================================================================
# Helper Functions for Frame Management
# ============================================================================

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
        topology_type: 'e_dx' (communication) or 'dx_coact' (co-activation)

    Returns:
        topology: (N, N) tensor where topology[i,j] is connection strength
    """
    # E: (N, D)
    # Dx: (H, D, N//H)
    H, D, Nh = model.Dx.shape
    N = H * Nh

    # Reshape Dx from (H, D, N//H) to (D, N)
    Dx_reshaped = model.Dx.permute(1, 0, 2).reshape(D, N)

    if topology_type == 'e_dx':
        # E @ Dx: Communication structure
        # Shows how neuron i's output affects neuron j via embedding updates
        topology = model.E @ Dx_reshaped
    elif topology_type == 'dx_coact':
        # Dx.T @ Dx: Co-activation structure
        # Shows which neurons respond to similar embedding patterns
        topology = Dx_reshaped.T @ Dx_reshaped
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
    interpolate_frames: int = 1
) -> List[Image.Image]:
    """
    Generate PIL images with dual-layer color encoding:

    - Graph structure (layout): Topology matrix (E @ Dx or Dx.T @ Dx)
    - Edge base color (gray): Structural weight (always visible if > threshold)
    - Edge overlay (red): Synapse activation per layer
    - Node base color: Light gray (always visible)
    - Node overlay (red): Neuron activation per layer

    Args:
        x_frames: List of L tensors, each shape (N,) with neuron activations per layer
        synapse_frames: List of L tensors, each shape (N, N) with synapse values per layer
        save_path: Path to save the GIF (e.g., 'graph_activations.gif')
        model: BDH model (to extract topology)
        top_k_edges: Number of strongest connections to include in graph (default: 5000)
        duration: Duration of each frame in milliseconds (default: 500ms)
        layout_seed: Random seed for reproducible layout (default: 42)
        topology_type: 'e_dx' (communication) or 'dx_coact' (co-activation)
        hub_only: If True, show only connected neurons (zoomed view)
        interpolate_frames: Number of interpolated frames between each layer (1=no interpolation, 3=2 extra frames)
    """
    import io
    from matplotlib.colors import Normalize

    # Get parameter topology
    topology_matrix = get_parameter_topology(model, topology_type=topology_type)

    N = topology_matrix.shape[0]

    topology_desc = {
        'e_dx': 'E @ Dx (communication)',
        'dx_coact': 'Dx.T @ Dx (co-activation)'
    }[topology_type]

    print(f"Building graph with top {top_k_edges} connections from {N} neurons...")
    print(f"Topology: {topology_desc}")
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

        for i in range(len(x_frames) - 1):
            # Add current frame
            x_frames_interp.append(x_frames[i])
            synapse_frames_interp.append(synapse_frames[i])

            # Add interpolated frames
            for j in range(1, interpolate_frames):
                alpha = j / interpolate_frames  # 0 to 1
                # Linear interpolation
                x_interp = (1 - alpha) * x_frames[i] + alpha * x_frames[i + 1]
                synapse_interp = (1 - alpha) * synapse_frames[i] + alpha * synapse_frames[i + 1]
                x_frames_interp.append(x_interp)
                synapse_frames_interp.append(synapse_interp)

        # Add final frame
        x_frames_interp.append(x_frames[-1])
        synapse_frames_interp.append(synapse_frames[-1])

        x_frames = x_frames_interp
        synapse_frames = synapse_frames_interp
        print(f"Total frames after interpolation: {len(x_frames)}")

    # Normalize structural weights for gray base color (0 to 1)
    struct_norm = Normalize(vmin=0, vmax=edge_weights_structural.max())
    edge_weights_struct_norm = struct_norm(edge_weights_structural)

    images = []
    for layer_idx, (x_frame, synapse_frame) in enumerate(zip(x_frames, synapse_frames)):
        fig, ax = plt.subplots(figsize=(12, 12))

        # Node activations (full array)
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

        # Compute synapse activation for edges in the graph
        edge_activations = []
        for i, j in edge_list:
            syn_weight = (synapse_np[i, j] + synapse_np[j, i]) / 2
            edge_activations.append(syn_weight)

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

        # Build title: topology (mode) - layer: X
        title = f'{topology_desc}'
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
