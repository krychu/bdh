import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import torch
from typing import List, Optional
import networkx as nx
from datasets.build_boardpath_dataset import FLOOR, WALL, START, END, PATH

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

def get_averaged_synapse_topology(synapse_frames: List[torch.Tensor]) -> torch.Tensor:
    """
    Average synapse matrices across all L layers.

    This shows the average functional connectivity across the entire
    computation, smoothing out layer-specific variations.

    Args:
        synapse_frames: List of L tensors, each shape (N, N)

    Returns:
        averaged_topology: (N, N) tensor averaged across all layers
    """
    # Stack along new dimension and average
    stacked = torch.stack(synapse_frames, dim=0)  # (L, N, N)
    averaged = stacked.mean(dim=0)  # (N, N)
    return averaged

def visualize_output_frames(
    output_frames: List[torch.Tensor],
    board_size: int,
    save_path: str,
    duration: int = 500
):
    """
    Create animated GIF of board predictions through layers.

    Args:
        output_frames: List of tensors, each shape (T,) with predicted tokens
        board_size: Size of the board (e.g., 8 for 8x8)
        save_path: Path to save the GIF (e.g., 'output_predictions.gif')
        duration: Duration of each frame in milliseconds (default: 500ms)
    """
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

        # Add title
        ax.set_title(f'Layer {layer_idx} Predictions', fontsize=18, fontweight='bold', pad=20)

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

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    print(f"Saved animated GIF to: {save_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration per frame: {duration}ms")

def visualize_x_frames(
    x_frames: List[torch.Tensor],
    save_path: str,
    duration: int = 500
):
    """
    Create animated GIF of neuron activations through layers.
    Arranges N neurons in a square grid and visualizes activation strength.

    Args:
        x_frames: List of tensors, each shape (N,) with neuron activations
        save_path: Path to save the GIF (e.g., 'neuron_activations.gif')
        duration: Duration of each frame in milliseconds (default: 500ms)
    """
    import io

    N = x_frames[0].shape[0]
    # Arrange neurons in a square grid
    grid_size = int(np.ceil(np.sqrt(N)))

    images = []
    for layer_idx, frame in enumerate(x_frames):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Pad to fill square grid
        activations = frame.cpu().numpy()
        padded = np.zeros(grid_size * grid_size)
        padded[:N] = activations
        grid = padded.reshape(grid_size, grid_size)

        # Display as heatmap with black to red gradient
        im = ax.imshow(grid, cmap='Reds', interpolation='nearest', aspect='auto')

        ax.set_title(f'Layer {layer_idx} Neuron Activations', fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')

        # Add stats text
        stats_text = f'Sparsity: {(activations == 0).sum() / N * 100:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).copy()
        images.append(image)
        plt.close(fig)
        buf.close()

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    print(f"Saved neuron activations GIF to: {save_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Grid size: {grid_size}x{grid_size} (showing {N} neurons)")
    print(f"  Duration per frame: {duration}ms")

def visualize_synapse_frames(
    synapse_frames: List[torch.Tensor],
    save_path: str,
    duration: int = 500
):
    """
    Create animated GIF of synapse connectivity matrices through layers.
    Full NxN heatmap showing all synapse strengths.

    Args:
        synapse_frames: List of tensors, each shape (N, N) with synapse values
        save_path: Path to save the GIF (e.g., 'synapse_matrix.gif')
        duration: Duration of each frame in milliseconds (default: 500ms)
    """
    import io

    N = synapse_frames[0].shape[0]

    images = []
    for layer_idx, frame in enumerate(synapse_frames):
        fig, ax = plt.subplots(figsize=(10, 10))

        synapse_matrix = frame.cpu().numpy()

        # Display as heatmap with black to red gradient
        # Use log scale to handle sparsity
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_matrix = np.log10(synapse_matrix + epsilon)

        im = ax.imshow(log_matrix, cmap='Reds', interpolation='nearest', aspect='auto')

        ax.set_title(f'Layer {layer_idx} Synapse Matrix', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('To Neuron', fontsize=12)
        ax.set_ylabel('From Neuron', fontsize=12)

        # Add minimal ticks
        ax.set_xticks([0, N//2, N-1])
        ax.set_yticks([0, N//2, N-1])
        ax.set_xticklabels(['0', f'{N//2}', f'{N-1}'])
        ax.set_yticklabels(['0', f'{N//2}', f'{N-1}'])

        # Add stats text
        nonzero_count = (synapse_matrix > 0).sum()
        total_synapses = N * N
        stats_text = f'Active: {nonzero_count:,}/{total_synapses:,}\n'
        stats_text += f'Sparsity: {(1 - nonzero_count/total_synapses) * 100:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).copy()
        images.append(image)
        plt.close(fig)
        buf.close()

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    print(f"Saved synapse matrix GIF to: {save_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Matrix size: {N}x{N}")
    print(f"  Duration per frame: {duration}ms")

def visualize_graph_activations(
    x_frames: List[torch.Tensor],
    synapse_frames: List[torch.Tensor],
    save_path: str,
    model,
    top_k_edges: int = 5000,
    duration: int = 500,
    layout_seed: int = 42,
    topology_type: str = 'e_dx',
    hub_only: bool = False
):
    """
    Create animated GIF with dual-layer color encoding:

    - Graph structure (layout): Topology matrix (E @ Dx or Dx.T @ Dx)
    - Edge base color (gray): Structural weight (always visible if > threshold)
    - Edge overlay (green): Synapse activation per layer
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

        title_mode = " (Hub Only)" if hub_only else ""
        ax.set_title(f'Layer {layer_idx} - {topology_desc}{title_mode}',
                     fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add legend explaining the color scheme
        legend_text = (
            'Color Encoding:\n'
            f'  Structure: {topology_desc}\n'
            '  Activation: Synapse strength (edges)\n'
            '              Neuron activity (nodes)\n\n'
            'Gray → Red:\n'
            '  Light gray = Inactive\n'
            '  Bright red = Active'
        )
        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                family='monospace')

        # Add stats
        active_neurons = (activations > 0.1 * activations.max()).sum() if activations.max() > 0 else 0
        active_synapses = (edge_activations > 0.1 * edge_activations.max()).sum() if edge_activations.max() > 0 else 0

        if hub_only:
            stats_text = f'Hub neurons: {N_viz}/{N}\n'
            stats_text += f'Active neurons: {active_neurons}/{N_viz}\n'
        else:
            stats_text = f'Active neurons: {active_neurons}/{N_viz}\n'
        stats_text += f'Active edges: {active_synapses}/{edge_count}\n'
        stats_text += f'Total edges: {edge_count:,}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

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

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    print(f"Saved graph activation GIF to: {save_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration per frame: {duration}ms")

if __name__ == '__main__':
    # Example usage
    from bdh import BDH
    from boardpath import load_bdh, get_device, generate_board

    device = get_device()
    bdh, boardpath_params, bdh_params, bdh_train_params = load_bdh('boardpath.pt', device)
    bdh.to(device)
    bdh.eval()

    # Generate a test board
    input_board, target_board = generate_board(
        size=boardpath_params.board_size,
        max_wall_prob=boardpath_params.wall_prob
    )
    input_flat_bs = input_board.flatten().unsqueeze(0).to(device)

    print("Running inference with frame capture...")
    with torch.no_grad():
        logits_btv, output_frames, x_frames, synapse_frames = bdh(input_flat_bs, capture_frames=True)
        predicted_board = logits_btv.argmax(dim=-1).squeeze(0)

    print("\nCreating visualizations...")

    # Create animated GIF of layer predictions
    visualize_output_frames(
        output_frames=output_frames,
        board_size=boardpath_params.board_size,
        save_path='output_predictions.gif',
        duration=500
    )

    # Create animated GIF of neuron activations
    visualize_x_frames(
        x_frames=x_frames,
        save_path='neuron_activations.gif',
        duration=500
    )

    # Create animated GIF of synapse matrices (full)
    visualize_synapse_frames(
        synapse_frames=synapse_frames,
        save_path='synapse_matrix.gif',
        duration=500
    )

    # Create unified graph visualization
    print("\n=== Creating unified graph visualization ===")
    visualize_graph_activations(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        model=bdh,
        save_path='graph_activations.gif',
        top_k_edges=5000,
        duration=500
    )

    print("\nDone! Generated files:")
    print("  - output_predictions.gif: Board predictions through layers")
    print("  - neuron_activations.gif: Neuron activation patterns through layers")
    print("  - synapse_matrix.gif: Full synapse connectivity matrices through layers")
    print("  - graph_activations.gif: Dual-layer encoding (structure + activation)")
