import argparse
import math
import random
from dataclasses import asdict
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets.build_boardpath_dataset import *
from bdh import *

def get_loaders(boardpath_params: BoardPathParameters, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(boardpath_params)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader

def get_config() -> Tuple[BoardPathParameters, BDHParameters, BDHTrainParameters]:
    boardpath_params = BoardPathParameters(
        board_size=8,
        train_count=4000,
        val_count=500,
        wall_prob=0.3
    )

    bdh_params = BDHParameters(
        vocab_cnt=get_vocab_cnt(),
        seq_len=boardpath_params.board_size * boardpath_params.board_size, # TODO: **2?
        H=4,
        # N=4*1028,
        N=2*2056,
        # D=128,
        D=128,
        L=8,
        # dropout=0.05,
        dropout=0.2,
        use_rope=True,
        use_abs_pos=False
    )

    bdh_train_params = BDHTrainParameters(
        epoch_cnt=100,
        batch_size=64,
        learning_rate=1e-3,
        # weight_decay=0.05
        weight_decay=0.1,
        grad_clip=None
    )

    return boardpath_params, bdh_params, bdh_train_params

def get_device():
    # return torch.device("cpu") # TODO

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def save_bdh(
        bdh: BDH,
        boardpath_params: BoardPathParameters,
        bdh_params: BDHParameters,
        bdh_train_params: BDHTrainParameters,
        path: str
):
    ckpt = {
        "bdh_state_dict": bdh.state_dict(),
        "boardpath_params_dict": asdict(boardpath_params),
        "bdh_params_dict": asdict(bdh_params),
        "bdh_train_params_dict": asdict(bdh_train_params),
    }
    torch.save(ckpt, path)

def load_bdh(path: str, map_location="cpu") -> Tuple[BDH, BoardPathParameters, BDHParameters, BDHTrainParameters]:
    ckpt = torch.load(path, map_location=map_location)
    boardpath_params = BoardPathParameters(**ckpt["boardpath_params_dict"])
    bdh_params = BDHParameters(**ckpt["bdh_params_dict"])
    bdh_train_params = BDHTrainParameters(**ckpt["bdh_train_params_dict"])
    bdh = BDH(bdh_params)
    bdh.load_state_dict(ckpt["bdh_state_dict"])
    return bdh, boardpath_params, bdh_params, bdh_train_params

def create_epoch_callback(
        boardpath_params: BoardPathParameters,
        bdh_params: BDHParameters,
        bdh_train_params: BDHTrainParameters,
        path: str,
        # val_loader: DataLoader,
        # device: torch.device
):
    best_val_loss = math.inf

    def epoch_callback(
            bdh: BDH,
            epoch_idx: int,
            epoch_loss: float,
            epoch_time: int,
            val_loader: DataLoader,
            ce_loss: nn.Module,
            device: torch.device
    ) -> None:
        nonlocal best_val_loss
        val_loss, val_acc_tokens, val_acc_samples = evaluate(
            bdh=bdh,
            ce_loss=ce_loss,
            loader=val_loader,
            device=device
        )

        if epoch_idx==-1:
            best_val_loss = math.inf
            print(f"epoch: --- [trn] loss: ------ [val] loss: {val_loss:.4f}, cell acc: {val_acc_tokens:.3f}, board acc: {val_acc_samples:.3f}")
        else:
            print(f"epoch: {epoch_idx+1:03d} [trn] loss: {epoch_loss:.4f} [val] loss: {val_loss:.4f}, cell acc: {val_acc_tokens:.3f}, board acc: {val_acc_samples:.3f} (time: {epoch_time:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_bdh(
                bdh=bdh,
                boardpath_params=boardpath_params,
                bdh_params=bdh_params,
                bdh_train_params=bdh_train_params,
                path=path
            )

    return epoch_callback

def run_training():
    boardpath_params, bdh_params, bdh_train_params = get_config()
    device = get_device()
    train_loader, val_loader = get_loaders(boardpath_params, bdh_train_params.batch_size)

    bdh = BDH(bdh_params).to(device)
    epoch_callback = create_epoch_callback(
        boardpath_params=boardpath_params,
        bdh_params=bdh_params,
        bdh_train_params=bdh_train_params,
        path="boardpath.pt"
    )

    print()
    boardpath_summary(boardpath_params)
    bdh_summary(bdh_params, bdh_train_params, bdh, device)

    train(
        bdh=bdh,
        bdh_train_params=bdh_train_params,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epoch_callback=epoch_callback
    )

def run_inference(path: str):
    device=get_device()
    bdh, boardpath_params, bdh_params, bdh_train_params = load_bdh(path, device)
    print(f"Model loaded from: {path}")

    bdh.to(device)
    bdh.eval()
    input_board, target_board = generate_board(
        size=boardpath_params.board_size,
        max_wall_prob=boardpath_params.wall_prob
    )
    input_flat_bs = input_board.flatten().unsqueeze(0).to(device) # [1, seq_len]

    with torch.no_grad():
        logits_btv, output_frames, x_frames, synapse_frames = bdh(input_flat_bs, capture_frames=True)
        predicted = logits_btv.argmax(dim=-1) # BS

    print("\nINPUT BOARD:")
    print(format_board(input_board.flatten(), boardpath_params.board_size))

    print("\nTARGET BOARD:")
    print(format_board(target_board.flatten(), boardpath_params.board_size))

    print("\nPREDICTED BOARD:")
    print(format_board(predicted.squeeze(0).cpu(), boardpath_params.board_size))

    print("\nLegend: . = Floor, # = Wall, S = Start, E = End, * = Path")

    # Generate visualizations
    print("\nGenerating visualizations...")
    from visualize import (
        visualize_output_frames,
        visualize_x_frames,
        visualize_synapse_frames,
        visualize_graph_activations,
        get_parameter_topology
    )

    visualize_output_frames(
        output_frames=output_frames,
        board_size=boardpath_params.board_size,
        save_path='output_predictions.gif',
        duration=500
    )

    visualize_x_frames(
        x_frames=x_frames,
        save_path='neuron_activations.gif',
        duration=500
    )

    visualize_synapse_frames(
        synapse_frames=synapse_frames,
        save_path='synapse_matrix.gif',
        duration=500
    )

    # Generate graph visualizations (2 topologies × 2 modes = 4 variants)
    print("\n  Creating graph visualizations...")

    # 1. E @ Dx (communication) - Full
    print("\n    1/4: E @ Dx (communication) - Full view")
    visualize_graph_activations(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        model=bdh,
        save_path='graph_e_dx_full.gif',
        top_k_edges=5000,
        duration=500,
        topology_type='e_dx',
        hub_only=False
    )

    # 2. E @ Dx (communication) - Hub only (with interpolation)
    print("\n    2/4: E @ Dx (communication) - Hub only (smooth)")
    visualize_graph_activations(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        model=bdh,
        save_path='graph_e_dx_hub.gif',
        top_k_edges=5000,
        duration=170,  # Shorter duration for smoother animation
        topology_type='e_dx',
        hub_only=True,
        interpolate_frames=5  # 3x interpolation for smooth transitions
    )

    # 3. Dx.T @ Dx (co-activation) - Full
    print("\n    3/4: Dx.T @ Dx (co-activation) - Full view")
    visualize_graph_activations(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        model=bdh,
        save_path='graph_dx_coact_full.gif',
        top_k_edges=5000,
        duration=500,
        topology_type='dx_coact',
        hub_only=False
    )

    # 4. Dx.T @ Dx (co-activation) - Hub only (with interpolation)
    print("\n    4/4: Dx.T @ Dx (co-activation) - Hub only (smooth)")
    visualize_graph_activations(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        model=bdh,
        save_path='graph_dx_coact_hub.gif',
        top_k_edges=5000,
        duration=170,  # Shorter duration for smoother animation
        topology_type='dx_coact',
        hub_only=True,
        interpolate_frames=3  # 3x interpolation for smooth transitions
    )

    print("\nVisualization files generated:")
    print("  - output_predictions.gif")
    print("  - neuron_activations.gif")
    print("  - synapse_matrix.gif")
    print("  - graph_e_dx_full.gif (E@Dx communication, all neurons)")
    print("  - graph_e_dx_hub.gif (E@Dx communication, hub only, 3x interpolated)")
    print("  - graph_dx_coact_full.gif (Dx.T@Dx co-activation, all neurons)")
    print("  - graph_dx_coact_hub.gif (Dx.T@Dx co-activation, hub only, 3x interpolated)")
    print()

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def format_board(board_tensor: torch.Tensor, board_size: int) -> str:
    """Format a flattened board tensor as a visual grid."""
    board = board_tensor.view(board_size, board_size)
    symbols = {FLOOR: '.', WALL: '#', START: 'S', END: 'E', PATH: '*'}

    result = []
    for row in board:
        result.append(' '.join(symbols.get(int(cell), str(int(cell))) for cell in row))
    return '\n'.join(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BDH Boardpath Training and Inference")
    parser.add_argument("--mode", choices=["train", "inference"], required=True,
                        help="Mode to run: train (trains and saced model) or inference (loads model and runs on random sample)")
    parser.add_argument("--seed",
                        help="Seed, only relevant in train mode")
    parser.add_argument("--model", default="boardpath.pt",
                        help="Model file path (default: boardpath.pt)")
    args = parser.parse_args()

    if args.mode == "train":
        if args.seed:
            seed = int(args.seed)
            set_all_seeds(seed) # 1337
            print(f"seed: {seed}")
        else:
            print("seed: random")
        run_training()
    elif args.mode == "inference":
        run_inference(args.model)
