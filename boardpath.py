import argparse
import random
from typing import Tuple
from dataclasses import asdict
from torch.utils.data import DataLoader
from utils.build_boardpath_dataset import *
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
        board_size=12,
        train_count=8000,
        val_count=500,
        wall_prob=0.3
    )

    bdh_params = BDHParameters(
        V=get_vocab_cnt(),
        T=boardpath_params.board_size ** 2,
        H=4,
        N=4*1028,
        D=1*128,
        L=16,
        dropout=0.2, # 0.05
        use_rope=True,
        use_abs_pos=False
    )

    bdh_train_params = BDHTrainParameters(
        epoch_cnt=100,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.1, # 0.05
        grad_clip=None
    )

    return boardpath_params, bdh_params, bdh_train_params

def get_device():
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
        path: str
):
    best_val_acc_samples = 0

    def epoch_callback(
            bdh: BDH,
            epoch_idx: int,
            epoch_loss: float,
            epoch_time: int,
            val_loader: DataLoader,
            ce_loss: nn.Module,
            device: torch.device
    ) -> None:
        nonlocal best_val_acc_samples
        val_loss, val_acc_tokens, val_acc_samples = evaluate(
            bdh=bdh,
            ce_loss=ce_loss,
            loader=val_loader,
            device=device
        )

        mark = "" if val_acc_samples <= best_val_acc_samples else "*"
        if epoch_idx==-1:
            best_val_acc_samples = 0
            print(f"epoch: --- [trn] loss: ------ [val] loss: {val_loss:.4f}, cell acc: {val_acc_tokens:.3f}, board acc: {val_acc_samples:.3f}")
        else:
            print(f"epoch: {epoch_idx+1:03d} [trn] loss: {epoch_loss:.4f} [val] loss: {val_loss:.4f}, cell acc: {val_acc_tokens:.3f}, board acc: {val_acc_samples:.3f} (time: {epoch_time:.0f}s) {mark}")

        if val_acc_samples > best_val_acc_samples:
            best_val_acc_samples = val_acc_samples
            if epoch_idx != -1:
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
        logits_btv, output_frames, x_frames, y_frames, synapse_frames = bdh(input_flat_bs, capture_frames=True)
        predicted = logits_btv.argmax(dim=-1) # BS

    print("\nINPUT BOARD:")
    print(format_board(input_board.flatten(), boardpath_params.board_size))

    print("\nTARGET BOARD:")
    print(format_board(target_board.flatten(), boardpath_params.board_size))

    print("\nPREDICTED BOARD:")
    print(format_board(predicted.squeeze(0).cpu(), boardpath_params.board_size))

    print("\nLegend: . = Floor, # = Wall, S = Start, E = End, * = Path")

    # Generate visualizations
    # Note: This generates hub-only (connected neurons) visualizations for clarity.
    # For full-view or alternative visualizations, see visualize_more.py
    print("\nGenerating visualizations...")
    from utils.visualize_refactored import (
        generate_board_frames,
        generate_graph_frames as generate_graph_frames_new,
        generate_interleaved_graph_frames,
        combine_image_lists,
        save_gif
    )
    from utils.visualize import (
        generate_graph_frames as generate_graph_frames_old
    )

    # 1. Generate board prediction frames
    print("\n  1/7: Generating board predictions...")
    board_images = generate_board_frames(
        output_frames=output_frames,
        board_size=boardpath_params.board_size
    )

    # 2. Generate Dx signal flow graph - BOTH OLD AND NEW
    print("\n  2/7: Generating E @ Dx - signal flow (OLD version)...")
    hub_flow_images_old = generate_graph_frames_old(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        y_frames=y_frames,
        model=bdh,
        top_k_edges=5000,
        topology_type='e_dx',
        interpolate_frames=1,
        visualization_mode='signal_flow'
    )

    print("\n  2b/7: Generating E @ Dx - signal flow (NEW version)...")
    hub_flow_images_new = generate_graph_frames_new(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        y_frames=y_frames,
        model=bdh,
        top_k_edges=5000,
        topology_type='e_dx',
        visualization_mode='signal_flow'
    )

    # Use the new version for the rest of the pipeline
    hub_flow_images = hub_flow_images_new

    # 3. Generate Dy co-activation graph
    print("\n  3/7: Generating Dy.T @ Dy (attention decoder)...")
    dy_hub_images = generate_graph_frames_new(
        x_frames=x_frames,
        synapse_frames=synapse_frames,
        y_frames=y_frames,
        model=bdh,
        top_k_edges=5000,
        topology_type='dy_coact'
    )

    # 4. Generate interleaved dual-network visualization
    print("\n  4/7: Generating Interleaved Dy+Dx (two-stage flow)...")
    interleaved_hub_images = generate_interleaved_graph_frames(
        x_frames=x_frames,
        y_frames=y_frames,
        synapse_frames=synapse_frames,
        model=bdh,
        top_k_edges=5000
    )

    # 5. Save individual GIFs
    print("\n  5/7: Saving individual GIFs...")
    save_gif(board_images, 'output_predictions.gif', duration=170)
    save_gif(hub_flow_images_old, 'graph_e_dx_hub_flow_OLD.gif', duration=170)
    save_gif(hub_flow_images_new, 'graph_e_dx_hub_flow_NEW.gif', duration=170)
    save_gif(dy_hub_images, 'graph_dy_coact_hub.gif', duration=170)
    save_gif(interleaved_hub_images, 'graph_interleaved_hub.gif', duration=200)

    # 6. Create combined visualization
    print("\n  6/7: Creating combined visualization...")

    # Three-way: board + interleaved + Dx flow (THE ULTIMATE!)
    combined_board_interleaved = combine_image_lists([board_images, interleaved_hub_images, hub_flow_images], spacing=20)
    save_gif(combined_board_interleaved, 'combined_board_interleaved.gif', duration=200)

    # 7. Summary
    print("\n  7/7: Done!")
    print("\n✓ Visualization files generated:")
    print("  Individual:")
    print("    - output_predictions.gif (board predictions)")
    print("    - graph_e_dx_hub_flow_OLD.gif (Dx signal flow - ORIGINAL)")
    print("    - graph_e_dx_hub_flow_NEW.gif (Dx signal flow - REFACTORED)")
    print("    - graph_dy_coact_hub.gif (Dy attention decoder)")
    print("    - graph_interleaved_hub.gif (Dy→Dx two-stage) ⭐⭐")
    print("  Combined:")
    print("    - combined_board_interleaved.gif (board + interleaved + Dx flow) ⭐⭐⭐ ULTIMATE!")
    print()
    print("  🔍 COMPARISON: Check graph_e_dx_hub_flow_OLD.gif vs graph_e_dx_hub_flow_NEW.gif")
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

    if args.seed:
        seed = int(args.seed)
        set_all_seeds(seed) # 1337
        print(f"seed: {seed}")
    else:
        print("seed: random")

    if args.mode == "train":
        run_training()
    elif args.mode == "inference":
        run_inference(args.model)
