from dataclasses import dataclass
from typing import Tuple, List
import math
import time
import random
from collections import deque
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

@dataclass
class BDHParameters:
    # Architecture
    V: int  # vocabulary size
    T: int  # tokens (sequence length)
    H: int  # heads
    N: int  # neurons
    D: int  # latent dimension
    L: int  # layers

    # Training/regularization
    dropout: float
    use_rope: bool
    use_abs_pos: bool

@dataclass
class BDHTrainParameters:
    epoch_cnt: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float | None

# Shapes:
#
#   input_    BT
#   x         BHTNh
#   y         BHTNh
#   a_ast     BHTD
#   v_ast     B1TD
#   E         ND
#   Dx        HDNh
#   Dy        HDNh
#   readout   DV
#
#   B  batch
#   N  neurons
#   D  latent dimension (low-rank)
#   T  tokens (sequence)
#   H  heads
#   Nh neurons/head
#   V  vocabulary
#   L  layers
#
class BDH(nn.Module):
    def __init__(self, params: BDHParameters):
        super().__init__()
        V, T, H, N, D, L = params.V, params.T, params.H, params.N, params.D, params.L
        self.N = N
        self.H = H
        self.L = L
        self.linear_attn = LinearAttention(self.N, self.H, params.use_rope, T)
        self.E = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        self.Dx = nn.Parameter(torch.zeros((H, D, N//H)).normal_(std=0.02))
        self.Dy = nn.Parameter(torch.zeros((H, D, N//H)).normal_(std=0.02))
        self.readout = nn.Parameter(torch.zeros((D, V)).normal_(std=0.02))
        self.emb = nn.Embedding(V, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(params.dropout)

        self.use_abs_pos = params.use_abs_pos
        if params.use_abs_pos:
            self.pos = nn.Embedding(T, D)
            self.register_buffer("pos_idx", torch.arange(T, dtype=torch.long), persistent=False)
            # scale by 1/sqrt(L) since we add positional embeddings L times
            nn.init.normal_(self.pos.weight, mean=0.0, std=1.0/math.sqrt(L))

    def forward(self, input_, capture_frames = False):
        B, T = input_.size()

        # BT -> B1TD
        v_ast = self.ln(self.emb(input_).unsqueeze(1))

        if self.use_abs_pos:
            # TD
            abs_pos_ast = self.pos(self.pos_idx)

        if capture_frames:
            output_frames: List[torch.Tensor] = []
            x_frames: List[torch.Tensor] = []
            synapse_frames: List[torch.Tensor] = []

        for _ in range(self.L):
            if self.use_abs_pos:
                # B1TD + TD -> B1TD
                v_ast = v_ast + abs_pos_ast

            # B1TD @ HDNh -> BHTNh
            x = F.relu(v_ast @ self.Dx)

            # BHTNh @ (BHTNh^T @ B1TD) -> BHTNh @ (BHNhT @ B1TD) -> BHTNh @ BHNhD -> BHTD
            a_ast = self.linear_attn(x, x, v_ast)

            # (BHTD @ HDNh) * BHTNh -> BHTNh * BHTNh -> BHTNh
            y = F.relu(self.ln(a_ast) @ self.Dy) * x
            # re(tr(BHTNh)) -> re(BTHNh) -> B1TN
            y = y.transpose(1, 2).reshape(B, 1, T, self.N)
            y = self.drop(y)

            # B1TD + (B1TN @ ND) -> B1TD + B1TD -> B1TD
            v_ast = v_ast + self.ln(y @ self.E)
            v_ast = self.ln(v_ast)

            if capture_frames:
                self._capture_frame(v_ast, x, y, T, output_frames, x_frames, synapse_frames)

        # squ(B1TD) @ DV -> BTD @ DV -> BTV
        logits = v_ast.squeeze(1) @ self.readout

        if capture_frames:
            return logits, output_frames, x_frames, synapse_frames
        return logits

    def _capture_frame(
        self,
        v_ast: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        T: int,
        output_frames: List[torch.Tensor],
        x_frames: List[torch.Tensor],
        synapse_frames: List[torch.Tensor]
    ):
        # B1TD @ DV -> BTD @ DV -> BTV
        logits = v_ast.squeeze(1) @ self.readout
        predicted = logits.argmax(dim=-1)
        output_frames.append(predicted[0])  # (T,) - single sample

        # Use only first sample for x and synapse frames
        # re(tr(BHTNh[0])) -> re(tr(HTNh)) -> re(THNh) -> TN (first sample only)
        x_reshaped = x[0].transpose(0, 1).reshape(T, self.N)
        # (N,) - avg activation per neuron across tokens (positions)
        x_frames.append(x_reshaped.mean(dim=0).detach().clone())

        # Compute synapse matrix for first sample: average(x.T @ y) across tokens
        # re(tr(BHTNh[0])) -> re(tr(HTNh)) -> re(THNh) -> TN (first sample only)
        y_reshaped = y[0].transpose(0, 1).reshape(T, self.N)

        # TN^T @ TN -> NT @ TN -> NN (avg over tokens)
        synapse = (x_reshaped.T @ y_reshaped) / T
        synapse_frames.append(synapse.detach().clone())

# For RoPE pairs we use concatenated layout, instead of interleaved. For
# (a,b,c,d) the pairs are (a,c) and (b,d).
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., Nh], Dh must be even
    Nh = x.shape[-1]
    x1 = x[..., :Nh // 2]
    x2 = x[..., Nh // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Returns roped q with original dtype preserved
# q          BHTNh
# cos, sin   TNh (broadcasted to BHTNh)
def apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q = q.to(cos.dtype)

    # Broadcast cos/sin over batch and heads
    cos_ = cos.unsqueeze(0).unsqueeze(0)  # 11TNh
    sin_ = sin.unsqueeze(0).unsqueeze(0)  # 11TNh

    q = (q * cos_) + (rotate_half(q) * sin_)
    return q.to(q.dtype)

# Precomputes cos/sin tables for RoPE
# head_dim     per-head dimension (Nh), must be even
# max_T        maximum sequence length
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, max_T: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_T = max_T

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)) # Nh/2
        t = torch.arange(self.max_T, dtype=torch.float32) # T
        freqs = torch.outer(t, inv_freq)  # TNh/2
        emb = torch.cat((freqs, freqs), dim=-1)  # TNh

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    # Returns cos, sin of shape TNh for given T (defaults to max_T)
    def forward(self, seq_len: int | None = None):
        if seq_len is None:
            seq_len = self.max_T
        assert seq_len <= self.max_T
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class LinearAttention(nn.Module):
    def __init__(self, N: int, H: int, use_rope: bool, max_T: int):
        super().__init__()
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                head_dim=N//H,
                max_T=max_T,
                base=10000.0
            )

    def forward(self, Q, K, V):
        if self.use_rope:
            _, _, T, _ = Q.size()
            cos_sin = self.rotary(T)
            cos, sin = cos_sin
            QR = apply_rope(Q, cos, sin)
        else:
            QR = Q

        KR = QR
        return (QR @ KR.mT) @ V

def count_matching_corresponding_rows(a: torch.Tensor, b: torch.Tensor) -> int:
    assert(len(a.shape)==2 and len(b.shape)==2)
    assert(a.shape == b.shape)
    matches = (a == b).all(dim=1)
    return int(matches.sum().item())

@torch.no_grad()
def evaluate(
        bdh: BDH,
        ce_loss: nn.Module,
        loader: DataLoader,
        device: torch.device
):
    bdh.eval()

    total_loss = 0.0
    total_loss_tokens = 0.0
    total_tokens = 0
    total_correct = 0
    total_correct_samples = 0
    total_samples = 0

    for x_bs, y_bs in loader:
        x_bs = x_bs.to(device)
        y_bs = y_bs.to(device)
        B, S = x_bs.shape

        logits = bdh(x_bs) # BTV
        loss = ce_loss(logits.transpose(1,2), y_bs)
        total_loss += loss.item() * B * S
        total_loss_tokens += B * S

        predicted = logits.argmax(dim=-1) # BS
        total_correct += (predicted == y_bs).sum().item()
        total_tokens += y_bs.numel() # B*S
        total_correct_samples += count_matching_corresponding_rows(predicted, y_bs)
        total_samples += predicted.size(0)

    avg_loss = total_loss / total_loss_tokens
    acc_tokens = total_correct / total_tokens
    acc_samples = total_correct_samples / total_samples
    return avg_loss, acc_tokens, acc_samples

def train(
        bdh: BDH,
        bdh_train_params: BDHTrainParameters,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epoch_callback
):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        bdh.parameters(),
        lr=bdh_train_params.learning_rate,
        weight_decay=bdh_train_params.weight_decay
    )

    epoch_callback(
        bdh=bdh,
        epoch_idx=-1,
        epoch_loss=0.0,
        epoch_time=0.0,
        val_loader=val_loader,
        ce_loss=ce_loss,
        device=device
    )

    batch_cnt = len(train_loader)
    for epoch_idx in range(bdh_train_params.epoch_cnt):
        bdh.train()
        epoch_start_time = time.time()

        total_epoch_loss = 0.0
        total_epoch_tokens = 0
        for batch_idx, (x_bs, y_bs) in enumerate(train_loader):
            print(f"\rbatch: {batch_idx+1}/{batch_cnt}", end="", flush=True)
            x_bs = x_bs.to(device)
            y_bs = y_bs.to(device)
            B, S = x_bs.shape

            optimizer.zero_grad(set_to_none=True)
            logits = bdh(x_bs) # BTV
            logits= logits.transpose(1, 2) # BVT
            loss = ce_loss(logits, y_bs)
            loss.backward()
            if bdh_train_params.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(bdh.parameters(), bdh_train_params.grad_clip)
            optimizer.step()

            total_epoch_loss += loss.item() * B * S
            total_epoch_tokens += B * S

        epoch_time = time.time() - epoch_start_time
        epoch_loss = total_epoch_loss / total_epoch_tokens

        print("\r", end='', flush=True)
        epoch_callback(
            bdh=bdh,
            epoch_idx=epoch_idx,
            epoch_loss=epoch_loss,
            epoch_time=epoch_time,
            val_loader=val_loader,
            ce_loss=ce_loss,
            device=device
        )

def bdh_summary(
        bdh_params: BDHParameters,
        bdh_train_params: BDHTrainParameters,
        bdh: BDH,
        device: torch.device
) -> None:
    trainable_params = sum(p.numel() for p in bdh.parameters() if p.requires_grad)

    print("BDH Parameters:")
    print("-" * 31)
    print(f"{'V (vocab)':<20} {bdh_params.V:>10}")
    print(f"{'T (tokens)':<20} {bdh_params.T:>10}")
    print(f"{'H (heads)':<20} {bdh_params.H:>10}")
    print(f"{'N (neurons)':<20} {bdh_params.N:>10}")
    print(f"{'D (latent_dim)':<20} {bdh_params.D:>10}")
    print(f"{'L (layers)':<20} {bdh_params.L:>10}")
    print(f"{'dropout':<20} {bdh_params.dropout:>10}")
    print(f"{'use_rope':<20} {bdh_params.use_rope:>10}")
    print(f"{'use_abs_pos':<20} {bdh_params.use_abs_pos:>10}")

    print("\nBDH Training Parameters:")
    print("-" * 31)
    print(f"{'epoch_cnt':<20} {bdh_train_params.epoch_cnt:>10}")
    print(f"{'batch_size':<20} {bdh_train_params.batch_size:>10}")
    print(f"{'lr':<20} {bdh_train_params.learning_rate:>10}")
    print(f"{'weight_decay':<20} {bdh_train_params.weight_decay:>10}")
    print(f"{'grad_clip':<20} {str(bdh_train_params.grad_clip):>10}")

    print("\nModel Statistics:")
    print("-" * 31)
    print(f"{'trainable_params':<20} {trainable_params:>10}")
    print(f"{'device':<20} {str(device):>10}")
    print()
