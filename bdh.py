from dataclasses import dataclass
from typing import Tuple
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
    # ?
    vocab_cnt: int
    seq_len: int

    # model
    H: int
    N: int
    D: int
    L: int
    dropout: float

    use_rope: bool
    use_abs_pos: bool

@dataclass
class BDHTrainParameters:
    epoch_cnt: int
    batch_size: int
    learning_rate: float
    weight_decay: float

class BDH(nn.Module):
    def __init__(self, params: BDHParameters):
        super().__init__()
        H, N, D, L = params.H, params.N, params.D, params.L
        self.N = N
        self.H = H
        self.L = L
        self.linear_attn = LinearAttention(self.N, self.H, params.use_rope)
        self.E = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        self.Dx = nn.Parameter(torch.zeros((H, D, N//H)).normal_(std=0.02))
        self.Dy = nn.Parameter(torch.zeros((H, D, N//H)).normal_(std=0.02))
        self.readout = nn.Parameter(torch.zeros((D, params.vocab_cnt)).normal_(std=0.02))
        self.emb = nn.Embedding(params.vocab_cnt, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(params.dropout)

        self.use_abs_pos = params.use_abs_pos
        if params.use_abs_pos:
            self.pos = nn.Embedding(params.seq_len, D)
            self.register_buffer("pos_idx", torch.arange(params.seq_len, dtype=torch.long), persistent=False)
            nn.init.normal_(self.pos.weight, std=0.02)

    # input_ BT
    # x      BHTNh
    # y      BHTNh
    # v_ast  B1TD
    # a_ast  ??BTD
    #
    # E      ND
    # Dx     HDNh
    # Dy     HDNh

    def forward(self, input_):
        B, T = input_.size()
        v_ast = self.ln(self.emb(input_).unsqueeze(1)) # BT[int] -> B1TD
        if self.use_abs_pos:
            abs_pos_ast = self.pos(self.pos_idx) # TD
            v_ast = v_ast + abs_pos_ast

        for _ in range(self.L):
            # residual?
            x = F.relu(v_ast @ self.Dx) # B1TD @ HDNh -> BHTNh

            a_ast = self.linear_attn(x, x, v_ast) # BHTNh @ (BHTNh^T @ B1TD) -> BHTNh @ (BHNhT @ B1TD) -> BHTNh @ BHNhD -> BHTD

            y = F.relu(a_ast @ self.Dy) * x # BHTD @ HDNh -> BHTNh
            y = y.transpose(1, 2).reshape(B, 1, T, self.N) # BHTNh -> BTHNh -> B1TN
            y = self.drop(y)

            v_ast = v_ast + self.ln(y @ self.E) # B1TD + (B1TN @ ND) -> B1TD + B1TD -> B1TD
            v_ast = self.ln(v_ast)

        return v_ast.squeeze(1) @ self.readout # squeeze(B1TD) @ BDV -> BTD @ BDV -> BTV

# For RoPE pairs we use concatenated layout, instead of interleaved. For
# (a,b,c,d) the pairs are (a,c) and (b,d).
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., Dh], Dh must be even
    Dh = x.shape[-1]
    x1 = x[..., :Dh // 2]
    x2 = x[..., Dh // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q: [B, H, S, Dh]
    cos,sin: [S, Dh]  (broadcasted to [B,H,S,Dh])
    Returns roped q with original dtypes preserved.
    """
    # q_dtype, k_dtype = q.dtype, k.dtype
    q_dtype = q.dtype

    # Option A (stable): promote to cos/sin dtype (usually fp32)
    q = q.to(cos.dtype)
    # k = k.to(cos.dtype)


    # Option B (fast): keep q/k dtype, cast tables down
    # cos = cos.to(q.dtype)
    # sin = sin.to(q.dtype)

    # Broadcast cos/sin over batch and heads
    cos_ = cos.unsqueeze(0).unsqueeze(0)  # [1,1,S,Dh]
    sin_ = sin.unsqueeze(0).unsqueeze(0)  # [1,1,S,Dh]

    q = (q * cos_) + (rotate_half(q) * sin_)
    return q.to(q_dtype)
    # k = (k * cos_) + (rotate_half(k) * sin_)
    # return q.to(q_dtype), k.to(k_dtype)

class RotaryEmbedding(torch.nn.Module):
    """
    Precomputes cos/sin tables for RoPE.
    - head_dim: per-head dimension (Dh), must be even
    - max_position_embeddings: maximum S you will use
    - base (theta): standard 10000.0
    Buffers move with the module device/dtype via .to / .cuda().
    """
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.max_pos = max_position_embeddings

        # inv_freq: [Dh/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # positions: [S]
        t = torch.arange(self.max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [S, Dh/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, Dh]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int | None = None):
        """
        Returns cos,sin of shape [S, Dh] for given S (defaults to max_pos).
        """
        if seq_len is None:
            seq_len = self.max_pos
        if seq_len > self.max_pos:
            raise ValueError(f"Requested RoPE seq_len {seq_len} > max {self.max_pos}")
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class LinearAttention(nn.Module):
    def __init__(self, N: int, H: int, use_rope: bool):
        super().__init__()
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                head_dim=N//H,
                max_position_embeddings=N//H,
                base=10000.0
            )

    def forward(self, Q, K, V):
        if self.use_rope:
            _, _, T, _ = Q.size()
            cos_sin = self.rotary(T) if self.rotary is not None else None # (cos, sin) each [S, Dh]
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

        logits_btv = bdh(x_bs) # BTV
        loss = ce_loss(logits_btv.transpose(1,2), y_bs)
        #logics_bvt = logitcs_bcv.transpose(1, 2) # BVT
        # loss = ce_loss(logitcs_bvt, y_bs)

        total_loss += float(loss.detach()) * B * S
        total_loss_tokens += B * S

        preds = logits_btv.argmax(dim=-1) # BS

        total_correct += (preds == y_bs).sum().item()
        total_tokens += y_bs.numel() # B * S
        total_correct_samples += count_matching_corresponding_rows(preds, y_bs)
        total_samples += preds.size(0)

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
    bdh.train()

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        bdh.parameters(),
        lr=bdh_train_params.learning_rate,
        weight_decay=bdh_train_params.weight_decay
    )

    # dtype = float16
    scaler = torch.amp.GradScaler(device=device.type, enabled=False)

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
        epoch_start_time = time.time()

        total_epoch_loss = 0.0
        total_epoch_tokens = 0
        for batch_idx, (x_bs, y_bs) in enumerate(train_loader):
            print(f"\rbatch: {batch_idx+1}/{batch_cnt}", end="", flush=True)
            x_bs = x_bs.to(device)
            y_bs = y_bs.to(device)
            B, S = x_bs.shape

            logits_bcs = bdh(x_bs)  # BTC (batch, seq, vocab)
            logits_bcs = logits_bcs.transpose(1, 2)  # BCT (batch, vocab, seq)
            loss = ce_loss(logits_bcs, y_bs)
            total_epoch_loss += float(loss.detach()) * B * S
            total_epoch_tokens += B * S

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
    print(f"{'seq_len':<20} {bdh_params.seq_len:>10}")
    print(f"{'vocab_cnt':<20} {bdh_params.vocab_cnt:>10}")
    print(f"{'H':<20} {bdh_params.H:>10}")
    print(f"{'N':<20} {bdh_params.N:>10}")
    print(f"{'D':<20} {bdh_params.D:>10}")
    print(f"{'L':<20} {bdh_params.L:>10}")
    print(f"{'dropout':<20} {bdh_params.dropout:>10}")
    print(f"{'use_rope':<20} {bdh_params.use_rope:>10}")
    print(f"{'use_abs_pos':<20} {bdh_params.use_abs_pos:>10}")

    print("\nBDH Training Parameters:")
    print("-" * 31)
    print(f"{'epoch_cnt':<20} {bdh_train_params.epoch_cnt:>10}")
    print(f"{'batch_size':<20} {bdh_train_params.batch_size:>10}")
    print(f"{'lr':<20} {bdh_train_params.learning_rate:>10}")
    print(f"{'weight_decay':<20} {bdh_train_params.weight_decay:>10}")

    print("\nModel Statistics:")
    print("-" * 31)
    print(f"{'trainable_params':<20} {trainable_params:>10}")
    print(f"{'device':<20} {str(device):>10}")
    print()
