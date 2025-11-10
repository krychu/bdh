from dataclasses import dataclass
from typing import Tuple
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
    N: int
    D: int
    L: int
    dropout: float

    # train
    # batch_size: int
    # learning_rate: float
    # weight_decay: float

@dataclass
class BDHTrainParameters:
    epoch_cnt: int
    batch_size: int
    learning_rate: float
    weight_decay: float

class BDH(nn.Module):
    def __init__(self, params: BDHParameters):
        super().__init__()
        N, D = params.N, params.D
        self.linear_attn = LinearAttention()
        self.E = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        self.Dx = nn.Parameter(torch.zeros((D, N)).normal_(std=0.02))
        self.Dy = nn.Parameter(torch.zeros((D, N)).normal_(std=0.02))
        self.readout = nn.Parameter(torch.zeros((D, params.vocab_cnt)).normal_(std=0.02))
        self.emb = nn.Embedding(params.vocab_cnt, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(params.dropout)
        self.L = params.L

    # input_ BT
    # x      BTN
    # y      BTN
    # v_ast  BTD
    # a_ast  BTD
    #
    # E      ND
    # Dx     DN
    # Dy     DN

    def forward(self, input_):
        v_ast = self.emb(input_) # BT[int] -> BTD

        for _ in range(self.L):
            # x = x + F.relu(v_ast @ self.Dx) # BTN + (BTD @ DN) -> BTN + BTN -> BTN
            x = F.relu(v_ast @ self.Dx) # BTN + (BTD @ DN) -> BTN + BTN -> BTN

            a_ast = self.linear_attn(x, x, v_ast) # BTN @ (BTN^T @ BTD) -> BTN @ BND -> BTD

            y = F.relu(a_ast @ self.Dy) * x # BTD @ DN -> BTN
            y = self.drop(y)

            v_ast = v_ast + self.ln(y @ self.E) # BTN @ ND -> BTD
            v_ast = self.ln(v_ast)

        return v_ast @ self.readout # BTD @ BDV -> BTV

            # v_ast = F.layer_norm(v_ast, normalized_shape=(v_ast.size(-1), ))

            # a_ast = linear_attention(x, x, v_ast) # BTN @ (BTN^T @ BTD) -> BTN @ BND -> BTD
            # a_ast = F.layer_norm(a, normalized_shape=(a.size(-1), ))

            # y = F.relu(a_ast @ Dy) * x # BTD @ DN -> BTN

class LinearAttention(nn.Module):
    def forward(self, Q, K, V):
        # rope
        return (Q @ K.mT) @ V # no causal mask for boardpath

# def train_one_epoch(
#         bdh: BDH,
#         loader: DataLoader,
#         ce_loss: nn.Module,
#         scaler: torch.amp.GradScaler,
#         optimizer: torch.optim.Optimizer,
#         device: torch.device
# ):
#     for batch_idx, (x_bs, y_bs) in enumerate(train_loader):
#         x_bs = x_bs.to(device)
#         y_bs = y_bs.to(device)

#         logits_bcs = bdh(x_bs)  # BTC (batch, seq, vocab)
#         logits_bcs = logits_bcs.transpose(1, 2)  # BCT (batch, vocab, seq)
#         loss = ce_loss(logits_bcs, y_bs)
#         print(f"loss: {loss}")

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad()

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
    # bdh = BDH(bdh_params).to(device)
    bdh.train()

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        bdh.parameters(),
        lr=bdh_train_params.learning_rate,
        weight_decay=bdh_train_params.weight_decay
    )

    # dtype = float16
    scaler = torch.amp.GradScaler(device=device.type, enabled=False)
    torch.manual_seed(1337)

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
    print(f"{'N':<20} {bdh_params.N:>10}")
    print(f"{'D':<20} {bdh_params.D:>10}")
    print(f"{'L':<20} {bdh_params.L:>10}")
    print(f"{'dropout':<20} {bdh_params.dropout:>10}")

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
