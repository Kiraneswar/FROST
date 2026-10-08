"""FL primitives: local training, evaluation, and parameter aggregation.

State dicts are the unit of exchange between clients (zones) and the server.
`fedavg` is the sample-weighted mean; Byzantine-robust rules (Stage 4) plug in
here as alternative aggregators with the same signature.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def to_tensor(X: np.ndarray, y: Optional[np.ndarray] = None, device="cpu"):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    if y is None:
        return Xt
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    return Xt, yt


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def local_train(model: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int,
                lr: float, batch_size: int, device="cpu",
                pos_weight: Optional[float] = None,
                only_params: Optional[list] = None,
                seed: int = 0) -> Tuple[Dict[str, torch.Tensor], int]:
    """Train `model` in place on (X, y); return its state dict and sample count."""
    g = torch.Generator().manual_seed(seed)
    Xt, yt = to_tensor(X, y, device)
    n = len(Xt)
    params = only_params if only_params is not None else model.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
    return clone_state(model), n


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device="cpu", batch_size=4096) -> np.ndarray:
    model.eval()
    Xt = to_tensor(X, device=device)
    outs = []
    for i in range(0, len(Xt), batch_size):
        outs.append(torch.sigmoid(model(Xt[i:i + batch_size])).cpu().numpy())
    return np.concatenate(outs) if outs else np.array([])


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "n": int(len(y_true)), "pos_rate": float(np.mean(y_true)),
    }


def fedavg(states: List[Dict[str, torch.Tensor]], weights: List[float]
           ) -> Dict[str, torch.Tensor]:
    """Sample-weighted average of state dicts."""
    assert states, "no states to aggregate"
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            stacked = torch.stack([s[k].float() for s in states], dim=0)
            wt = torch.tensor(w, dtype=torch.float32).view(-1, *([1] * (stacked.dim() - 1)))
            out[k] = (stacked * wt).sum(dim=0).to(states[0][k].dtype)
        else:
            out[k] = states[0][k].clone()  # e.g. num_batches_tracked
    return out


def load_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)
