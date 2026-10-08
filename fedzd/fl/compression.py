"""Bandwidth-aware update compression, applied to the client's delta
(update_state - pulled_global) before transmission.

  topk     : keep the top-k-magnitude entries per tensor, zero the rest
             (gradient sparsification, Aji & Heafield 2017 / DGC).
  quantize : uniform per-tensor quantization to `bits` bits.

Each returns the reconstructed delta plus the achieved compression ratio
(original bits / transmitted bits), so we can report ratio vs accuracy trade-off.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def _delta(update: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]
           ) -> Dict[str, torch.Tensor]:
    return {k: (update[k].float() - base[k].float()) for k in update
            if update[k].dtype.is_floating_point}


def topk_compress(update, base, k_frac: float) -> Tuple[Dict[str, torch.Tensor], float]:
    delta = _delta(update, base)
    total, kept = 0, 0
    out = {}
    for k, d in delta.items():
        flat = d.flatten()
        n = flat.numel(); total += n
        kk = max(1, int(k_frac * n))
        kept += kk
        if kk < n:
            _, idx = torch.topk(flat.abs(), kk)
            mask = torch.zeros_like(flat)
            mask[idx] = 1.0
            flat = flat * mask
        out[k] = flat.view_as(d)
    # transmitted: kept values (32b) + kept indices (32b) => 64b each vs 32b dense
    ratio = (total * 32) / max(1, kept * 64)
    return out, float(ratio)


def quantize_compress(update, base, bits: int) -> Tuple[Dict[str, torch.Tensor], float]:
    delta = _delta(update, base)
    levels = 2 ** bits - 1
    out = {}
    for k, d in delta.items():
        lo, hi = d.min(), d.max()
        if hi > lo:
            q = torch.round((d - lo) / (hi - lo) * levels)
            out[k] = q / levels * (hi - lo) + lo
        else:
            out[k] = d.clone()
    ratio = 32.0 / bits
    return out, float(ratio)


def apply_compression(update, base, cfg: dict) -> Tuple[Dict[str, torch.Tensor], float]:
    """Returns (reconstructed_state, compression_ratio). Reconstructed state =
    base + decompressed_delta, i.e. what the server actually applies."""
    method = cfg.get("method", "none")
    if method == "none":
        return {k: update[k].clone() for k in update}, 1.0
    if method == "topk":
        d, ratio = topk_compress(update, base, float(cfg.get("k_frac", 0.1)))
    elif method == "quantize":
        d, ratio = quantize_compress(update, base, int(cfg.get("bits", 8)))
    else:
        raise ValueError(f"unknown compression method '{method}'")
    recon = {k: update[k].clone() for k in update}
    for k in d:
        recon[k] = (base[k].float() + d[k]).to(update[k].dtype)
    return recon, ratio
