"""Model architectures, faithful to ZAID (arXiv 2602.16098).

The datasets are tabular flow-feature vectors of width d; ZAID applies 1D
convolutions over the feature axis, so we reshape each sample to [1, d] and use
Conv1d throughout.

UniversalCNN  : 3 conv layers (filters 64, 32, 16, kernel 3, ReLU) -> dense
                256, 64, 16 (dropout 0.2) -> sigmoid (binary abnormality prob).
ConvAutoencoder: conv-pool (16, 32) -> bottleneck 64 -> upsample-conv decoder,
                dropout 0.2; reconstruction error is the unsupervised anomaly
                signal fused at zero-day evaluation (Stage 5).
Adapter       : 1x1 conv down-projection -> ReLU -> 1x1 conv up-projection, with
                a residual connection; inserted after the frozen conv backbone
                for lightweight per-zone personalization (ZAID adapter modules).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """1x1 conv down/up projection with residual (bottleneck ratio configurable)."""

    def __init__(self, channels: int, bottleneck: int = 4):
        super().__init__()
        self.down = nn.Conv1d(channels, max(1, channels // bottleneck), kernel_size=1)
        self.act = nn.ReLU()
        self.up = nn.Conv1d(max(1, channels // bottleneck), channels, kernel_size=1)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class UniversalCNN(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2, use_adapters: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
        ])
        self.use_adapters = use_adapters
        self.adapters = nn.ModuleList(
            [Adapter(c) for c in (64, 32, 16)]) if use_adapters else None
        self.relu = nn.ReLU()
        self.flat_dim = 16 * input_dim
        self.head = nn.Sequential(
            nn.Linear(self.flat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def features(self, x):
        h = x.unsqueeze(1)  # [N, 1, d]
        for i, c in enumerate(self.conv):
            h = self.relu(c(h))
            if self.use_adapters:
                h = self.adapters[i](h)
        return h

    def forward(self, x):
        h = self.features(x)
        return self.head(h.flatten(1)).squeeze(-1)  # logits (BCEWithLogits)

    def freeze_backbone(self):
        for c in self.conv:
            for p in c.parameters():
                p.requires_grad = False

    def adapter_and_head_parameters(self):
        params = list(self.head.parameters())
        if self.use_adapters:
            params += list(self.adapters.parameters())
        return params


class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),   # bottleneck (64 filters)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        h = x.unsqueeze(1)
        z = self.encoder(h)
        r = self.decoder(z)
        return r.squeeze(1)

    def reconstruction_error(self, x):
        with torch.no_grad():
            r = self.forward(x)
            return ((r - x) ** 2).mean(dim=1)  # per-sample MSE
