"""
src/model.py

DeepKriging network architecture.

Input  : (batch, 426)  — 411 Wendland RBF basis + 15 covariates
Output : (batch,)      — predicted CSI residual

Architecture:
    No input BatchNorm — phi values already in [0,1] from basis_functions.py,
    covariates standardized externally. Input BN would normalise phi columns
    using running stats from only 3 training stations, causing near-zero
    running_var for some phi columns and exploding outputs at inference time.

    Input (426)
        → Dense(100, ReLU) → Dropout(0.5) → BatchNorm(100)
        → Dense(100, ReLU) → Dropout(0.5) → BatchNorm(100)
        → Dense(100, ReLU) → Dropout(0.5)
        → Dense(1, linear)
"""

import torch
import torch.nn as nn


class DeepKriging(nn.Module):

    def __init__(self, input_dim: int,
                 hidden_size: int = 100,
                 dropout: float   = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            # ── Hidden layer 1 ───────────────────────────────
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),

            # ── Hidden layer 2 ───────────────────────────────
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),

            # ── Hidden layer 3 ───────────────────────────────
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Output ────────────────────────────────────────
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)