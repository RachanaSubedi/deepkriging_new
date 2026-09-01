"""Parameter-matched two-branch DeepKriging model for DK-3.

Place at: src/model_branch.py

Input ordering must be:
    [spatial basis, temporal basis, 15 standardized covariates]

The basis branch is deliberately bottlenecked while the atmospheric branch
has its own pathway.  The fusion network has roughly the same total parameter
count as the monolithic TPS-K16 comparator, so the experiment tests structure
rather than simply adding capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def block(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class BranchDeepKriging(nn.Module):
    """Separate basis and atmospheric encoders followed by nonlinear fusion."""

    def __init__(
        self,
        n_spatial_basis: int,
        n_temporal_basis: int,
        n_covariates: int = 15,
        branch_width: int = 64,
        branch_embedding: int = 32,
        fusion_width: int = 100,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if min(n_spatial_basis, n_temporal_basis, n_covariates) <= 0:
            raise ValueError("All input groups must contain at least one column")

        self.n_spatial_basis = int(n_spatial_basis)
        self.n_temporal_basis = int(n_temporal_basis)
        self.n_covariates = int(n_covariates)
        self.n_basis = self.n_spatial_basis + self.n_temporal_basis

        # Phi(s) and Psi(t) jointly represent the latent spatiotemporal state.
        # The 32-dimensional bottleneck prevents this representation from
        # overwhelming the physically observed atmospheric covariates.
        self.basis_encoder = nn.Sequential(
            block(self.n_basis, branch_width, dropout),
            block(branch_width, branch_embedding, dropout),
        )

        # Contains GOES C13/C02, NSRDB/background, meteorology, solar geometry,
        # and cyclic time variables. It receives equal embedding capacity.
        self.atmospheric_encoder = nn.Sequential(
            block(self.n_covariates, branch_width, dropout),
            block(branch_width, branch_embedding, dropout),
        )

        self.fusion = nn.Sequential(
            block(2 * branch_embedding, fusion_width, dropout),
            block(fusion_width, fusion_width, dropout),
            nn.Linear(fusion_width, 1),
        )

    @property
    def input_dim(self) -> int:
        return self.n_basis + self.n_covariates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.input_dim}); got {tuple(x.shape)}"
            )
        basis = x[:, : self.n_basis]
        atmosphere = x[:, self.n_basis :]
        h_basis = self.basis_encoder(basis)
        h_atmosphere = self.atmospheric_encoder(atmosphere)
        return self.fusion(torch.cat([h_basis, h_atmosphere], dim=1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
