"""Lightweight MLP that predicts a per-position steering coefficient.

Two architectures (selected via ``per_cat``):

  per_cat=False (default, shared-trunk):
    shared trunk:  Linear(d_in, d_hidden) -> GELU
    per-cat heads: ModuleList of Linear(d_hidden, 1)
    output:        Softplus(head[c](trunk(h))) -> non-negative alpha

  per_cat=True (one full MLP per category):
    per-cat MLPs:  ModuleList of
                     Linear(d_in, d_hidden) -> GELU -> Linear(d_hidden, 1)
    output:        Softplus(mlp[c](h)) -> non-negative alpha
    i.e. each category gets its OWN independent MLP (no shared weights).

The module is jointly trained with the category vectors V so that
  shift = alpha(h, cat) * V[cat]
where alpha is predicted from the residual stream h at each position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor


class CatCoefMLP(nn.Module):
    def __init__(self, d_in: int, n_cats: int, d_hidden: int = 64,
                 per_cat: bool = False):
        super().__init__()
        self.d_in = d_in
        self.n_cats = n_cats
        self.d_hidden = d_hidden
        self.per_cat = bool(per_cat)
        if self.per_cat:
            # One independent MLP per category.
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_in, d_hidden),
                    nn.GELU(),
                    nn.Linear(d_hidden, 1),
                )
                for _ in range(n_cats)
            ])
        else:
            # Shared trunk + per-category linear heads.
            self.trunk = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
            )
            self.heads = nn.ModuleList([
                nn.Linear(d_hidden, 1) for _ in range(n_cats)
            ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)

    def forward(self, h: Tensor, cat_ids: LongTensor) -> Tensor:
        """Predict non-negative coefficient for each position.

        Args:
            h:       (N, d_in) residual-stream vectors at disagreement positions
            cat_ids: (N,) category index per position

        Returns:
            alpha:   (N,) non-negative coefficients via Softplus
        """
        out = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        if self.per_cat:
            for c in cat_ids.unique():
                mask = cat_ids == c
                raw = self.mlps[int(c.item())](h[mask]).squeeze(-1)
                out[mask] = F.softplus(raw)
        else:
            feat = self.trunk(h)
            for c in cat_ids.unique():
                mask = cat_ids == c
                raw = self.heads[int(c.item())](feat[mask]).squeeze(-1)
                out[mask] = F.softplus(raw)
        return out
