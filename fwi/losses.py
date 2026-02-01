from __future__ import annotations

import torch


def l2_misfit(d_syn: torch.Tensor, d_obs: torch.Tensor) -> torch.Tensor:
    return torch.mean((d_syn - d_obs) ** 2)
