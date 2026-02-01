from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Acquisition:
    source_locations: torch.Tensor
    receiver_locations: torch.Tensor
    n_shots: int
    n_receivers: int


def make_surface_acquisition(
    nx: int,
    nz: int,
    n_shots: int,
    n_receivers: int,
    src_depth: int = 2,
    rec_depth: int = 2,
    src_x_start: int | None = None,
    src_x_end: int | None = None,
    rec_x_start: int | None = None,
    rec_x_end: int | None = None,
    device: torch.device | None = None,
) -> Acquisition:
    """Create simple surface source/receiver geometry.

    Deepwave expects locations in (z, x) grid indices as int64 tensors.
    """
    if src_x_start is None:
        src_x_start = 5
    if src_x_end is None:
        src_x_end = nx - 6
    if rec_x_start is None:
        rec_x_start = 0
    if rec_x_end is None:
        rec_x_end = nx - 1

    src_x = np.round(np.linspace(src_x_start, src_x_end, n_shots)).astype(np.int64)
    rec_x = np.round(np.linspace(rec_x_start, rec_x_end, n_receivers)).astype(np.int64)

    source_locations = torch.zeros((n_shots, 1, 2), dtype=torch.long, device=device)
    source_locations[:, 0, 0] = int(src_depth)
    source_locations[:, 0, 1] = torch.from_numpy(src_x).to(device=device)

    receiver_locations = torch.zeros((n_shots, n_receivers, 2), dtype=torch.long, device=device)
    receiver_locations[:, :, 0] = int(rec_depth)
    receiver_locations[:, :, 1] = torch.from_numpy(rec_x).to(device=device).unsqueeze(0).repeat(n_shots, 1)

    return Acquisition(
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        n_shots=n_shots,
        n_receivers=n_receivers,
    )
