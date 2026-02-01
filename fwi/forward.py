from __future__ import annotations

from typing import Iterable

import torch

try:
    import deepwave
except Exception:  # pragma: no cover - handled by runtime check
    deepwave = None


def ricker_wavelet(f0_hz: float, dt_s: float, nt: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(nt, device=device, dtype=torch.float32) * dt_s
    t0 = 1.0 / f0_hz
    pi_f_t = torch.pi * f0_hz * (t - t0)
    w = (1.0 - 2.0 * pi_f_t**2) * torch.exp(-pi_f_t**2)
    return w


def lowpass_frequency(data: torch.Tensor, dt_s: float, fmax_hz: float | None) -> torch.Tensor:
    if fmax_hz is None:
        return data
    n = data.shape[-1]
    spec = torch.fft.rfft(data, dim=-1)
    freq = torch.fft.rfftfreq(n, d=dt_s).to(spec.device)
    mask = (freq <= fmax_hz).to(spec.dtype)
    spec = spec * mask
    return torch.fft.irfft(spec, n=n, dim=-1)


def forward_model(
    v_m_s: torch.Tensor,
    dx_m: float,
    dt_s: float,
    source_amplitudes: torch.Tensor,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    pml_width: int | Iterable[int] = 20,
    accuracy: int = 4,
) -> torch.Tensor:
    if deepwave is None:
        raise RuntimeError("deepwave is not available; install it to run forward modeling.")

    v_m_s = v_m_s.contiguous()
    source_amplitudes = source_amplitudes.contiguous()

    out = deepwave.scalar(
        v_m_s,
        dx_m,
        dt_s,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=pml_width,
        accuracy=accuracy,
    )

    if isinstance(out, (tuple, list)):
        return out[-1]
    return out
