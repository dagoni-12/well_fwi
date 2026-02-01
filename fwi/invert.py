from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from forward import forward_model, lowpass_frequency, ricker_wavelet
from geometry import make_surface_acquisition
from losses import l2_misfit

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class FWIConfig:
    nx: int = 201
    nz: int = 121
    dx_m: float = 10.0
    dt_s: float = 0.001
    nt: int = 750
    f0_hz: float = 12.0
    n_shots: int = 5
    n_receivers: int = 64
    pml_width: int = 20
    accuracy: int = 4
    niter: int = 40
    lr: float = 5e-3
    vmin: float = 1400.0
    vmax: float = 4200.0
    smooth_ks: int = 11
    fmax_stages: tuple[float, ...] = (5.0, 10.0, 15.0)
    out_dir: str = "data/fwi"
    device: str = "cpu"
    plot: bool = False


def make_true_model(cfg: FWIConfig, device: torch.device) -> torch.Tensor:
    z = torch.arange(cfg.nz, device=device, dtype=torch.float32).view(-1, 1)
    background = 1500.0 + 0.6 * z * cfg.dx_m
    v = background.repeat(1, cfg.nx)

    # Add a couple of smooth anomalies.
    zz = torch.arange(cfg.nz, device=device).view(-1, 1)
    xx = torch.arange(cfg.nx, device=device).view(1, -1)
    anomaly1 = 250.0 * torch.exp(-((zz - 60.0) ** 2 + (xx - 90.0) ** 2) / (2 * 12.0**2))
    anomaly2 = -200.0 * torch.exp(-((zz - 85.0) ** 2 + (xx - 140.0) ** 2) / (2 * 10.0**2))
    v = v + anomaly1 + anomaly2
    return v


def make_initial_model(cfg: FWIConfig, device: torch.device) -> torch.Tensor:
    """Simple 1D background without anomalies (harder starting point)."""
    z = torch.arange(cfg.nz, device=device, dtype=torch.float32).view(-1, 1)
    background = 1500.0 + 0.6 * z * cfg.dx_m
    return background.repeat(1, cfg.nx)


def smooth_model(v: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    v4 = v.unsqueeze(0).unsqueeze(0)
    v_smooth = F.avg_pool2d(v4, kernel_size=kernel_size, stride=1, padding=pad)
    return v_smooth[0, 0]


def build_wavelet(cfg: FWIConfig, device: torch.device) -> torch.Tensor:
    return ricker_wavelet(cfg.f0_hz, cfg.dt_s, cfg.nt, device)


def build_source_amplitudes(wavelet: torch.Tensor, n_shots: int) -> torch.Tensor:
    return wavelet.view(1, 1, -1).repeat(n_shots, 1, 1)


def fwi(cfg: FWIConfig, device: torch.device) -> dict[str, np.ndarray]:
    torch.manual_seed(0)

    acquisition = make_surface_acquisition(
        nx=cfg.nx,
        nz=cfg.nz,
        n_shots=cfg.n_shots,
        n_receivers=cfg.n_receivers,
        src_depth=2,
        rec_depth=2,
        device=device,
    )

    v_true = make_true_model(cfg, device)
    v0 = make_initial_model(cfg, device).clamp(cfg.vmin, cfg.vmax)

    wavelet = build_wavelet(cfg, device)
    source_amplitudes = build_source_amplitudes(wavelet, cfg.n_shots)

    with torch.no_grad():
        d_obs = forward_model(
            v_true,
            cfg.dx_m,
            cfg.dt_s,
            source_amplitudes,
            acquisition.source_locations,
            acquisition.receiver_locations,
            pml_width=cfg.pml_width,
            accuracy=cfg.accuracy,
        )

    v = torch.nn.Parameter(v0.clone())
    optim = torch.optim.Adam([v], lr=cfg.lr)

    d_syn_last = None
    for fmax in cfg.fmax_stages:
        print(f"Starting stage fmax={fmax:5.1f}Hz", flush=True)
        for it in range(cfg.niter):
            optim.zero_grad()
            d_syn = forward_model(
                v,
                cfg.dx_m,
                cfg.dt_s,
                source_amplitudes,
                acquisition.source_locations,
                acquisition.receiver_locations,
                pml_width=cfg.pml_width,
                accuracy=cfg.accuracy,
            )
            d_syn_f = lowpass_frequency(d_syn, cfg.dt_s, fmax)
            d_obs_f = lowpass_frequency(d_obs, cfg.dt_s, fmax)
            loss = l2_misfit(d_syn_f, d_obs_f)
            loss.backward()
            optim.step()
            with torch.no_grad():
                v.clamp_(cfg.vmin, cfg.vmax)

            if it % 5 == 0 or it == cfg.niter - 1:
                print(f"fmax={fmax:5.1f}Hz iter={it:3d} loss={loss.item():.6e}", flush=True)
            d_syn_last = d_syn.detach()

    return {
        "v_true": v_true.detach().cpu().numpy(),
        "v0": v0.detach().cpu().numpy(),
        "v_inv": v.detach().cpu().numpy(),
        "d_obs": d_obs.detach().cpu().numpy(),
        "d_syn": None if d_syn_last is None else d_syn_last.cpu().numpy(),
    }


def parse_args() -> FWIConfig:
    parser = argparse.ArgumentParser(description="Simple FWI skeleton (deepwave).")
    parser.add_argument("--nx", type=int, default=FWIConfig.nx)
    parser.add_argument("--nz", type=int, default=FWIConfig.nz)
    parser.add_argument("--dx", type=float, default=FWIConfig.dx_m)
    parser.add_argument("--dt", type=float, default=FWIConfig.dt_s)
    parser.add_argument("--nt", type=int, default=FWIConfig.nt)
    parser.add_argument("--f0", type=float, default=FWIConfig.f0_hz)
    parser.add_argument("--shots", type=int, default=FWIConfig.n_shots)
    parser.add_argument("--receivers", type=int, default=FWIConfig.n_receivers)
    parser.add_argument("--niter", type=int, default=FWIConfig.niter)
    parser.add_argument("--lr", type=float, default=FWIConfig.lr)
    parser.add_argument("--fmax-stages", type=str, default="5,10,15")
    parser.add_argument("--out", type=str, default="data/fwi")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic plots")
    args = parser.parse_args()

    fmax_stages = tuple(float(x) for x in args.fmax_stages.split(",") if x)

    return FWIConfig(
        nx=args.nx,
        nz=args.nz,
        dx_m=args.dx,
        dt_s=args.dt,
        nt=args.nt,
        f0_hz=args.f0,
        n_shots=args.shots,
        n_receivers=args.receivers,
        niter=args.niter,
        lr=args.lr,
        fmax_stages=fmax_stages,
        out_dir=args.out,
        device=args.device,
        plot=args.plot,
    )


def main() -> None:
    cfg = parse_args()
    device = torch.device(resolve_device(cfg.device))

    results = fwi(cfg, device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fwi_result.npz"
    np.savez(out_path, **results)
    print(f"Saved results to {out_path}")

    if cfg.plot:
        if plt is None:
            raise RuntimeError("matplotlib is not available; install it to enable plotting.")

        v_true = results["v_true"]
        v0 = results["v0"]
        v_inv = results["v_inv"]
        d_obs = results["d_obs"]
        d_syn = results.get("d_syn")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        im0 = axes[0].imshow(v_true, cmap="viridis", origin="upper")
        axes[0].set_title("v_true")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(v0, cmap="viridis", origin="upper")
        axes[1].set_title("v0")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(v_inv, cmap="viridis", origin="upper")
        axes[2].set_title("v_inv")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        fig.tight_layout()
        model_path = out_dir / "fwi_models.png"
        fig.savefig(model_path, dpi=200)
        plt.close(fig)

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        shot = d_obs[0]
        im3 = axes2[0].imshow(shot.T, cmap="gray", aspect="auto", origin="upper")
        axes2[0].set_title("d_obs shot0")
        fig2.colorbar(im3, ax=axes2[0], fraction=0.046, pad=0.04)
        if d_syn is not None:
            im4 = axes2[1].imshow(d_syn[0].T, cmap="gray", aspect="auto", origin="upper")
            axes2[1].set_title("d_syn shot0")
            fig2.colorbar(im4, ax=axes2[1], fraction=0.046, pad=0.04)
        else:
            axes2[1].set_axis_off()
            axes2[1].set_title("d_syn shot0 (missing)")
        fig2.tight_layout()
        seis_path = out_dir / "fwi_shots.png"
        fig2.savefig(seis_path, dpi=200)
        plt.close(fig2)

        print(f"Saved plots to {model_path} and {seis_path}")


def resolve_device(device_str: str) -> str:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return device_str
    return "cpu"


if __name__ == "__main__":
    main()
