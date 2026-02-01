import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from welly import Well

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from bruges.filters import ricker as bruges_ricker
except Exception:
    bruges_ricker = None


def _find_curve(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"Missing curves. Tried: {', '.join(names)}")


def _clean_series(values: np.ndarray) -> np.ndarray:
    cleaned = values.astype(float)
    cleaned[cleaned <= -998.0] = np.nan
    return cleaned


def ricker_wavelet(freq_hz: float, dt_s: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    if bruges_ricker is not None:
        try:
            w, t = bruges_ricker(duration_s, dt_s, freq_hz)
            return np.asarray(w, dtype=float), np.asarray(t, dtype=float)
        except Exception:
            pass

    # Fallback implementation if bruges is unavailable or signature differs.
    t = np.arange(-duration_s / 2.0, duration_s / 2.0 + dt_s / 2.0, dt_s)
    pi_f_t = np.pi * freq_hz * t
    w = (1.0 - 2.0 * pi_f_t**2) * np.exp(-pi_f_t**2)
    return w, t


def load_las_curves(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    well = Well.from_las(str(path))
    basis_curve = None
    for basis_name in ("DEPT", "DEPTH", "MD", "TVD", "DEPTH_MD"):
        basis_curve = well.get_curve(basis_name)
        if basis_curve is not None:
            break

    if basis_curve is None:
        df = well.df()
    else:
        df = well.df(basis=basis_curve.basis)

    df.index = df.index.astype(float)

    dtc_col = _find_curve(df, ["DTC", "DT", "DTCO", "DT_DTC"])
    rhob_col = _find_curve(df, ["RHOB", "RHOZ", "DEN", "RHO"])

    depth_m = df.index.to_numpy(dtype=float)
    dtc = _clean_series(df[dtc_col].to_numpy())
    rhob = _clean_series(df[rhob_col].to_numpy())

    return depth_m, dtc, rhob


def impedance_reflectivity(
    depth_m: np.ndarray,
    dtc_us_per_ft: np.ndarray,
    rhob_g_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_m) & np.isfinite(dtc_us_per_ft) & np.isfinite(rhob_g_cm3)
    depth_m = depth_m[valid]
    dtc_us_per_ft = dtc_us_per_ft[valid]
    rhob_g_cm3 = rhob_g_cm3[valid]

    velocity_m_s = 304800.0 / dtc_us_per_ft
    rho_kg_m3 = rhob_g_cm3 * 1000.0
    impedance = velocity_m_s * rho_kg_m3

    reflectivity = np.zeros_like(impedance)
    z1 = impedance[:-1]
    z2 = impedance[1:]
    reflectivity[1:] = (z2 - z1) / (z2 + z1)

    # Two-way time from slowness and depth sampling.
    depth_step_m = np.median(np.diff(depth_m))
    depth_step_ft = depth_step_m * 3.280839895
    dt_s = (dtc_us_per_ft * 1e-6) * depth_step_ft * 2.0
    twt_s = np.cumsum(dt_s)
    twt_s -= twt_s[0]

    return velocity_m_s, rho_kg_m3, impedance, reflectivity, twt_s


def resample_reflectivity(twt_s: np.ndarray, reflectivity: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dt_s = float(np.median(np.diff(twt_s)))
    t_uniform = np.arange(twt_s[0], twt_s[-1] + dt_s / 2.0, dt_s)
    r_uniform = np.interp(t_uniform, twt_s, reflectivity)
    return t_uniform, r_uniform, dt_s


def synthesize_trace(reflectivity: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    return np.convolve(reflectivity, wavelet, mode="same")


def run_constant_test(vp_m_s: float, rho_kg_m3: float, n_samples: int, dt_s: float) -> dict[str, np.ndarray]:
    impedance = np.full(n_samples, vp_m_s * rho_kg_m3, dtype=float)
    reflectivity = np.zeros_like(impedance)
    wavelet, t_wavelet = ricker_wavelet(25.0, dt_s, 0.128)
    synthetic = synthesize_trace(reflectivity, wavelet)
    t = np.arange(n_samples, dtype=float) * dt_s
    return {
        "t": t,
        "impedance": impedance,
        "reflectivity": reflectivity,
        "wavelet": wavelet,
        "wavelet_time": t_wavelet,
        "synthetic": synthetic,
    }


def run_step_test(
    vp1_m_s: float,
    rho1_kg_m3: float,
    vp2_m_s: float,
    rho2_kg_m3: float,
    n_samples: int,
    dt_s: float,
    step_index: int) -> dict[str, np.ndarray]:
    impedance = np.empty(n_samples, dtype=float)
    impedance[:step_index] = vp1_m_s * rho1_kg_m3
    impedance[step_index:] = vp2_m_s * rho2_kg_m3

    reflectivity = np.zeros_like(impedance)
    z1 = impedance[:-1]
    z2 = impedance[1:]
    reflectivity[1:] = (z2 - z1) / (z2 + z1)

    wavelet, t_wavelet = ricker_wavelet(25.0, dt_s, 0.128)
    synthetic = synthesize_trace(reflectivity, wavelet)
    t = np.arange(n_samples, dtype=float) * dt_s

    return {
        "t": t,
        "impedance": impedance,
        "reflectivity": reflectivity,
        "wavelet": wavelet,
        "wavelet_time": t_wavelet,
        "synthetic": synthetic,
        "step_index": np.array([step_index], dtype=int),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic seismogram from LAS logs.")
    parser.add_argument("--las", type=str, default=None, help="Path to LAS file")
    parser.add_argument("--f0", type=float, default=25.0, help="Ricker central frequency (Hz)")
    parser.add_argument("--wavelet-length", type=float, default=0.128, help="Wavelet duration (s)")
    parser.add_argument("--out", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Show and save diagnostic plots")
    parser.add_argument("--plot-out", type=str, default=None, help="Directory for plots (defaults to --out)")
    parser.add_argument("--test-constant", action="store_true", help="Run constant Vp/rho test case")
    parser.add_argument("--test-vp", type=float, default=2500.0, help="Constant Vp for test (m/s)")
    parser.add_argument("--test-rho", type=float, default=2200.0, help="Constant rho for test (kg/m3)")
    parser.add_argument("--test-n", type=int, default=1024, help="Number of samples for test")
    parser.add_argument("--test-dt", type=float, default=0.002, help="Sample interval for test (s)")
    parser.add_argument("--test-step", action="store_true", help="Run single impedance step test case")
    parser.add_argument("--test-step-vp1", type=float, default=2500.0, help="Vp before step (m/s)")
    parser.add_argument("--test-step-rho1", type=float, default=2200.0, help="Rho before step (kg/m3)")
    parser.add_argument("--test-step-vp2", type=float, default=3000.0, help="Vp after step (m/s)")
    parser.add_argument("--test-step-rho2", type=float, default=2400.0, help="Rho after step (kg/m3)")
    parser.add_argument("--test-step-index", type=int, default=512, help="Sample index for impedance step")
    args = parser.parse_args()

    if args.test_constant:
        test = run_constant_test(args.test_vp, args.test_rho, args.test_n, args.test_dt)
        max_abs = float(np.max(np.abs(test["synthetic"])))
        print(f"Constant test: max|synthetic|={max_abs:.6e} (should be near 0)")

        if args.plot:
            if plt is None:
                raise RuntimeError("matplotlib is not available; install it to enable plotting.")
            plot_dir = Path(args.plot_out) if args.plot_out else Path(args.out)
            plot_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            axes[0].plot(test["impedance"], test["t"], color="black")
            axes[0].set_title("Impedance (const)")
            axes[0].invert_yaxis()
            axes[1].plot(test["reflectivity"], test["t"], color="black")
            axes[1].set_title("Reflectivity (0)")
            axes[1].axvline(0.0, color="gray", linewidth=0.8)
            axes[2].plot(test["synthetic"], test["t"], color="black")
            axes[2].set_title("Synthetic (near 0)")
            axes[2].axvline(0.0, color="gray", linewidth=0.8)
            fig.tight_layout()
            plot_path = plot_dir / "constant_test.png"
            fig.savefig(plot_path, dpi=200)
            plt.show()
            print(f"Saved plot to {plot_path}")
        return

    if args.test_step:
        test = run_step_test(
            args.test_step_vp1,
            args.test_step_rho1,
            args.test_step_vp2,
            args.test_step_rho2,
            args.test_n,
            args.test_dt,
            args.test_step_index,
        )
        step_idx = int(test["step_index"][0])
        print(
            "Step test: reflectivity spike at index "
            f"{step_idx} (R={test['reflectivity'][step_idx]:.6f})"
        )

        if args.plot:
            if plt is None:
                raise RuntimeError("matplotlib is not available; install it to enable plotting.")
            plot_dir = Path(args.plot_out) if args.plot_out else Path(args.out)
            plot_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            axes[0].plot(test["impedance"], test["t"], color="black")
            axes[0].set_title("Impedance (step)")
            axes[0].invert_yaxis()
            axes[1].plot(test["reflectivity"], test["t"], color="black")
            axes[1].set_title("Reflectivity (spike)")
            axes[1].axvline(0.0, color="gray", linewidth=0.8)
            axes[2].plot(test["synthetic"], test["t"], color="black")
            axes[2].set_title("Synthetic")
            axes[2].axvline(0.0, color="gray", linewidth=0.8)
            fig.tight_layout()
            plot_path = plot_dir / "step_test.png"
            fig.savefig(plot_path, dpi=200)
            plt.show()
            print(f"Saved plot to {plot_path}")
        return

    las_path = Path(args.las) if args.las else None
    if las_path is None:
        las_candidates = sorted(Path("data/well_logs_data").glob("*.las"))
        if not las_candidates:
            raise FileNotFoundError("No LAS files found in data/well_logs_data")
        las_path = las_candidates[0]

    depth_m, dtc, rhob = load_las_curves(las_path)
    vp_m_s, rho_kg_m3, impedance, reflectivity, twt_s = impedance_reflectivity(depth_m, dtc, rhob)
    t_uniform, r_uniform, dt_s = resample_reflectivity(twt_s, reflectivity)

    wavelet, t_wavelet = ricker_wavelet(args.f0, dt_s, args.wavelet_length)
    synthetic = synthesize_trace(r_uniform, wavelet)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{las_path.stem}_synthetic.npz"

    np.savez(
        out_path,
        depth_m=depth_m,
        twt_s=twt_s,
        vp_m_s=vp_m_s,
        rho_kg_m3=rho_kg_m3,
        impedance=impedance,
        reflectivity=reflectivity,
        t_uniform=t_uniform,
        reflectivity_uniform=r_uniform,
        wavelet=wavelet,
        wavelet_time=t_wavelet,
        synthetic=synthetic,
        dt_s=dt_s,
        las_path=str(las_path),
    )

    print(f"Saved synthetic trace to {out_path}")

    if args.plot:
        if plt is None:
            raise RuntimeError("matplotlib is not available; install it to enable plotting.")

        plot_dir = Path(args.plot_out) if args.plot_out else out_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

        axes[0].plot(impedance, twt_s, color="black")
        axes[0].set_title("Impedance")
        axes[0].set_xlabel("kg/m2/s")
        axes[0].invert_yaxis()

        axes[1].plot(reflectivity, twt_s, color="black")
        axes[1].set_title("Reflectivity")
        axes[1].set_xlabel("R")
        axes[1].axvline(0.0, color="gray", linewidth=0.8)

        axes[2].plot(wavelet, t_wavelet, color="black")
        axes[2].set_title("Ricker")
        axes[2].set_xlabel("Amplitude")
        axes[2].set_ylabel("Time (s)")
        axes[2].invert_yaxis()

        axes[3].plot(synthetic, t_uniform, color="black")
        axes[3].set_title("Synthetic")
        axes[3].set_xlabel("Amplitude")
        axes[3].invert_yaxis()

        fig.suptitle(las_path.stem)
        fig.tight_layout()

        plot_path = plot_dir / f"{las_path.stem}_synthetic.png"
        fig.savefig(plot_path, dpi=200)
        plt.show()
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
