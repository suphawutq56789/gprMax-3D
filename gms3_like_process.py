"""gms3_like_process.py

Make GMS3-like looking B-scan & C-scan from a gprMax *_merged.out file.
- Dewow (moving-average)
- Background removal
- SEC gain
- Robust contrast clip
- C-scan uses a time-gate (t1..t2)

Usage:
  python gms3_like_process.py path/to/model_merged.out --nx 80 --ny 60 --comp Ez --line_y 33 \
      --tgate_ns 18 32 --xcut 50

Outputs:
  *_Bscan_<comp>_y<line_y>_processed.png
  *_Cscan_<comp>_tgate<start>-<end>ns_processed.png
"""

import argparse
import os
import numpy as np

try:
    import h5py
except Exception as e:
    raise SystemExit("Missing dependency: h5py. Install with: pip install h5py") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit("Missing dependency: matplotlib. Install with: pip install matplotlib") from e


def dewow_mavg(B: np.ndarray, win: int = 25) -> np.ndarray:
    """Remove low-frequency 'wow' using moving-average along time for each trace."""
    win = max(3, int(win))
    kernel = np.ones(win, dtype=np.float32) / win
    # Convolve each trace (axis=0 is time)
    trend = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, B)
    return B - trend


def background_remove(B: np.ndarray) -> np.ndarray:
    """Remove background by subtracting mean across traces for each time sample."""
    return B - np.mean(B, axis=1, keepdims=True)


def sec_gain(B: np.ndarray, power: float = 0.85) -> np.ndarray:
    """Simple SEC gain increasing with time."""
    nt = B.shape[0]
    g = (np.linspace(0, 1, nt, dtype=np.float32) + 1e-6) ** float(power)
    return B * g[:, None]


def robust_clip(B: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[np.ndarray, float, float]:
    vmin, vmax = np.percentile(B, [lo, hi])
    return np.clip(B, vmin, vmax), float(vmin), float(vmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_out", help="Path to gprMax *_merged.out (HDF5)")
    ap.add_argument("--nx", type=int, required=True, help="Nx traces in X")
    ap.add_argument("--ny", type=int, required=True, help="Ny traces in Y")
    ap.add_argument("--comp", default="Ez", choices=["Ex", "Ey", "Ez"], help="Field component")
    ap.add_argument("--line_y", type=int, default=33, help="Y-line index for B-scan (0..ny-1)")
    ap.add_argument("--xcut", type=int, default=50, help="X index for cross-section overlay (optional)")
    ap.add_argument("--dewow_win", type=int, default=25)
    ap.add_argument("--gain_pow", type=float, default=0.85)
    ap.add_argument("--clip", type=float, nargs=2, default=(1.0, 99.0), metavar=("LO", "HI"))
    ap.add_argument("--tgate_ns", type=float, nargs=2, default=(18.0, 32.0), metavar=("T1", "T2"),
                    help="Time gate in ns for C-scan (start end)")
    # Survey geometry for real-world axis labels
    ap.add_argument("--x0", type=float, default=0.5, help="Starting X position (m)")
    ap.add_argument("--y0", type=float, default=0.5, help="Starting Y position (m)")
    ap.add_argument("--dx", type=float, default=0.05, help="Trace spacing in X (m)")
    ap.add_argument("--dy", type=float, default=0.05, help="Line spacing in Y (m)")
    args = ap.parse_args()

    merged_out = args.merged_out
    base = os.path.splitext(merged_out)[0]

    with h5py.File(merged_out, "r") as f:
        if "rxs/rx1" not in f:
            raise SystemExit("Unexpected file structure: missing 'rxs/rx1'")
        ds_path = f"rxs/rx1/{args.comp}"
        if ds_path not in f:
            raise SystemExit(f"Missing dataset: {ds_path}")
        data = np.array(f[ds_path], dtype=np.float32)  # (nt, ntraces)
        dt = float(f.attrs.get("dt", np.nan))

    nt, ntr = data.shape
    expected = args.nx * args.ny
    if ntr < expected:
        raise SystemExit(f"Not enough traces in merged file: ntraces={ntr}, expected nx*ny={expected}")
    if ntr > expected:
        data = data[:, :expected]

    # ----- B-scan (pick a y-line) -----
    iy = int(np.clip(args.line_y, 0, args.ny - 1))
    B = data[:, iy * args.nx:(iy + 1) * args.nx]  # (nt, nx)

    # Processing for B-scan
    Bp = dewow_mavg(B, win=args.dewow_win)
    Bp = background_remove(Bp)
    Bp = sec_gain(Bp, power=args.gain_pow)
    Bp, vmin, vmax = robust_clip(Bp, lo=args.clip[0], hi=args.clip[1])

    # Calculate real-world extents for B-scan
    x_start = args.x0
    x_end = args.x0 + (args.nx - 1) * args.dx
    time_ns = nt * dt * 1e9 if np.isfinite(dt) and dt > 0 else nt  # Time in ns

    # ----- C-scan (time-gate max amplitude) -----
    if np.isfinite(dt) and dt > 0:
        t1_ns, t2_ns = args.tgate_ns
        i1 = int(max(0, (t1_ns * 1e-9) / dt))
        i2 = int(min(nt, (t2_ns * 1e-9) / dt))
    else:
        # fallback: interpret as samples if dt missing
        i1, i2 = int(args.tgate_ns[0]), int(args.tgate_ns[1])
        i1 = max(0, min(nt - 1, i1))
        i2 = max(i1 + 1, min(nt, i2))

    gate = data[i1:i2, :]  # (ng, ntr)
    amp = np.max(np.abs(gate), axis=0)  # (ntr,)

    # reshape to (ny, nx)
    C = amp.reshape(args.ny, args.nx)

    # mild normalisation/clip
    Cc, cmin, cmax = robust_clip(C, lo=2.0, hi=98.0)

    # Calculate real-world extents for C-scan
    y_start = args.y0
    y_end = args.y0 + (args.ny - 1) * args.dy

    # ----- Combined plot: B-scan (top) + C-scan (bottom) -----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={'height_ratios': [1.5, 1]})

    # B-scan
    im1 = ax1.imshow(Bp, aspect="auto", cmap="gray", extent=[x_start, x_end, time_ns, 0],
                     vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Time (ns)")
    ax1.set_title(f"B-scan ({args.comp}) at line_y={iy}")
    if args.xcut is not None:
        xc = int(np.clip(args.xcut, 0, args.nx - 1))
        xc_m = args.x0 + xc * args.dx
        ax1.axvline(x=xc_m, color='blue', linewidth=1)

    # C-scan
    im2 = ax2.imshow(Cc, origin="lower", aspect="auto", cmap="gray", vmin=cmin, vmax=cmax,
                     extent=[x_start, x_end, y_start, y_end])
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Width (m)")
    ax2.set_title(f"C-scan ({args.comp}) time-gate {args.tgate_ns[0]:g}-{args.tgate_ns[1]:g} ns")

    plt.tight_layout()
    out_combined = f"{base}_BScan_CScan_{args.comp}_processed.png"
    plt.savefig(out_combined, dpi=200)
    plt.close()

    print("Saved:")
    print(" ", out_combined)


if __name__ == "__main__":
    main()
