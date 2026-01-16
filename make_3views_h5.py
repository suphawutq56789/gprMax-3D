import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def find_dataset(h5, comp: str) -> str:
    comp_l = comp.lower()
    candidates = []

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            n = name.lower()
            if (n.endswith("/" + comp_l) or n.endswith(comp_l) or ("/" + comp_l + "/" in n) or (comp_l in n)):
                if np.issubdtype(obj.dtype, np.number):
                    candidates.append(name)

    h5.visititems(walk)

    if not candidates:
        raise RuntimeError(f"หา dataset ของ comp='{comp}' ไม่เจอในไฟล์ .out")

    best = None
    best_score = -1
    for n in candidates:
        shape = h5[n].shape
        if len(shape) < 2:
            continue
        score = int(np.prod(shape))
        if score > best_score:
            best_score = score
            best = n

    if best is None:
        raise RuntimeError(f"เจอ candidates แต่ไม่มีตัวไหนมีมิติ >=2 สำหรับ comp='{comp}'")

    return best


def load_data(out_path: str, comp: str):
    with h5py.File(out_path, "r") as f:
        ds_name = find_dataset(f, comp)
        data = np.array(f[ds_name])

    if data.ndim == 2:
        return data, ds_name

    if data.ndim == 3:
        if data.shape[0] < 10:
            data2 = data[0, :, :]
        else:
            data2 = data[:, :, 0]
        if data2.ndim != 2:
            data2 = data2.reshape(data2.shape[0], -1)
        return data2, ds_name

    data2 = data.reshape(data.shape[0], -1)
    return data2, ds_name


def clip_norm_signed(A: np.ndarray, p: float) -> np.ndarray:
    s = np.percentile(np.abs(A), p)
    if s <= 0:
        return A
    return np.clip(A, -s, s) / s


def clip_norm_positive(A: np.ndarray, p: float) -> np.ndarray:
    s = np.percentile(A, p)
    if s <= 0:
        return A
    return np.clip(A, 0, s) / s


def hilbert_envelope(x2d: np.ndarray) -> np.ndarray:
    x = x2d.astype(np.float64, copy=False)
    n = x.shape[0]
    Xf = np.fft.fft(x, axis=0)

    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    Xf *= h[:, None]
    xa = np.fft.ifft(Xf, axis=0)
    env = np.abs(xa)
    return env


def background_remove(cube: np.ndarray) -> np.ndarray:
    mean_xy = cube.mean(axis=(1, 2), keepdims=True)
    return cube - mean_xy


def apply_time_gain(cube: np.ndarray, gain_pow: float) -> np.ndarray:
    if gain_pow <= 0:
        return cube
    t = np.arange(cube.shape[0], dtype=np.float64) + 1.0
    g = (t / t.max()) ** gain_pow
    return cube * g[:, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_out")

    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)

    ap.add_argument("--comp", default="Ez")
    ap.add_argument("--line_y", type=int, default=5)
    ap.add_argument("--x_cross", type=int, default=None)

    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=-1)

    ap.add_argument("--t_slice", type=int, default=None)
    ap.add_argument("--tmin", type=int, default=None)
    ap.add_argument("--tmax", type=int, default=None)

    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--dpi", type=int, default=220)

    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--bgremove", action="store_true")
    ap.add_argument("--mute", type=int, default=0, help="zero early samples to reduce direct wave/coupling")
    ap.add_argument("--gain_pow", type=float, default=0.0, help="time-gain power (e.g. 1.5)")

    args = ap.parse_args()

    data, ds_name = load_data(args.merged_out, args.comp)
    nt, ntr = data.shape

    expected = args.nx * args.ny
    if ntr != expected:
        raise ValueError(f"traces={ntr} แต่ nx*ny={expected} (เช็ก --nx/--ny). dataset={ds_name}")

    t0 = max(0, args.t0)
    t1 = nt if args.t1 == -1 else min(nt, args.t1)
    data = data[t0:t1, :]
    nt2 = data.shape[0]

    cube = data.reshape(nt2, args.ny, args.nx)

    if args.mute > 0:
        m = int(np.clip(args.mute, 0, nt2))
        cube[:m, :, :] = 0.0

    if args.bgremove:
        cube = background_remove(cube)

    cube = apply_time_gain(cube, args.gain_pow)

    y_idx = int(np.clip(args.line_y, 0, args.ny - 1))
    x_cross = args.nx // 2 if args.x_cross is None else int(np.clip(args.x_cross, 0, args.nx - 1))
    t_slice = nt2 // 2 if args.t_slice is None else int(np.clip(args.t_slice, 0, nt2 - 1))

    bscan = cube[:, y_idx, :]
    cross = cube[:, :, x_cross]

    if args.env:
        bscan_v = hilbert_envelope(bscan)
        cross_v = hilbert_envelope(cross)

        if args.tmin is not None and args.tmax is not None:
            a = int(np.clip(args.tmin, 0, nt2 - 1))
            b = int(np.clip(args.tmax, a + 1, nt2))
            cube_env = hilbert_envelope(cube.reshape(nt2, -1)).reshape(nt2, args.ny, args.nx)
            cscan_v = cube_env[a:b, :, :].max(axis=0)
            cscan_title = f"Horizontal section (C-scan top view) | max envelope t={a}:{b} (crop {t0}:{t1})"
        else:
            cscan_v = np.abs(cube[t_slice, :, :])
            cscan_title = f"Horizontal section (C-scan top view) | |amp| at t={t_slice} (crop {t0}:{t1})"

        bscan_v = clip_norm_positive(bscan_v, args.clip)
        cross_v = clip_norm_positive(cross_v, args.clip)
        cscan_v = clip_norm_positive(cscan_v, args.clip)

        vmin, vmax = 0.0, 1.0
    else:
        bscan_v = clip_norm_signed(bscan, args.clip)
        cross_v = clip_norm_signed(cross, args.clip)
        cscan_v = clip_norm_signed(cube[t_slice, :, :], args.clip)

        cscan_title = f"Horizontal section (C-scan top view) | t={t_slice} (crop {t0}:{t1})"
        vmin, vmax = -1.0, 1.0

    cmap = "gray_r" if args.invert else "gray"

    fig = plt.figure(figsize=(13.5, 7.5))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)

    ax1.imshow(bscan_v, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"Longitudinal section (B-scan) | {args.comp} | y={y_idx} | dataset={ds_name}")
    ax1.set_xlabel("x trace index")
    ax1.set_ylabel("time sample index")

    ax2.imshow(cross_v, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Cross section (B-scan) | x={x_cross}")
    ax2.set_xlabel("y line index")
    ax2.set_ylabel("time sample index")

    ax3.imshow(cscan_v, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.set_title(cscan_title)
    ax3.set_xlabel("x trace index")
    ax3.set_ylabel("y line index")

    tag = "env" if args.env else "raw"
    out_png = args.merged_out.replace(
        ".out",
        f"_3views_{args.comp}_{tag}_y{y_idx}_x{x_cross}_crop{t0}-{t1}.png"
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi)
    print("Saved:", out_png)
    plt.show()


if __name__ == "__main__":
    main()
