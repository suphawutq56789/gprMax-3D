import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def find_dataset(h5, comp: str) -> str:
    """
    Heuristic หา dataset field ที่ตรงกับ component (เช่น Ez) ในไฟล์ .out (HDF5)
    ของ gprMax แต่ละเวอร์ชันชื่อ path อาจต่างกัน
    """
    comp_l = comp.lower()
    candidates = []

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            n = name.lower()
            # เอาแบบที่มักเจอบ่อย: มี /rxs/ ... /Ez หรือมีชื่อ Ez อยู่ใน path
            if (n.endswith("/" + comp_l) or n.endswith(comp_l) or ("/" + comp_l + "/" in n) or (comp_l in n)):
                # เก็บเฉพาะที่เป็น numeric
                if np.issubdtype(obj.dtype, np.number):
                    candidates.append(name)

    h5.visititems(walk)

    if not candidates:
        raise RuntimeError(f"หา dataset ของ comp='{comp}' ไม่เจอในไฟล์ .out")

    # เลือก dataset ที่ขนาดใหญ่สุด (มักเป็น field จริง)
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
    """
    คืนค่า data เป็น (nt, ntraces)
    """
    with h5py.File(out_path, "r") as f:
        ds_name = find_dataset(f, comp)
        data = np.array(f[ds_name])

    if data.ndim == 2:
        return data, ds_name

    # บางทีเป็น 3D เช่น (nrx, nt, ntr) หรือ (nt, ntr, 1)
    if data.ndim == 3:
        # ถ้ามิติแรกเล็ก -> น่าจะเป็น nrx
        if data.shape[0] < 10:
            data2 = data[0, :, :]
        else:
            # ถ้ามิติสุดท้ายเล็ก -> เอา channel แรก
            data2 = data[:, :, 0]
        if data2.ndim != 2:
            data2 = data2.reshape(data2.shape[0], -1)
        return data2, ds_name

    # fallback
    data2 = data.reshape(data.shape[0], -1)
    return data2, ds_name


def clip_norm_signed(A: np.ndarray, p: float) -> np.ndarray:
    """
    normalize แบบมีเครื่องหมาย (-1..1) สำหรับ B-scan raw
    """
    s = np.percentile(np.abs(A), p)
    if s <= 0:
        return A
    return np.clip(A, -s, s) / s


def clip_norm_positive(A: np.ndarray, p: float) -> np.ndarray:
    """
    normalize แบบ 0..1 สำหรับ envelope / magnitude
    """
    s = np.percentile(A, p)
    if s <= 0:
        return A
    return np.clip(A, 0, s) / s


def hilbert_envelope(x2d: np.ndarray) -> np.ndarray:
    """
    Hilbert envelope แบบไม่พึ่ง scipy
    ทำ analytic signal ด้วย FFT ตามแกน time (axis=0)
    """
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
    """
    ลบ mean trace ตามแกน x/y (ลดแถบแนวนอน/พื้นหลังคงที่)
    cube shape: (time, ny, nx)
    """
    mean_xy = cube.mean(axis=(1, 2), keepdims=True)  # mean ต่อ time
    return cube - mean_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_out")

    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)

    ap.add_argument("--comp", default="Ez")
    ap.add_argument("--line_y", type=int, default=2)       # B-scan ที่ y index
    ap.add_argument("--x_cross", type=int, default=None)   # cross-section ที่ x index
    ap.add_argument("--t_slice", type=int, default=None)   # C-scan ที่ time index (หลัง crop)

    ap.add_argument("--t0", type=int, default=0)           # crop start (sample index)
    ap.add_argument("--t1", type=int, default=-1)          # crop end (-1 = ถึงท้าย)

    ap.add_argument("--clip", type=float, default=99.0)    # percentile clip
    ap.add_argument("--dpi", type=int, default=220)

    # --- display options ---
    ap.add_argument("--invert", action="store_true", help="สลับขาวดำ (gray_r)")
    ap.add_argument("--env", action="store_true", help="ทำ envelope (เหมือน magnitude)")
    ap.add_argument("--bgremove", action="store_true", help="ลบ background (ลดแถบแนวนอน)")

    args = ap.parse_args()

    data, ds_name = load_data(args.merged_out, args.comp)
    nt, ntr = data.shape

    expected = args.nx * args.ny
    if ntr != expected:
        raise ValueError(f"traces={ntr} แต่ nx*ny={expected} (เช็ก --nx/--ny). dataset={ds_name}")

    # crop time
    t0 = max(0, args.t0)
    t1 = nt if args.t1 == -1 else min(nt, args.t1)
    data = data[t0:t1, :]
    nt2 = data.shape[0]

    # reshape to cube: (time, y, x)
    cube = data.reshape(nt2, args.ny, args.nx)

    # optional processing
    if args.bgremove:
        cube = background_remove(cube)

    y_idx = int(np.clip(args.line_y, 0, args.ny - 1))
    x_cross = args.nx // 2 if args.x_cross is None else int(np.clip(args.x_cross, 0, args.nx - 1))
    t_slice = nt2 // 2 if args.t_slice is None else int(np.clip(args.t_slice, 0, nt2 - 1))

    # slices
    bscan = cube[:, y_idx, :]         # (time, x)
    cross = cube[:, :, x_cross]       # (time, y)
    cscan = cube[t_slice, :, :]       # (y, x)

    # envelope option
    if args.env:
        bscan = hilbert_envelope(bscan)
        cross = hilbert_envelope(cross)
        cscan = np.abs(cscan)  # C-scan เป็น 2D ณ time เดียว ใช้ abs

        bscan = clip_norm_positive(bscan, args.clip)
        cross = clip_norm_positive(cross, args.clip)
        cscan = clip_norm_positive(cscan, args.clip)

        vmin, vmax = 0.0, 1.0
    else:
        bscan = clip_norm_signed(bscan, args.clip)
        cross = clip_norm_signed(cross, args.clip)
        cscan = clip_norm_signed(cscan, args.clip)

        vmin, vmax = -1.0, 1.0

    cmap = "gray_r" if args.invert else "gray"

    # plot 3 views
    fig = plt.figure(figsize=(13.5, 7.5))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)

    ax1.imshow(bscan, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"Longitudinal section (B-scan) | {args.comp} | y={y_idx} | dataset={ds_name}")
    ax1.set_xlabel("x trace index")
    ax1.set_ylabel("time sample index")

    ax2.imshow(cross, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Cross section (B-scan) | x={x_cross}")
    ax2.set_xlabel("y line index")
    ax2.set_ylabel("time sample index")

    ax3.imshow(cscan, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.set_title(f"Horizontal section (C-scan top view) | t={t_slice} (crop {t0}:{t1})")
    ax3.set_xlabel("x trace index")
    ax3.set_ylabel("y line index")

    out_png = args.merged_out.replace(
        ".out",
        f"_3views_{args.comp}_y{y_idx}_x{x_cross}_t{t_slice}_t{t0}-{t1}_gray.png"
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi)
    print("Saved:", out_png)
    plt.show()


if __name__ == "__main__":
    main()
