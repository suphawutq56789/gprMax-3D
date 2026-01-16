import argparse
import numpy as np
import matplotlib.pyplot as plt
from gprMax.outputfiles import read_outputfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_out")
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--comp", default="Ez")
    ap.add_argument("--line_y", type=int, default=2)    # B-scan ที่ y index ไหน
    ap.add_argument("--x_cross", type=int, default=None) # cross-section ที่ x index ไหน
    ap.add_argument("--t_slice", type=int, default=None) # C-scan ที่ time index ไหน
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=-1)
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    data, dt = read_outputfile(args.merged_out, args.comp)  # data: (nt, ntraces)
    nt, ntr = data.shape
    if ntr != args.nx * args.ny:
        raise ValueError(f"ntraces={ntr} แต่ nx*ny={args.nx*args.ny} (เช็ก --nx/--ny)")

    t0 = max(0, args.t0)
    t1 = nt if args.t1 == -1 else min(nt, args.t1)
    data = data[t0:t1, :]
    nt2 = data.shape[0]

    cube = data.reshape(nt2, args.ny, args.nx)  # (time, y, x)

    y_idx = int(np.clip(args.line_y, 0, args.ny - 1))
    x_cross = args.nx // 2 if args.x_cross is None else int(np.clip(args.x_cross, 0, args.nx - 1))
    t_slice = nt2 // 2 if args.t_slice is None else int(np.clip(args.t_slice, 0, nt2 - 1))

    bscan = cube[:, y_idx, :]          # (time, x)
    cross = cube[:, :, x_cross]        # (time, y)
    cscan = cube[t_slice, :, :]        # (y, x)

    # clip normalize (ทำให้คอนทราสต์ดูดีขึ้น)
    def norm(A, p=99.0):
        s = np.percentile(np.abs(A), p)
        if s <= 0:
            return A
        return np.clip(A, -s, s) / s

    bscan = norm(bscan, args.clip)
    cross = norm(cross, args.clip)
    cscan = norm(cscan, args.clip)

    fig = plt.figure(figsize=(13.5, 7.5))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)

    ax1.imshow(bscan, aspect="auto")
    ax1.set_title(f"Longitudinal section (B-scan) | comp={args.comp} | y={y_idx} | t={t0}:{t1}")
    ax1.set_xlabel("x trace index")
    ax1.set_ylabel("time sample index")

    ax2.imshow(cross, aspect="auto")
    ax2.set_title(f"Cross section (B-scan) | x={x_cross}")
    ax2.set_xlabel("y line index")
    ax2.set_ylabel("time sample index")

    ax3.imshow(cscan, aspect="auto")
    ax3.set_title(f"Horizontal section (C-scan) | t={t_slice}")
    ax3.set_xlabel("x trace index")
    ax3.set_ylabel("y line index")

    out_png = args.merged_out.replace(".out", f"_3views_{args.comp}_y{y_idx}_x{x_cross}_t{t_slice}_t{t0}-{t1}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi)
    print("Saved:", out_png)
    plt.show()

if __name__ == "__main__":
    main()
