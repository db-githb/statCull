import argparse
from pathlib import Path
from src.core import run_cull

def main():
    p = argparse.ArgumentParser("Cull 3DGS gaussians and export a PLY.")
    p.add_argument("--checkpoint", "-ckpt", type=Path)
    p.add_argument("--out", "-o", type=Path, default=None)
    p.add_argument("--threshold-x", "-thr-x", type=float, default=0.2)
    p.add_argument("--threshold-y", "-thr-y", type=float, default=0.2)
    p.add_argument("--threshold-z", "-thr-z", type=float, default=1.0)
    args = p.parse_args()

    run_cull(args.checkpoint, args.out, thr_xyz=[args.threshold_x, args.threshold_y, args.threshold_z])
    #print(f"Culled {stats['culled']}/{stats['total']} → {args.out}")

if __name__ == "__main__":
    main()