"""Extract per-category vectors from the trainer's best.pt snapshot
into the per-cat layered format expected by hybrid_eval.py."""
import argparse
import json
import os
import shutil

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--dst_dir", required=True)
    p.add_argument("--model_short", type=str, default="qwen2.5-32b")
    args = p.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    best_path = os.path.join(args.src_dir, f"{args.model_short}_cats_seed42_best.pt")
    if not os.path.exists(best_path):
        # Find any cats_*_best.pt
        cands = [f for f in os.listdir(args.src_dir) if f.endswith("_best.pt")]
        if not cands:
            raise FileNotFoundError(f"No best.pt found in {args.src_dir}")
        best_path = os.path.join(args.src_dir, cands[0])
        print(f"  using {cands[0]}")

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    V = ckpt["V"]  # (n_cats, hidden)
    print(f"  V shape: {tuple(V.shape)}, norms: " + ", ".join(f"{float(V[i].norm()):.2f}" for i in range(V.shape[0])))
    print(f"  epoch: {ckpt.get('epoch')}, holdout_kl: {ckpt.get('holdout_kl'):.4f}")

    n_cats = V.shape[0]
    # Copy ancillary files unchanged.
    for fn in os.listdir(args.src_dir):
        if fn.endswith(".pt") and "_idx" in fn:
            continue  # skip; we'll write fresh
        src = os.path.join(args.src_dir, fn)
        if os.path.isfile(src):
            dst = os.path.join(args.dst_dir, fn)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  [copy] {fn}")

    # Write per-cat files.
    for i in range(n_cats):
        v = V[i].clone()
        out = {f"idx{i}": v}
        out_path = os.path.join(args.dst_dir, f"{args.model_short}_idx{i}_linear.pt")
        torch.save(out, out_path)
        print(f"  [write] {os.path.basename(out_path)}: norm={float(v.norm()):.2f}")

    # Ensure layer_map.json is present (steer-layer is uniform for cats).
    lm_src = os.path.join(args.src_dir, "layer_map.json")
    lm_dst = os.path.join(args.dst_dir, "layer_map.json")
    if not os.path.exists(lm_dst):
        # Default layer_map: every cat goes to the same steer_layer.  Fall
        # back to copying the source dir's layer_map if available.
        if os.path.exists(lm_src):
            shutil.copy2(lm_src, lm_dst)
        else:
            print("  WARN: layer_map.json missing; hybrid_eval may fail.")

    print(f"Done. Output dir: {args.dst_dir}")


if __name__ == "__main__":
    main()
