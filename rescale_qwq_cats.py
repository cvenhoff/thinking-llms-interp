"""Post-hoc rescale QwQ-32B cat vectors to a target L2 norm.

Loads each saved cat vector from src_dir, normalises to target_norm, and
saves to dst_dir under the same filenames so hybrid_eval can drop-in
swap via --old_vectors_dir.
"""
import argparse
import os
import shutil

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--dst_dir", required=True)
    p.add_argument("--target_norm", type=float, default=10.0)
    p.add_argument("--model_short", type=str, default="qwen2.5-32b")
    args = p.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    for fn in sorted(os.listdir(args.src_dir)):
        src = os.path.join(args.src_dir, fn)
        dst = os.path.join(args.dst_dir, fn)
        if not fn.endswith(".pt"):
            shutil.copy2(src, dst)
            print(f"  [copy] {fn}")
            continue
        if "_idx" not in fn or "_linear.pt" not in fn:
            shutil.copy2(src, dst)
            print(f"  [copy] {fn}")
            continue
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        # Find the vector tensor.  Format historically:
        # ckpt = {"v": tensor, ...} or just the tensor itself.
        if isinstance(ckpt, dict):
            # Trainer saves cats as {"idxN": tensor} for layered files.
            tensor_keys = [k for k, v in ckpt.items()
                           if isinstance(v, torch.Tensor) and v.ndim == 1]
            if tensor_keys:
                touched = []
                for k in tensor_keys:
                    v = ckpt[k]
                    n0 = float(v.norm())
                    if n0 < 1e-8:
                        new_v = v.clone()
                    else:
                        new_v = (v.float() / n0 * args.target_norm).to(v.dtype)
                    ckpt[k] = new_v
                    touched.append(f"{k}: {n0:.2f}->{float(new_v.norm()):.2f}")
                print(f"  [rescale] {fn}: " + ", ".join(touched))
            else:
                print(f"  [skip] {fn}: no rescalable tensor; keys={list(ckpt.keys())}")
                shutil.copy2(src, dst)
                continue
        elif isinstance(ckpt, torch.Tensor):
            v = ckpt
            n0 = float(v.norm())
            if n0 < 1e-8:
                new_v = v.clone()
            else:
                new_v = (v.float() / n0 * args.target_norm).to(v.dtype)
            ckpt = new_v
            n1 = float(new_v.norm())
            print(f"  [rescale] {fn}: norm {n0:.2f} -> {n1:.2f}")
        else:
            print(f"  [skip] {fn}: unsupported type {type(ckpt)}")
            shutil.copy2(src, dst)
            continue
        torch.save(ckpt, dst)


if __name__ == "__main__":
    main()
