#!/usr/bin/env python3
"""
Write layer_map.json into the vectors save directory.

Usage:
    python write_layer_map.py --save_dir PATH --layer L --n_cats N [--model_short S]

Creates: save_dir/layer_map.json  {"idx0": L, "idx1": L, ...}
"""
import argparse, json, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n_cats", type=int, required=True)
    args = p.parse_args()

    layer_map = {f"idx{i}": args.layer for i in range(args.n_cats)}
    out = os.path.join(args.save_dir, "layer_map.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(layer_map, f, indent=2)
    print(f"Wrote layer_map.json -> {out}: {layer_map}")

if __name__ == "__main__":
    main()
