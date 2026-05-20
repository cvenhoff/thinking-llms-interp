#!/usr/bin/env python3
"""
Write layer_map.json into the vectors save directory.

Usage:
    python write_layer_map.py --save_dir PATH --layer L --n_cats N [--model_short MODEL]

Creates: save_dir/layer_map_<model_short>.json  {"idx0": L, ...}
When --model_short is omitted, also writes the legacy save_dir/layer_map.json.
"""
import argparse, json, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n_cats", type=int, required=True)
    p.add_argument("--model_short", default=None,
                   help="Short model name, e.g. qwen2.5-1.5b. "
                        "Writes layer_map_<model_short>.json (model-specific, no conflicts).")
    args = p.parse_args()

    layer_map = {f"idx{i}": args.layer for i in range(args.n_cats)}
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_short:
        # Model-specific file — safe when multiple models share the same save_dir
        out = os.path.join(args.save_dir, f"layer_map_{args.model_short}.json")
    else:
        out = os.path.join(args.save_dir, "layer_map.json")

    with open(out, "w") as f:
        json.dump(layer_map, f, indent=2)
    print(f"Wrote layer_map.json -> {out}: {layer_map}")

if __name__ == "__main__":
    main()
