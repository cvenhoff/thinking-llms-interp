"""Compare cat vectors trained under different conditions."""
import os
import sys
import json
import torch

ROOT = "/workspace/thinking-llms-interp/train-vectors/results/vars"

S1 = "correction_vectors_orz7b_biasfirst_stage1_canon"
S1_SAN = "correction_vectors_orz7b_biasfirst_stage1_sanity"

S2 = {
    "canon (3-GPU PP)": "correction_vectors_orz7b_biasfirst_stage2_canon",
    "singlegpu_diag":  "correction_vectors_orz7b_biasfirst_stage2_singlegpu_diag",
    "b078d01_trainer": "correction_vectors_orz7b_biasfirst_stage2_b078d01",
    "sanity":          "correction_vectors_orz7b_biasfirst_stage2_sanity",
}


def _load_bias(s1_dir):
    p = os.path.join(ROOT, s1_dir, "qwen2.5-7b_bias_global.pt")
    if not os.path.exists(p):
        p = os.path.join(ROOT, s1_dir, "qwen2.5-7b_bias_best.pt")
    if not os.path.exists(p):
        return None
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("bias", "vector", "v", "b"):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k].float()
    elif torch.is_tensor(obj):
        return obj.float()
    return None


def _load_cats(d):
    out = {}
    for i in range(10):
        p = os.path.join(ROOT, d, f"qwen2.5-7b_idx{i}_linear.pt")
        if not os.path.exists(p):
            continue
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            v = None
            # try idx-keyed first
            for k in (f"idx{i}", "vector", "v", "V"):
                if k in obj and torch.is_tensor(obj[k]):
                    v = obj[k].float()
                    break
            if v is None:
                # fallback: take any tensor in dict
                for k, val in obj.items():
                    if torch.is_tensor(val):
                        v = val.float()
                        break
            if v is None:
                continue
        elif torch.is_tensor(obj):
            v = obj.float()
        else:
            continue
        out[i] = v
    return out


def _cos(a, b):
    na = a.flatten().float()
    nb = b.flatten().float()
    return float(torch.dot(na, nb) / (na.norm() * nb.norm() + 1e-12))


# Bias from canon (used by canon, singlegpu, b078d01, all share same disagreements)
b_canon = _load_bias(S1)
b_san = _load_bias(S1_SAN)
print("bias norms: canon=", float(b_canon.norm()),
      "  sanity=", float(b_san.norm() if b_san is not None else float('nan')))
print("cos(bias_canon, bias_sanity)=",
      _cos(b_canon, b_san) if b_san is not None else "n/a")

cats = {name: _load_cats(d) for name, d in S2.items()}

print("\n=== cosine of cat vectors with bias_canon (or bias_sanity for sanity) ===")
for name, vs in cats.items():
    bias = b_san if name == "sanity" else b_canon
    print(f"\n[{name}] avg_cos_to_bias and per-cat:")
    cs = []
    for i in range(10):
        if i not in vs:
            continue
        c = _cos(vs[i], bias)
        cs.append(c)
        print(f"  idx{i}: norm={float(vs[i].norm()):.3f}  cos(bias)={c:+.3f}")
    if cs:
        import statistics
        print(f"  AVG cos(bias) = {statistics.mean(cs):+.3f}")

# Cross-table: compare canon vs each other
print("\n=== pairwise cosine between SAME idx, across runs ===")
ref = cats["canon (3-GPU PP)"]
for name, vs in cats.items():
    if name == "canon (3-GPU PP)":
        continue
    print(f"\n[canon vs {name}]")
    for i in range(10):
        if i not in vs or i not in ref:
            continue
        c = _cos(ref[i], vs[i])
        print(f"  idx{i}: cos={c:+.4f}  norms canon={float(ref[i].norm()):.2f} other={float(vs[i].norm()):.2f}")
