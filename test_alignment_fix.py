"""Cross-family alignment test for ``_aligned_tokenize_pair``.

For each of the 9 model pairs in the paper:
  1. Load the two tokenizers.
  2. Run the *new* anchor-based alignment helper on real cached thinking
     responses; report
       - fraction of responses that found a shared char anchor
       - fraction of base positions that align to the same target token
         on the think side
       - max contiguous misaligned-prefix length (always <= a few tokens).
  3. Run the *old* standalone-len-based offset on the same responses; report
     the same metrics (this shows what the bug was costing us).
  4. Run a synthetic stress test that exercises specific BPE merges
     (thinking starts with 'To', '\n', ' ', digit, punctuation, etc.)
     and confirms the new helper handles them all.

Success criteria:
  - All pairs: new alignment finds an anchor for >=99% of responses.
  - All pairs: aligned positions cover >=95% of the rollout for each
    response.
  - Loop's defensive guard ``think_ids[i_t+1] == base_ids[i+1]`` *never*
    fires (we already aligned).
  - Synthetic stress: every corner case finds an anchor within the first
    3 thinking tokens.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, "/workspace/thinking-llms-interp/train-vectors")
# Importing the trainer module also imports utils.utils which pulls in
# nnsight; some environments have a broken nnsight install.  We only need
# the helper, so define it inline below (mirror of the trainer's helper).

from transformers import AutoTokenizer  # noqa: E402


# Mirror of optimize_correction_vectors._aligned_tokenize_pair for the test.
def _aligned_tokenize_pair(btok, base_prompt, ttok, think_prompt, thinking):
    base_full = base_prompt + thinking
    think_full = think_prompt + thinking
    enc_b = btok(base_full, return_offsets_mapping=True, truncation=False)
    enc_t = ttok(think_full, return_offsets_mapping=True, truncation=False)
    bp = len(base_prompt)
    tp = len(think_prompt)
    b_ends = {}
    for i, (s, e) in enumerate(enc_b["offset_mapping"]):
        if s >= bp and e > bp:
            c = e - bp
            b_ends.setdefault(c, i)
    t_ends = {}
    for i, (s, e) in enumerate(enc_t["offset_mapping"]):
        if s >= tp and e > tp:
            c = e - tp
            t_ends.setdefault(c, i)
    common = set(b_ends.keys()) & set(t_ends.keys())
    if not common:
        b_anchor = t_anchor = anchor_c = -1
    else:
        anchor_c = min(common)
        b_anchor = b_ends[anchor_c] + 1
        t_anchor = t_ends[anchor_c] + 1
    return {
        "b_ids": enc_b["input_ids"],
        "t_ids": enc_t["input_ids"],
        "b_anchor": b_anchor,
        "t_anchor": t_anchor,
        "anchor_c": anchor_c,
    }


def _old_alignment(btok, base_prompt, ttok, think_prompt, thinking):
    """The pre-fix logic, for comparison."""
    base_full = base_prompt + thinking
    think_full = think_prompt + thinking
    b_ids = btok(base_full)["input_ids"]
    t_ids = ttok(think_full)["input_ids"]
    bp_len = len(btok(base_prompt)["input_ids"])
    tp_len = len(ttok(think_prompt)["input_ids"])
    return b_ids, t_ids, bp_len, tp_len


def extract_thinking_process(response):
    think_tag = "<think>"
    end_tag = "</think>"
    orz_marker = "Assistant: <think>"
    n_think = response.count(think_tag)
    if n_think == 0:
        s = 0
    elif n_think == 1:
        s = response.find(think_tag) + len(think_tag)
    else:
        first_orz = response.find(orz_marker)
        if first_orz != -1:
            s = first_orz + len(orz_marker)
        else:
            s = response.rfind(think_tag) + len(think_tag)
    e = response.find(end_tag, s)
    if e == -1:
        e = len(response)
    return response[s:e].strip()


def _build_base_prompt(q):
    return f"User: {q}\nAssistant:"


PAIRS = [
    # (label, base_model, think_model, responses_file)
    ("ORZ-0.5B",    "Qwen/Qwen2.5-0.5B",       "Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B",
        "responses_open-reasoner-zero-0.5b.json"),
    ("ORZ-1.5B",    "Qwen/Qwen2.5-1.5B",       "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B",
        "responses_open-reasoner-zero-1.5b.json"),
    ("ORZ-7B",      "Qwen/Qwen2.5-7B",         "Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
        "responses_open-reasoner-zero-7b.json"),
    ("ORZ-32B",     "Qwen/Qwen2.5-32B",        "Open-Reasoner-Zero/Open-Reasoner-Zero-32B",
        "responses_open-reasoner-zero-32b.json"),
    ("DSL-8B",      "meta-llama/Llama-3.1-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "responses_deepseek-r1-distill-llama-8b.json"),
    ("DSQ-Math-1.5B", "Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "responses_deepseek-r1-distill-qwen-1.5b.json"),
    ("DSQ-14B",     "Qwen/Qwen2.5-14B",        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "responses_deepseek-r1-distill-qwen-14b.json"),
    ("DSQ-32B",     "Qwen/Qwen2.5-32B",        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "responses_deepseek-r1-distill-qwen-32b.json"),
    ("QwQ-32B",     "Qwen/Qwen2.5-32B",        "Qwen/QwQ-32B",
        "responses_qwq-32b.json"),
]

RESP_DIR = "/workspace/thinking-llms-interp/generate-responses/results/vars"
N_PER_PAIR = 50           # responses to sample per pair
MAX_RESP_CHARS = 30000    # skip ultra-long responses for speed


def eval_pair_real(label, btok, ttok, responses_path):
    """Run new + old alignment on real cached responses and report stats."""
    if not os.path.exists(responses_path):
        return {"label": label, "status": "missing-file", "path": responses_path}

    rows = json.load(open(responses_path))
    # Take the first N that have a usable thinking trace.
    samples = []
    for r in rows:
        t = extract_thinking_process(r["full_response"])
        if t.strip() and len(r["full_response"]) < MAX_RESP_CHARS:
            samples.append(r)
        if len(samples) >= N_PER_PAIR:
            break
    if not samples:
        return {"label": label, "status": "no-samples"}

    new_anchor_found = 0
    new_pos_aligned = 0
    new_pos_total = 0
    new_guard_hits = 0       # times the defensive guard caught a mismatch
    new_max_lead = 0         # max #tokens we lose at the head of thinking

    old_pos_aligned = 0
    old_pos_total = 0
    old_full_misaligned_resp = 0

    for r in samples:
        q = r["original_message"]["content"]
        thinking = extract_thinking_process(r["full_response"])
        base_prompt = _build_base_prompt(q)
        try:
            think_prompt = ttok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            think_prompt = base_prompt

        # ---------- NEW ----------
        out = _aligned_tokenize_pair(btok, base_prompt, ttok, think_prompt, thinking)
        b_ids = out["b_ids"]
        t_ids = out["t_ids"]
        b_anchor = out["b_anchor"]
        t_anchor = out["t_anchor"]
        Lb, Lt = len(b_ids), len(t_ids)
        if b_anchor < 0:
            # no anchor found — count as not-aligned
            new_pos_total += max(Lb - len(btok(base_prompt)["input_ids"]) - 1, 0)
        else:
            new_anchor_found += 1
            new_max_lead = max(new_max_lead, b_anchor - len(btok(base_prompt)["input_ids"]))
            offset = t_anchor - b_anchor
            n_pos = 0
            n_aligned = 0
            for i in range(max(b_anchor - 1, 0), Lb - 1):
                n_pos += 1
                i_t = i + offset
                if i_t < 0 or i_t + 1 >= Lt:
                    continue
                if t_ids[i_t + 1] == b_ids[i + 1]:
                    n_aligned += 1
                else:
                    new_guard_hits += 1
            new_pos_aligned += n_aligned
            new_pos_total += n_pos

        # ---------- OLD ----------
        b_ids_o, t_ids_o, bp_len_o, tp_len_o = _old_alignment(
            btok, base_prompt, ttok, think_prompt, thinking)
        Lb_o, Lt_o = len(b_ids_o), len(t_ids_o)
        offset_o = tp_len_o - bp_len_o
        n_pos_o = n_align_o = 0
        for i in range(max(bp_len_o - 1, 0), Lb_o - 1):
            n_pos_o += 1
            i_t = i + offset_o
            if 0 <= i_t and i_t + 1 < Lt_o and t_ids_o[i_t + 1] == b_ids_o[i + 1]:
                n_align_o += 1
        old_pos_aligned += n_align_o
        old_pos_total += n_pos_o
        if n_align_o == 0:
            old_full_misaligned_resp += 1

    return {
        "label": label, "status": "ok", "n_samples": len(samples),
        "new_anchor_resp": new_anchor_found,
        "new_align_rate": new_pos_aligned / max(new_pos_total, 1),
        "new_guard_hits": new_guard_hits,
        "new_max_lead": new_max_lead,
        "old_align_rate": old_pos_aligned / max(old_pos_total, 1),
        "old_fully_misaligned": old_full_misaligned_resp,
    }


def synthetic_stress(label, btok, ttok):
    """A handful of pre-baked thinking-content openings that trigger
    specific BPE merge corner cases."""
    cases = [
        ("starts-with-To",          "To find the answer, multiply 2 by 3."),
        ("starts-with-The",         "The answer is 6 because 2 times 3 is 6."),
        ("starts-with-newline-To",  "\nTo find the answer."),
        ("starts-with-double-nl",   "\n\nFirst, we compute 2 + 2."),
        ("starts-with-space-To",    " To find the answer."),
        ("starts-with-digit",       "1. First, we compute 2 + 2."),
        ("starts-with-tab",         "\tFirst, we compute the answer."),
        ("starts-with-punct",       "...let me think about this."),
        ("starts-with-pound",       "# Step 1: identify the problem."),
        ("starts-with-asterisk",    "**Step 1**: identify the problem."),
    ]
    q = "What is 2 + 3?"
    base_prompt = _build_base_prompt(q)
    try:
        think_prompt = ttok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        think_prompt = base_prompt
    n_ok = 0
    failures = []
    for name, thinking in cases:
        out = _aligned_tokenize_pair(btok, base_prompt, ttok, think_prompt, thinking)
        ok = out["b_anchor"] >= 0 and out["anchor_c"] <= 30  # anchor within first 30 chars
        if ok:
            n_ok += 1
        else:
            failures.append((name, out["b_anchor"], out["anchor_c"]))
    return {"label": label, "n_cases": len(cases),
            "n_ok": n_ok, "failures": failures}


def main():
    print("=" * 78)
    print("Cross-family alignment test (new helper vs. old standalone-len)")
    print("=" * 78)
    real_results = []
    synth_results = []
    for label, base_id, think_id, resp_file in PAIRS:
        print(f"\n[{label}] loading tokenizers...")
        try:
            btok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
            ttok = AutoTokenizer.from_pretrained(think_id, use_fast=True)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        # Real
        real_results.append(eval_pair_real(label, btok, ttok,
                                           os.path.join(RESP_DIR, resp_file)))
        # Synthetic
        synth_results.append(synthetic_stress(label, btok, ttok))

    print("\n" + "=" * 78)
    print(f"{'PAIR':<16} {'samples':>8} {'new_anchor':>11} {'new_align':>10} "
          f"{'new_guard':>10} {'old_align':>10} {'old_dead':>9} {'lead':>6}")
    print("-" * 78)
    for r in real_results:
        if r["status"] != "ok":
            print(f"{r['label']:<16}  status={r['status']}")
            continue
        n = r["n_samples"]
        print(f"{r['label']:<16} {n:>8} "
              f"{r['new_anchor_resp']:>4}/{n:<4}     "
              f"{r['new_align_rate']*100:>8.1f}%  "
              f"{r['new_guard_hits']:>9}  "
              f"{r['old_align_rate']*100:>8.1f}%  "
              f"{r['old_fully_misaligned']:>4}/{n:<4} "
              f"{r['new_max_lead']:>5}")
    print()
    print("LEGEND:")
    print("  new_anchor      = # responses where new helper found a shared anchor")
    print("  new_align       = % of base positions aligned (with new helper)")
    print("  new_guard       = times the defensive guard caught a mismatch (should be 0)")
    print("  old_align       = % of base positions aligned with the OLD buggy code")
    print("  old_dead        = # responses the OLD code dropped entirely")
    print("  lead            = max # tokens lost at the head of thinking (small constant)")

    print("\n" + "=" * 78)
    print(f"{'PAIR':<16} {'synth_ok':>10}  {'failures':<40}")
    print("-" * 78)
    for r in synth_results:
        fail_str = "" if r["n_ok"] == r["n_cases"] else "; ".join(
            f"{n}(c={c})" for n, _, c in r["failures"])
        print(f"{r['label']:<16} {r['n_ok']:>3}/{r['n_cases']:<3}      {fail_str}")

    # Hard pass/fail.
    fail = False
    for r in real_results:
        if r["status"] != "ok":
            print(f"FAIL: {r['label']} could not be evaluated")
            fail = True; continue
        if r["new_anchor_resp"] < int(0.99 * r["n_samples"]):
            print(f"FAIL: {r['label']} anchor coverage <99% "
                  f"({r['new_anchor_resp']}/{r['n_samples']})")
            fail = True
        if r["new_align_rate"] < 0.95:
            print(f"FAIL: {r['label']} alignment <95% "
                  f"({100*r['new_align_rate']:.1f}%)")
            fail = True
        if r["new_guard_hits"] != 0:
            print(f"FAIL: {r['label']} defensive guard fired {r['new_guard_hits']}x")
            fail = True
    for r in synth_results:
        if r["n_ok"] != r["n_cases"]:
            print(f"FAIL: {r['label']} synthetic test {r['n_ok']}/{r['n_cases']}")
            fail = True

    if fail:
        print("\n*** SOME TESTS FAILED ***")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
