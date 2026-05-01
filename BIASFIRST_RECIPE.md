# Bias-first correction-vector recipe

A 3-stage pipeline that trains a single global bias vector first, then
trains per-category correction vectors as residuals on top of the frozen
bias.  At inference the steering hook applies `coef * (cat_vec[k] +
bias_vec)` at the steering layer for every disagreement-gated position.
The reference (ORZ-7B / Qwen2.5-7B, layer 16, math500) recovers ~107% of
the base-vs-thinking gap on 128 tasks under a perplexity-guardrail sweep
of [0.5, 1.0, 1.5].

## Stages

Stage 1 (`run_*_biasfirst_stage1.sh`):
1. Reuse the no-bias `disagreements.pt` (or collect fresh, see Stage 0).
2. Run `optimize_correction_vectors.py` with `--skip_cats_phase
   --train_global_bias`.  Output: `<save_dir>/<base>_bias_global.pt`
   plus `bias_layer.json`.

Stage 2 (`run_*_biasfirst_stage2.sh`):
- 2a (collect):  re-run `optimize_correction_vectors.py --collect_only`
  with `--frozen_bias_path` so the base model has the Stage-1 bias hooked
  in at the steering layer.  Disagreements collected this way are the
  positions where base+bias still disagrees with thinking.
- 2b (train cats):  run `optimize_correction_vectors.py
  --frozen_bias_path --load_collected`.  Per-category vectors are trained
  as the *residual* on top of the frozen bias.

Stage 0 (`run_dsqwen32b_biasfirst_stage0.sh`):  initial no-bias
disagreement collection for pairs that don't already have one.

## Hyperparameters (matched across pairs)

```
--steer_layer        layer for both bias and cat vectors
                       (ORZ-7B: 16, Llama-8B: 19, 32B: 38 = N/2 + N/10)
--topk 50            keep top-50 token logits at every retained position
--train_topk 3       KL targets are over thinking model's top-3 tokens
--kl_mode topk       KL-top3 objective
--max_seq_len 2048
--max_positions_per_example 64
--n_epochs 5
--example_batch_size 16
--lr 1e-2
--weight_decay 0.0
--max_norm 0.0
--seed 42
--holdout_frac 0.1
--min_disagreements 1
--min_disagreements_ratio 0.0
--min_category_share 0.0
--n_responses 20000   (stage-2a re-collect; stage-1 re-uses any prior dump)
```

The ORZ-7B 7B reference, the Llama-8B run and the 32B QwQ/DSQwen runs
were all trained with this recipe.  The 32B runs in this repo's
checkpoint history were trained with `n_epochs=2 example_batch_size=4
n_responses=2000` because a 32B model + bs=16 OOMs on a single H200; on
hardware that fits the canonical schedule, switch the values back to
match.

## Eval

`hybrid/run_*_biasfirst_eval.sh` runs three conditions on math500 tasks:

- `learn`     full bias-first model: `coef * (cat_vec[k] + bias_vec)`
- `rand`      per-category vectors replaced with norm-preserving random
              directions (bias kept) -- isolates the *direction*
              contribution
- `biasonly`  per-category vectors zeroed, only the global bias is
              applied -- isolates whether categories add anything

The canonical perplexity-guardrail sweep is `[0.5, 1.0, 1.5]`.  The
DeepSeek-R1-Distill-Llama-8B pair degrades under this sweep and is run
at the conservative `[0.1, 0.25, 0.5]` instead -- see comment in
`run_dsllama8b_biasfirst_eval.sh`.

## Observed gap recovery (128 math500 tasks, learn condition)

| Pair                                            | sweep              | gap recovered |
| ----------------------------------------------- | ------------------ | ------------- |
| Open-Reasoner-Zero-7B / Qwen2.5-7B              | [0.5, 1.0, 1.5]    | ~107%         |
| DeepSeek-R1-Distill-Llama-8B / Llama-3.1-8B     | [0.1, 0.25, 0.5]   | ~6.5%         |
| QwQ-32B / Qwen2.5-32B (with `n_epochs=2 bs=4`)  | [0.1 .. 1.0]       | +10.5%        |
| QwQ-32B / Qwen2.5-32B (with `n_epochs=2 bs=4`)  | [0.5, 1.0, 1.5]    | -54%          |
| DeepSeek-R1-Distill-Qwen-32B / Qwen2.5-32B      | not yet run with full recipe |     |

The 32B numbers above were measured with the reduced training schedule
and are expected to improve with the canonical 5-epoch / bs=16 / 20k
re-collect schedule on hardware that supports it.
