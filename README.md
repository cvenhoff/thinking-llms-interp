# Base Models Know How to Reason, Thinking Models Learn When

Code for the paper [Base Models Know How to Reason, Thinking Models Learn When](https://arxiv.org/abs/2510.07364) (ICML 2026).

This branch reproduces the **category-vector hybrid-steering** results: we discover
per-model reasoning taxonomies with SAEs, train a small MLP that decides which
taxonomy direction to steer with and where, and build hybrid models (a base model
plus the learned steering) that we compare against the matching thinking model.

## Setup

Requires Python 3.10+, [`uv`](https://docs.astral.sh/uv/), and GPUs (the 32B pairs
need 2×80 GB).

```bash
git clone https://github.com/cvenhoff/cot-interp.git
cd cot-interp
uv sync                                       # installs everything into .venv
cp .env_exports.sh.example .env_exports.sh    # then edit paths (HF cache, etc.)
echo "ANTHROPIC_API_KEY=sk-..." > .env        # LLM judge credentials
```

The rollout scripts activate `.venv_vllm` if you keep vLLM in a separate
environment, and otherwise fall back to `.venv`.

## Repository layout

| Path | Purpose |
|------|---------|
| `configs.sh`          | The nine model pairs and their SAE/steering settings (one source of truth). |
| `generate-responses/` | Generate and annotate thinking-model responses (SAE inputs). |
| `train-saes/`         | Train SAEs and build the reasoning taxonomies. |
| `vllm-serve/`         | Rollout generation (`gen_think.sh`, `gen_base.sh`, `gen_hendrycks.sh`, `run_rollouts.sh`). |
| `train-vectors/`      | Category-vector training (`train_vectors.sh`, `run.sh`). |
| `hybrid/`             | Best-of-3 selection and hybrid evaluation (`select_best_of_3.sh`, `eval_*.sh`, `run.sh`, `run_ablations.sh`). |
| `figures/`            | Scripts that render every paper figure and table into `figures/figs/`. |
| `data/`               | Training mix and held-out evaluation sets. |
| `artifacts/`          | All pipeline outputs: selected vectors and eval results (committed) plus regenerable per-run training outputs (git-ignored). |

The nine pairs are `orz-0.5b, orz-1.5b, orz-7b, orz-32b, r1-14b, r1-32b, qwq-32b,
r1-llama8b, r1-math1.5b`.

## Reproducing the paper

Each stage has one entry-point `run.sh` that takes an optional config name (default:
all nine). Every step is idempotent: it skips work whose output already exists, so
runs can be resumed. The base prompt is deliberately minimal
(`Answer the following question:\nQ: {q}\nA:`) so any reasoning behaviour comes from
the steering vectors, not the prompt.

```bash
# 1. Thinking-model responses and taxonomies
cd generate-responses && uv run ./run.sh && uv run ./run_annotation.sh && cd ..
cd train-saes && uv run ./run.sh && cd ..

# 2. Rollouts (base + thinking, all datasets)          -> hybrid/results/response_cache_*
bash vllm-serve/run_rollouts.sh

# 3. Train the best-of-3 vector sets (seeds 42/43/44)  -> artifacts/mlp_vectors_qa_instr_h512*
bash train-vectors/run.sh

# 4. Select best-of-3 + evaluate the hybrid models     -> artifacts/mlp_eval_qa_instr_holdoutsel_h512
#    (MATH500, GSM8K, and the Hendrycks-MATH holdout)
bash hybrid/run.sh

# 5. Negative-control ablations (orz-1.5b, orz-32b)    -> artifacts/mlp_eval_qa_instr_holdoutsel_ablations
bash hybrid/run_ablations.sh

# 6. Render all figures and tables                     -> figures/figs
bash figures/run.sh
```

Selection promotes, per pair, the vector set with the highest gap recovered on the
holdout mix into `artifacts/mlp_vectors_qa_instr_holdoutsel_h512/<cfg>/` (recorded in
`.selected_from`); every downstream eval reads those vectors.

To run a single pair through a stage, pass its name, e.g. `bash hybrid/run.sh orz-32b`.

## Artifacts

All pipeline outputs live under `artifacts/`. Committed so results are usable and
reproducible without any reruns:

- the dataset definitions in `data/`,
- the final **selected steering vectors** in
  `artifacts/mlp_vectors_qa_instr_holdoutsel_h512/` (load these to steer directly),
- every **eval result** in `artifacts/mlp_eval_*/` (summaries, judge traces,
  per-category metrics), so `bash figures/run.sh` rebuilds all figures/tables from a
  clean clone,
- the rendered figures/tables in `figures/figs/`.

Only the large, regenerable bulk is git-ignored: the raw per-sample rollout text
(`*.jsonl`, ~15 GB), cached model rollouts (`*/results/`, ~5 GB), SAE activations,
training activation caches (`disagree_cache.pt`) and per-epoch snapshots, and the
per-run best-of-3 training outputs. All are reproduced by the stages above.

## Citation

```bibtex
@inproceedings{venhoff2026basemodels,
  title={Base Models Know How to Reason, Thinking Models Learn When},
  author={Venhoff, Constantin and Arcuschin, Iv{\'a}n and Torr, Philip and Conmy, Arthur and Nanda, Neel},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
}
```
