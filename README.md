# Base Models Know How to Reason, Thinking Models Learn When

Code for the paper [Base Models Know How to Reason, Thinking Models Learn When](https://arxiv.org/abs/2510.07364).

**Website:** [thinking-llms-interp.com](https://thinking-llms-interp.com/)

This branch contains the **category-vector hybrid-steering** pipeline: we discover
per-model reasoning taxonomies with SAEs, train a small MLP that predicts *which*
taxonomy direction to apply *where*, and build "hybrid" models (a base model plus
the learned category-steering vectors) that we evaluate against the corresponding
thinking model. Everything needed to reproduce the paper's figures and tables lives
here; the canonical pipeline is in [`mlp_pipeline/canonical/`](mlp_pipeline/canonical).

## Setup

### Requirements

- Python 3.10+
- `uv` installed (`pip install uv` or see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/))
- A SLURM cluster with GPUs (the large models need 2× 80 GB GPUs). The `.sh`
  drivers submit `srun` jobs; the underlying `python` entry points also run
  stand-alone if you prefer to schedule them yourself.

### Install

```bash
git clone https://github.com/cvenhoff/cot-interp.git
cd cot-interp
uv sync
```

Copy your API credentials for the LLM judge into a local `.env` (git-ignored):

```bash
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

## Repository layout

| Path | Purpose |
|------|---------|
| `generate-responses/` | Generate + annotate thinking-model responses (SAE inputs). |
| `train-saes/`         | Train SAEs and build the per-model reasoning **taxonomies**. |
| `train-vectors/`      | Category-vector **training engine** (`optimize_correction_vectors.py`, `coef_mlp.py`). |
| `vllm-serve/`         | vLLM rollout generation (`generate_rollouts.py`). |
| `hybrid/`             | Hybrid-model **evaluation engine** (`hybrid_eval.py`). |
| `utils/`              | Shared library (SAE loading, clustering, steering helpers). |
| `mlp_pipeline/canonical/` | **All canonical run scripts** (training → selection → eval → figures). |
| `data/`               | Training mix + held-out evaluation sets (`training_mix_v1`, `hendrycks_holdout_eval`, `trainmix_holdout_eval`). |

The nine canonical model pairs (base ← thinking) and their SAE settings are defined
in `mlp_pipeline/canonical/train_qa_instr_hsweep.sh`:
`orz-0.5b, orz-1.5b, orz-7b, orz-32b, r1-14b, r1-32b, qwq-32b, r1-llama8b, r1-math1.5b`.
The steering MLP uses `MLP_HIDDEN=512` throughout.

## Reproducing the paper

The pipeline runs in the following stages. Each driver is **idempotent and
self-healing** — re-running skips already-completed work — so the whole pipeline
can be resumed after interruptions and will converge to the artifacts below.

### 1. Thinking-model responses (SAE inputs)

```bash
cd generate-responses && uv run ./run.sh          # generate responses
uv run ./run_annotation.sh                         # annotate traces with a taxonomy
```

### 2. SAE taxonomies

```bash
cd train-saes && uv run ./run.sh
```

Collects activations, trains SAEs across layers/cluster-sizes for each model,
generates cluster titles/descriptions, evaluates the candidate taxonomies, and
plots the taxonomy grid.

### 3. Rollouts for vector training + evaluation

From the repo root:

```bash
bash mlp_pipeline/canonical/gen_think_final_final.sh    # thinking rollouts (math500, gsm8k)
bash mlp_pipeline/canonical/gen_base_qa_instr.sh        # base rollouts (qa_instr prompt)
bash mlp_pipeline/canonical/gen_base_holdoutmix.sh      # base rollouts for the holdout-mix selection set
bash mlp_pipeline/canonical/gen_hendrycks_holdout.sh    # base + thinking rollouts for the Hendrycks-MATH holdout
```

The base prompt is deliberately minimal — `Answer the following question:\nQ: {q}\nA:`
(`--base_prompt_style qa_instr`) — so that any reasoning behaviour is induced by the
steering vectors rather than seeded by the prompt.

### 4. Train category vectors (best-of-3)

```bash
# run1 (the reference set) for each config -> mlp_vectors_qa_instr_h512/<cfg>
for cfg in orz-0.5b orz-1.5b orz-7b orz-32b r1-14b r1-32b qwq-32b r1-llama8b r1-math1.5b; do
    CONFIG=$cfg MLP_HIDDEN=512 bash mlp_pipeline/canonical/train_qa_instr_hsweep.sh
done

# run2 + run3 for every config -> mlp_vectors_qa_instr_h512_bo3/<cfg>/run{2,3}
bash mlp_pipeline/canonical/orchestrate_bo3_train.sh
```

### 5. Select best-of-3 + out-of-sample hybrid eval

```bash
bash mlp_pipeline/canonical/holdout_chains_launch.sh
```

For each config this builds a hybrid model from each of the three vector sets,
measures gap-recovered on the **holdout mix** (a gold-answer subset of the
training-mix validation split, never used for gradient updates), promotes the
**highest holdout-mix gap-recovered** set into
`mlp_vectors_qa_instr_holdoutsel_h512/<cfg>` (recorded in `.selected_from`), and
then evaluates it on GSM8K and MATH500 into `mlp_eval_qa_instr_holdoutsel_h512/`.

### 6. Hendrycks-MATH holdout eval

```bash
bash mlp_pipeline/canonical/hendrycks_launch.sh
```

Evaluates the selected vectors on the 1k Hendrycks-MATH holdout (disjoint from the
training mix and from MATH500) into `mlp_eval_hendrycks_holdout_qa_instr_holdoutsel_h512/`.

### 7. Negative-control ablations

```bash
bash mlp_pipeline/canonical/launch_holdoutsel_ablations.sh
```

Runs the four ablations (`randcat`, `randV`, `mlponly`, `randpos`) on the two
size-spanning configs (`orz-1.5b`, `orz-32b`) into
`mlp_eval_qa_instr_holdoutsel_ablations/`.

### 8. Figures and tables

```bash
cd mlp_pipeline/canonical
uv run python render_result_tables.py               # Tables 1-3 -> figs/
uv run python plot_ablation_bars.py                 # ablation bar plot -> figs/
uv run python render_loss_curves_qa_instr_h512.py   # vector-loss curves
uv run python make_hybrid_example_figure_orz32b.py  # qualitative hybrid rollout
```

Rendered PDFs/PNGs are written to `mlp_pipeline/canonical/figs/` (committed).

## Artifacts

Large, regenerable artifacts are git-ignored (see `.gitignore`): trained vector
checkpoints (`mlp_vectors_*`), hybrid-eval outputs (`mlp_eval_*`), cached rollouts
(`*/results/`), SAE activations (`train-saes/results/`), and cluster logs. They are
all reproduced by the stages above. The committed artifacts are the dataset
definitions (`data/`) and the final rendered figures/tables (`mlp_pipeline/canonical/figs/`).

## Citation

If you find this work useful, please cite:

```bibtex
@misc{venhoff2025basemodelsknowreason,
      title={Base Models Know How to Reason, Thinking Models Learn When},
      author={Constantin Venhoff and Iván Arcuschin and Philip Torr and Arthur Conmy and Neel Nanda},
      year={2025},
      eprint={2510.07364},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.07364},
}
```
