# Canonical category-vector pipeline

Every script needed to reproduce the paper's hybrid-steering results, in the order
they run. See the top-level `README.md` for the end-to-end walkthrough. All drivers
are idempotent and self-healing.

## Rollout generation
| Script | Produces |
|--------|----------|
| `gen_think_final_final.sh` | Thinking-model rollouts (math500, gsm8k) → `hybrid/results/response_cache_final/` |
| `gen_base_qa_instr.sh` / `gen_base_qa_instr.py` | Base rollouts with the `qa_instr` prompt |
| `gen_base_holdoutmix.sh` | Base rollouts for the holdout-mix selection set |
| `gen_hendrycks_holdout.sh` | Base + thinking rollouts for the Hendrycks-MATH holdout |
| `build_holdoutmix_eval.py`, `build_hendrycks_holdout.py` | Build the holdout-mix / Hendrycks-MATH evaluation sets under `data/` |

## Vector training (best-of-3)
| Script | Produces |
|--------|----------|
| `train_qa_instr_hsweep.sh` | Trains one MLP category-vector set (`CONFIG`, `MLP_HIDDEN=512`) → `mlp_vectors_qa_instr_h512/<cfg>` (run1). Calls `train-vectors/optimize_correction_vectors.py`. |
| `orchestrate_bo3_train.sh` | Trains run2 + run3 for every config → `mlp_vectors_qa_instr_h512_bo3/<cfg>/run{2,3}` |

## Selection + evaluation
| Script | Role |
|--------|------|
| `holdout_chains_launch.sh` → `holdout_chain.sh` | Per config: build a hybrid from each of the 3 vector sets, measure gap-recovered on the holdout mix, promote the best set → `mlp_vectors_qa_instr_holdoutsel_h512/<cfg>` (`.selected_from`), then eval GSM8K + MATH500. Uses `eval_holdoutmix_run.sh` + `eval_qa_instr_hsweep.sh`. |
| `hendrycks_launch.sh` → `hendrycks_chain.sh` | Evaluate the selected vectors on the 1k Hendrycks-MATH holdout. Uses `eval_hendrycks_holdout_hybrid.sh`. |
| `launch_holdoutsel_ablations.sh` (`_parallel.sh`) → `eval_qa_instr_holdoutsel_ablation.sh` | The four negative-control ablations (`randcat`, `randV`, `mlponly`, `randpos`) on `orz-1.5b` + `orz-32b`. |
| `eval_qa_instr_hsweep.sh`, `eval_holdoutmix_run.sh`, `eval_hendrycks_holdout_hybrid.sh` | Hybrid-eval drivers (call `hybrid/hybrid_eval.py`). |
| `judge_extra_think_samples.py`, `aggregate_samples_final.py` | LLM-judge scoring + summary aggregation. |

## Figures and tables
| Script | Output (in `figs/`) |
|--------|---------------------|
| `render_result_tables.py` | Training-mix table, main-results table, gap-recovered table |
| `plot_ablation_bars.py` | Ablation bar plot |
| `render_loss_curves_qa_instr_h512.py` | Vector training-loss curves |
| `make_hybrid_example_figure_orz32b.py` | Qualitative ORZ-32B hybrid rollout |

## Vector layout (uniform across all 9 configs)
```
mlp_vectors_qa_instr_h512/<cfg>            # run1 (reference set)
mlp_vectors_qa_instr_h512_bo3/<cfg>/run2   # run2
mlp_vectors_qa_instr_h512_bo3/<cfg>/run3   # run3
mlp_vectors_qa_instr_holdoutsel_h512/<cfg> # best-of-3 winner (see .selected_from)
```
