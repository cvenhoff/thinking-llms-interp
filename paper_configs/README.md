# Paper Run Configs (Reference)

These scripts document the **exact** hybrid inference configs used to produce
Table 1 results in the paper.  They use the original `hybrid_token.py` pipeline
(diff-of-means steering vectors, per-token coefficient/window grid search).

They are **NOT** the new `optimize_correction_vectors.py` training pipeline.
For the new joint bias+cats training, see `run_joint_small_models.sh`.

## Config summary

| # | Base           | Thinking                        | Steer L | SAE L | K  |
|---|----------------|---------------------------------|---------|-------|----|
| 1 | Qwen2.5-0.5B   | Open-Reasoner-Zero-0.5B         | 9       | 8     | 10 |
| 2 | Qwen2.5-1.5B   | Open-Reasoner-Zero-1.5B         | 10      | 8     | 5  |
| 3 | Qwen2.5-7B     | Open-Reasoner-Zero-7B           | 10      | 20    | 10 |
| 4 | Qwen2.5-14B    | DeepSeek-R1-Distill-Qwen-14B    | 18      | 38    | 5  |
| 5 | Llama-3.1-8B   | DeepSeek-R1-Distill-Llama-8B    | 12      | 6     | 15 |
| 6 | Qwen2.5-32B    | QwQ-32B                         | 24      | 27    | 10 |
| 7 | Qwen2.5-32B    | DeepSeek-R1-Distill-Qwen-32B    | 24      | 27    | 15 |
| 8 | Qwen2.5-32B    | Open-Reasoner-Zero-32B          | 24      | 27    | 15 |
| 9 | Qwen2.5-Math-1.5B | DeepSeek-R1-Distill-Qwen-1.5B | 10     | 4     | 15 |
