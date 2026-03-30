# GPQA Diamond Benchmark Results (Qwen 7B Hybrid)

## Setup

- **Dataset:** GPQA Diamond (Idavidrein/gpqa, gpqa_diamond split) -- 198 PhD-level science multiple choice questions
- **Thinking model:** Open-Reasoner-Zero/Open-Reasoner-Zero-7B (ORZ-7B)
- **Base model:** Qwen/Qwen2.5-7B
- **Config:** steering_layer=10, sae_layer=20, n_clusters=10, max_new_tokens=2000, max_thinking_tokens=2000
- **Coefficients:** 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
- **Token windows:** 0, -1, -15, -50, -100
- **Evaluated:** 133/198 questions (65 skipped where thinking model did not finish within 2000 tokens)

## Results

| Model | GPQA Diamond accuracy | Correct / Total |
|---|---|---|
| Qwen2.5-7B (base only) | 33.1% | 44/133 |
| ORZ-7B (thinking only) | 45.1% | 60/133 |
| Hybrid (adaptive) | 31.6% | 42/133 |
| Recovery rate | -16.7% | -- |

Recovery rate = (hybrid - base) / (thinking - base) = (31.6 - 33.1) / (45.1 - 33.1) = -1.5 / 12.0 = -12.5%

The hybrid does not recover any of the thinking model's gains on GPQA Diamond. It performs slightly below the base model.

## Comparison with Math500 (same 7B pair)

| Model | MATH500 | GPQA Diamond |
|---|---|---|
| Qwen2.5-7B (base) | 66.4% | 33.1% |
| ORZ-7B (thinking) | 81.8% | 45.1% |
| Hybrid (adaptive) | 79.6% | 31.6% |
| Recovery rate | 85.7% | -12.5% |

## Key Observations

1. **The hybrid method does not generalize from math to science reasoning on 7B models.** On Math500 the hybrid recovers 85.7% of the thinking model's gains; on GPQA Diamond it recovers none (slightly negative).

2. **EOS rate is the primary failure mode.** The hybrid model only produces a natural end-of-sequence 32.3% of the time (vs 98.5% for base, 100% for thinking). This means in ~68% of cases the hybrid hits the 2000 token ceiling with repetitive, degenerate output. The steering vectors disrupt the base model's ability to terminate generation.

3. **Qualitative failure pattern:** The hybrid often performs correct intermediate reasoning but then fails to commit to an answer, instead looping ("The final answer is X... The final answer is X...") until hitting the token limit. In some cases it arrives at the right intermediate answer but then second-guesses itself into a wrong final answer.

4. **The SAE clusters are math-reasoning oriented.** The 10 steering vector clusters (trained on ORZ-7B layer 20) are: Recalling Background Knowledge, Solution Outline, Performing Arithmetic Steps, Formula Evaluation, Listing Given Data, Formula Recall, Reporting Inference Results, Recalling Legal Rules, Outlining Reasoning Steps, Concept Definition. None encode "commit to answer / terminate" behavior, which may explain the EOS failure.

5. **65/198 questions (33%) were skipped** because the thinking model itself didn't finish within 2000 tokens. GPQA's long science questions with complex answer options require more generation budget than math problems. This biases evaluation toward shorter/easier questions.

6. **Base and thinking model performance:** ORZ-7B at 45.1% is well above random (25%), showing the thinking model has meaningful science reasoning capability. The base model at 33.1% is also above random. The gap (12 percentage points) is smaller than on Math500 (15.4 points), suggesting science reasoning benefits less from chain-of-thought at 7B scale.

## Conclusion

We evaluate the hybrid method on GPQA Diamond (198 PhD-level science multiple choice questions) using the Qwen 7B pair. The hybrid achieves 31.6% accuracy, compared to 45.1% for the thinking model alone and 33.1% for the base model alone, recovering -12.5% of the thinking model's gains over base (i.e., slightly degrading base performance). This contrasts with Math500 where the same method recovers 85.7%. The primary failure mode is the hybrid's inability to terminate generation (32.3% EOS rate), leading to repetitive degenerate output. The steering vectors, trained on math reasoning traces, do not include a "commit to answer" direction, which appears critical for MCQA tasks where concise answer selection is needed.
