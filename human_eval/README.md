# Human Evaluation of LLM Judge Quality

Validates the four LLM judges used in the paper with human annotations. Each judge gets 100 sampled datapoints; humans annotate them independently, and we compute agreement metrics (Cohen's kappa, Spearman correlation, etc.).

| Judge | Task | LLM | Human task |
|-------|------|-----|------------|
| A: Taxonomy Consistency | Does sentence belong to category? | GPT-4.1-mini | Binary (y/n) |
| B: Taxonomy Completeness | How well does sentence fit category? | GPT-4.1-mini | Rating (0-10) |
| C: Taxonomy Independence | How similar are two categories? | GPT-4.1-mini | Rating (0-10) |
| D: Benchmark Scoring | Is the model's answer correct? | GPT-5.2 | Binary (y/n) |

## Usage

All commands run from the repo root:

```bash
cd /Users/ivan/src/base-models-reasoning-interp
```

### 1. Sample datapoints

Creates `data/judge_{a,b,c,d}.json` with 100 items each, including LLM judge labels.

```bash
uv run python human_eval/sample.py --n 100 --seed 42
```

Judge A re-runs GPT-4.1-mini for each sampled pair (falls back to OpenRouter if OpenAI quota is exhausted). Judges B/C/D fetch labels from existing stored results.

Options:
- `--n N` -- target total samples per judge (default 100)
- `--seed S` -- random seed (default 42)
- `--judges a b c d` -- which judges to sample (default all)

Running again with a higher `--n` extends the existing files without duplicates.

### 2. Annotate

Interactive single-keypress CLI. Saves after every annotation; quit with `q` and resume later.

```bash
uv run python human_eval/annotate.py --judge a --annotator ivan
```

- Judges A/D: press `y` (Yes) or `n` (No)
- Judges B/C: press `0`-`9` for that rating, or `1` then `0` for 10

### 3. Compute agreement

Reads all JSONs, discovers annotators, prints metrics, and writes a results file.

```bash
uv run python human_eval/compute_agreement.py
```

Options:
- `--judges a b c d` -- which judges to analyze (default all)
- `--annotators name1 name2` -- filter to specific annotators (default all found)

## File layout

```
human_eval/
  sample.py
  annotate.py
  compute_agreement.py
  data/
    judge_a.json    # 100 sentence-category pairs (50 pos + 50 neg)
    judge_b.json    # 100 sentence-category pairs (positive only, with ratings)
    judge_c.json    # 100 category pairs (45 QwQ + 55 R1-Distill)
    judge_d.json    # 100 model responses (50 correct + 50 incorrect)
```

The JSON files are the single source of truth for samples, LLM labels, and human labels.
