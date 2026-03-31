"""
sample.py — Create or extend human evaluation datasets for the four LLM judges.

Usage:
    cd /Users/ivan/src/base-models-reasoning-interp
    uv run python human_eval/sample.py --n 100 --seed 42
    uv run python human_eval/sample.py --n 100 --seed 42 --judges a b
"""

import argparse
import asyncio
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from openai import AsyncOpenAI

from utils.autograder_prompts import (
    build_accuracy_autograder_prompt,
    format_sentences_text_simple,
)
from utils.clustering import (
    parse_json_response,
    run_chat_batch_with_event_loop_handling,
)

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
ANNOTATED_RESPONSES_QWQ = "generate-responses/results/vars/annotated_responses_qwq-32b.json"
SAE_RESULTS_QWQ = "train-saes/results/vars/sae_topk_results_qwq-32b_layer27.json"
SAE_RESULTS_R1 = "train-saes/results/vars/sae_topk_results_deepseek-r1-distill-qwen-32b_layer27.json"
ROLLING_QWQ = "hybrid/results/rolling/rolling_qwen2.5-32b_math500.jsonl"
ROLLING_R1 = "hybrid/results/rolling/rolling_qwen2.5-32b-on-deepseek-r1-distill-qwen-32b_math500.jsonl"
OUTPUT_DIR = "human_eval/data"

SENTENCE_PATTERN = re.compile(r'\["[\d.]+:idx(\d+)"\](.*?)\["end-section"\]')
NOISE_PATTERN = re.compile(r'^[\d\s.,()+\-*/=^|~&\\]+$')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            obj["_line_idx"] = i
            rows.append(obj)
    return rows


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_existing(judge_letter):
    path = os.path.join(OUTPUT_DIR, f"judge_{judge_letter}.json")
    if os.path.exists(path):
        return load_json(path)
    return []


def assign_row_ids(existing, new_rows):
    max_id = max((r["row_id"] for r in existing), default=-1)
    for i, row in enumerate(new_rows):
        row["row_id"] = max_id + 1 + i


def is_noise_sentence(text):
    text = text.strip()
    if len(text) < 15:
        return True
    if NOISE_PATTERN.match(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Parse annotated responses into sentence pool
# ---------------------------------------------------------------------------

def parse_all_sentences(annotated_responses):
    """Parse all sentences from annotated responses, grouped by cluster_id.

    Returns:
        by_cluster: dict[int, list of (question_id, sent_idx, text)]
    """
    by_cluster = defaultdict(list)
    for item in annotated_responses:
        qid = item["question_id"]
        matches = SENTENCE_PATTERN.findall(item["annotated_thinking"])
        for sent_idx, (cluster_str, text) in enumerate(matches):
            text = text.strip()
            if is_noise_sentence(text):
                continue
            cluster_id = int(cluster_str)
            by_cluster[cluster_id].append((qid, sent_idx, text))
    return by_cluster


def build_sentence_lookup(annotated_responses):
    """Build sentence_text -> (question_id, sentence_index_in_trace, cluster_id) lookup."""
    lookup = {}
    for item in annotated_responses:
        qid = item["question_id"]
        matches = SENTENCE_PATTERN.findall(item["annotated_thinking"])
        for sent_idx, (cluster_str, text) in enumerate(matches):
            text = text.strip()
            if text and text not in lookup:
                lookup[text] = (qid, sent_idx, int(cluster_str))
    return lookup


# ---------------------------------------------------------------------------
# Judge A: Taxonomy Consistency
# ---------------------------------------------------------------------------

def get_existing_keys_a(existing):
    keys = set()
    for row in existing:
        p = row["provenance"]
        keys.add((p["source_model"], p["question_id"], p["sentence_index_in_trace"], p["tested_cluster_id"]))
    return keys


def sample_judge_a(n, seed, existing):
    rng = random.Random(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    per_cat_pos = n_pos // 10
    per_cat_neg = n_neg // 10

    existing_keys = get_existing_keys_a(existing)
    n_needed = n - len(existing)
    if n_needed <= 0:
        print(f"  Judge A: already have {len(existing)} >= {n}, skipping.")
        return []

    print("  Loading annotated responses and parsing sentences...")
    annotated = load_json(ANNOTATED_RESPONSES_QWQ)
    by_cluster = parse_all_sentences(annotated)

    sae_data = load_json(SAE_RESULTS_QWQ)
    categories = sae_data["results_by_cluster_size"]["10"]["all_results"][0]["categories"]
    cat_map = {int(c[0]): (c[1], c[2]) for c in categories}

    # Collect available cluster ids that have categories
    valid_clusters = sorted(cat_map.keys())
    print(f"  Valid clusters: {valid_clusters}, sentences per cluster: {[(c, len(by_cluster.get(c, []))) for c in valid_clusters]}")

    # Recalculate per-category targets based on n_needed
    n_pos_needed = n_needed // 2
    n_neg_needed = n_needed - n_pos_needed
    per_cat_pos = max(1, n_pos_needed // len(valid_clusters))
    per_cat_neg = max(1, n_neg_needed // len(valid_clusters))

    rows = []
    used_texts = set()

    for cluster_id in valid_clusters:
        title, desc = cat_map[cluster_id]

        # --- Positive samples: sentence from this cluster, tested against this category ---
        pool = [(qid, si, txt) for qid, si, txt in by_cluster.get(cluster_id, [])
                if (("qwq-32b", qid, si, cluster_id) not in existing_keys and txt not in used_texts)]
        rng.shuffle(pool)
        sampled = pool[:per_cat_pos]
        for qid, si, txt in sampled:
            used_texts.add(txt)
            rows.append(_make_judge_a_row(txt, cluster_id, cluster_id, title, desc, True, qid, si))

        # --- Negative samples: sentence from OTHER cluster, tested against this category ---
        neg_pool = []
        for other_cid in valid_clusters:
            if other_cid == cluster_id:
                continue
            for qid, si, txt in by_cluster.get(other_cid, []):
                if ("qwq-32b", qid, si, cluster_id) not in existing_keys and txt not in used_texts:
                    neg_pool.append((qid, si, txt, other_cid))
        rng.shuffle(neg_pool)
        sampled_neg = neg_pool[:per_cat_neg]
        for qid, si, txt, assigned_cid in sampled_neg:
            used_texts.add(txt)
            rows.append(_make_judge_a_row(txt, assigned_cid, cluster_id, title, desc, False, qid, si))

    # Fill any remainder (if uneven division)
    while len(rows) < n_needed:
        # Add more from any available cluster
        for cluster_id in valid_clusters:
            if len(rows) >= n_needed:
                break
            title, desc = cat_map[cluster_id]
            pool = [(qid, si, txt) for qid, si, txt in by_cluster.get(cluster_id, [])
                    if txt not in used_texts]
            if pool:
                qid, si, txt = rng.choice(pool)
                used_texts.add(txt)
                is_pos = len([r for r in rows if r["is_positive"]]) < n_needed // 2
                tested_cid = cluster_id if is_pos else rng.choice([c for c in valid_clusters if c != cluster_id])
                t, d = cat_map[tested_cid]
                rows.append(_make_judge_a_row(txt, cluster_id, tested_cid, t, d, is_pos, qid, si))

    rng.shuffle(rows)
    return rows[:n_needed]


def _make_judge_a_row(sentence, assigned_cluster_id, tested_cluster_id, title, desc, is_positive, question_id, sent_idx):
    return {
        "row_id": -1,  # assigned later
        "sentence": sentence,
        "category_title": title,
        "category_description": desc,
        "is_positive": is_positive,
        "provenance": {
            "source_model": "qwq-32b",
            "annotated_responses_file": "annotated_responses_qwq-32b.json",
            "taxonomy_file": "sae_topk_results_qwq-32b_layer27.json",
            "question_id": question_id,
            "sentence_index_in_trace": sent_idx,
            "assigned_cluster_id": assigned_cluster_id,
            "tested_cluster_id": tested_cluster_id,
            "sae_layer": 27,
            "n_clusters": 10,
        },
        "llm_judge": {},  # filled by run_llm_judge_a
        "human_labels": {},
    }


async def _call_openrouter_batch(prompts, model="openai/gpt-4.1-mini", max_concurrent=20):
    """Call OpenRouter API in parallel with concurrency limit."""
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(prompts)

    async def call_one(idx, prompt):
        async with semaphore:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=1e-19,
                response_format={"type": "json_object"},
            )
            results[idx] = resp.choices[0].message.content

    tasks = [call_one(i, p) for i, p in enumerate(prompts)]
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        await coro
        if i % 20 == 0 or i == len(tasks):
            print(f"    {i}/{len(tasks)} done")

    await client.close()
    return results


def run_llm_judge_a(rows, model="gpt-4.1-mini"):
    """Re-run LLM judge for Judge A rows. Fills llm_judge in-place."""
    if not rows:
        return rows
    print(f"  Running LLM judge for {len(rows)} Judge A items (model={model})...")
    prompts = []
    for row in rows:
        sentences_text = format_sentences_text_simple([row["sentence"]])
        prompt = build_accuracy_autograder_prompt(
            row["category_title"], row["category_description"], sentences_text
        )
        prompts.append(prompt)

    # Try the codebase utility first, fall back to OpenRouter on failure
    responses = None
    try:
        responses = run_chat_batch_with_event_loop_handling(prompts, model=model, json_mode=True)
        # Check if responses came back (the utility returns empty list on total failure)
        if not responses or all(r is None for r in responses):
            responses = None
    except Exception as e:
        print(f"  Codebase batch utility failed ({e}), falling back to OpenRouter...")

    if responses is None:
        print("  Using OpenRouter for API calls...")
        openrouter_model = f"openai/{model}"
        responses = asyncio.run(_call_openrouter_batch(prompts, model=openrouter_model))

    ts = now_iso()

    for row, prompt, response in zip(rows, prompts, responses):
        label = "Yes"
        explanation = ""
        raw_response = str(response) if response else ""
        if response:
            try:
                parsed = parse_json_response(response, expected_field="classifications")
                classifications = parsed.get("classifications", [])
                if classifications:
                    label = classifications[0].get("belongs_to_category", "Yes")
                    explanation = classifications[0].get("explanation", "")
            except Exception as e:
                print(f"    Warning: failed to parse response for row question_id={row['provenance']['question_id']}: {e}")
                resp_upper = str(response).upper()
                if "NO" in resp_upper and "YES" not in resp_upper:
                    label = "No"

        row["llm_judge"] = {
            "label": label,
            "model": model,
            "raw_response": raw_response,
            "prompt_used": prompt,
            "timestamp": ts,
        }

    print(f"  Judge A LLM calls complete. {len(rows)} responses parsed.")
    return rows


# ---------------------------------------------------------------------------
# Judge B: Taxonomy Completeness
# ---------------------------------------------------------------------------

def get_existing_keys_b(existing):
    keys = set()
    for row in existing:
        p = row["provenance"]
        keys.add((p["source_model"], p["question_id"], p["sentence_index_in_trace"]))
    return keys


def sample_judge_b(n, seed, existing):
    rng = random.Random(seed)
    existing_keys = get_existing_keys_b(existing)
    n_needed = n - len(existing)
    if n_needed <= 0:
        print(f"  Judge B: already have {len(existing)} >= {n}, skipping.")
        return []

    print("  Loading completeness responses and building sentence lookup...")
    sae_data = load_json(SAE_RESULTS_QWQ)
    completeness = sae_data["results_by_cluster_size"]["10"]["all_results"][0]["completeness_responses"]
    categories = sae_data["results_by_cluster_size"]["10"]["all_results"][0]["categories"]
    cat_map = {c[0]: (c[1], c[2]) for c in categories}

    # Build sentence -> provenance lookup from annotated responses
    annotated = load_json(ANNOTATED_RESPONSES_QWQ)
    sent_lookup = build_sentence_lookup(annotated)

    # Resolve provenance for each completeness entry and filter out existing
    resolved = []
    for entry in completeness:
        sentence = entry["sentence"]
        prov_info = sent_lookup.get(sentence, (None, None, None))
        question_id, sent_idx_in_trace, _ = prov_info
        key = ("qwq-32b", question_id, sent_idx_in_trace)
        if question_id is not None and key in existing_keys:
            continue
        resolved.append((entry, question_id, sent_idx_in_trace))

    # Group by cluster_id
    by_cluster = defaultdict(list)
    for item in resolved:
        by_cluster[item[0]["cluster_id"]].append(item)

    # Stratified sampling: ~10 per cluster, fill remainder from larger clusters
    per_cluster = max(1, n_needed // len(by_cluster)) if by_cluster else 0
    sampled = []

    for cid in sorted(by_cluster.keys(), key=int):
        pool = by_cluster[cid]
        rng.shuffle(pool)
        take = min(len(pool), per_cluster)
        sampled.extend(pool[:take])

    # Fill remainder from all clusters
    remaining = n_needed - len(sampled)
    if remaining > 0:
        already = {id(item) for item in sampled}
        leftover = [item for cid in by_cluster for item in by_cluster[cid] if id(item) not in already]
        rng.shuffle(leftover)
        sampled.extend(leftover[:remaining])

    sampled = sampled[:n_needed]

    rows = []
    for entry, question_id, sent_idx_in_trace in sampled:
        rows.append({
            "row_id": -1,
            "sentence": sentence,
            "category_title": entry["title"],
            "category_description": entry["description"],
            "provenance": {
                "source_model": "qwq-32b",
                "annotated_responses_file": "annotated_responses_qwq-32b.json",
                "taxonomy_file": "sae_topk_results_qwq-32b_layer27.json",
                "question_id": question_id,
                "sentence_index_in_trace": sent_idx_in_trace,
                "assigned_cluster_id": int(entry["cluster_id"]),
                "sae_layer": 27,
                "n_clusters": 10,
            },
            "llm_judge": {
                "rating": entry["completeness_score"],
                "model": "gpt-4.1-mini",
                "explanation": entry["explanation"],
                "timestamp": now_iso(),
            },
            "human_labels": {},
        })

    cdist = defaultdict(int)
    for r in rows:
        cdist[r["provenance"]["assigned_cluster_id"]] += 1
    print(f"  Judge B: sampled {len(rows)} new entries (cluster distribution: {dict(sorted(cdist.items()))})")
    return rows[:n_needed]


# ---------------------------------------------------------------------------
# Judge C: Taxonomy Independence
# ---------------------------------------------------------------------------

def get_existing_keys_c(existing):
    keys = set()
    for row in existing:
        c1 = row["category_1"]
        c2 = row["category_2"]
        keys.add((c1["source_model"], c1["cluster_id"], c2["source_model"], c2["cluster_id"]))
    return keys


def _load_judge_c_pairs(sae_path, model_id, n_clusters_str):
    """Load category pairs with similarity data from SAE results."""
    sae_data = load_json(sae_path)
    result = sae_data["results_by_cluster_size"][n_clusters_str]["all_results"][0]
    categories = result["categories"]
    cat_map = {int(c[0]): (c[1], c[2]) for c in categories}
    matrix = result["semantic_orthogonality_matrix"]
    explanations = result["semantic_orthogonality_explanations"]

    pairs = []
    n = len(categories)
    for i, j in combinations(range(n), 2):
        orthogonality = matrix[i][j]
        similarity_0_to_1 = round(1 - orthogonality, 4)
        similarity_0_to_10 = round(similarity_0_to_1 * 10)
        expl_key = f"{i},{j}"
        explanation = explanations.get(expl_key, "")

        title_i, desc_i = cat_map[i]
        title_j, desc_j = cat_map[j]

        pairs.append({
            "category_1": {
                "title": title_i,
                "description": desc_i,
                "cluster_id": i,
                "source_model": model_id,
            },
            "category_2": {
                "title": title_j,
                "description": desc_j,
                "cluster_id": j,
                "source_model": model_id,
            },
            "provenance": {
                "taxonomy_file": os.path.basename(sae_path),
                "sae_layer": 27,
                "n_clusters": int(n_clusters_str),
            },
            "llm_judge": {
                "similarity_0_to_10": similarity_0_to_10,
                "similarity_0_to_1": similarity_0_to_1,
                "model": "gpt-4.1-mini",
                "explanation": explanation,
                "timestamp": now_iso(),
            },
        })
    return pairs


def sample_judge_c(n, seed, existing):
    rng = random.Random(seed)
    existing_keys = get_existing_keys_c(existing)
    n_needed = n - len(existing)
    if n_needed <= 0:
        print(f"  Judge C: already have {len(existing)} >= {n}, skipping.")
        return []

    # QwQ-32B: all 45 pairs
    qwq_pairs = _load_judge_c_pairs(SAE_RESULTS_QWQ, "qwq-32b", "10")
    # R1-Distill: all 105 pairs, sample 55
    r1_pairs = _load_judge_c_pairs(SAE_RESULTS_R1, "deepseek-r1-distill-qwen-32b", "15")

    # Filter out existing
    qwq_new = [p for p in qwq_pairs
                if (p["category_1"]["source_model"], p["category_1"]["cluster_id"],
                    p["category_2"]["source_model"], p["category_2"]["cluster_id"]) not in existing_keys]
    r1_new = [p for p in r1_pairs
              if (p["category_1"]["source_model"], p["category_1"]["cluster_id"],
                  p["category_2"]["source_model"], p["category_2"]["cluster_id"]) not in existing_keys]

    # Take all QwQ pairs, sample from R1 to fill
    rows = []
    for p in qwq_new:
        p["row_id"] = -1
        p["human_labels"] = {}
        rows.append(p)

    r1_needed = n_needed - len(rows)
    if r1_needed > 0:
        rng.shuffle(r1_new)
        for p in r1_new[:r1_needed]:
            p["row_id"] = -1
            p["human_labels"] = {}
            rows.append(p)

    rng.shuffle(rows)
    print(f"  Judge C: {len(rows)} new pairs (QwQ: {sum(1 for r in rows if r['category_1']['source_model']=='qwq-32b')}, R1-Distill: {sum(1 for r in rows if r['category_1']['source_model']!='qwq-32b')})")
    return rows[:n_needed]


# ---------------------------------------------------------------------------
# Judge D: Benchmark Answer Scoring
# ---------------------------------------------------------------------------

def get_existing_keys_d(existing):
    keys = set()
    for row in existing:
        p = row["provenance"]
        keys.add((p["rolling_results_file"], p["question_id"], p["model_type"]))
    return keys


def _flatten_rolling(filepath, source_model):
    """Flatten a rolling JSONL into per-model-type entries."""
    lines = load_jsonl(filepath)
    basename = os.path.basename(filepath)
    entries = []
    for line in lines:
        line_idx = line["_line_idx"]
        question_id = f"math500_{line_idx:04d}"
        for model_type in ["thinking", "base", "hybrid"]:
            judge_data = line["judges"][model_type]
            correct = judge_data["correct"]
            raw = judge_data["raw"]
            repetitions = judge_data.get("repetitions", [])

            hybrid_details = line.get("hybrid_details", {})
            if model_type == "hybrid":
                coefficient = hybrid_details.get("coefficients")
                token_window = hybrid_details.get("token_windows", [None])[0] if hybrid_details.get("token_windows") else None
            else:
                coefficient = None
                token_window = None

            entries.append({
                "math_question": line["question"],
                "correct_answer": line["gold_answer"],
                "model_response": line["answers"][model_type],
                "provenance": {
                    "source_model": source_model,
                    "base_model": "qwen2.5-32b",
                    "model_type": model_type,
                    "rolling_results_file": basename,
                    "question_id": question_id,
                    "dataset": line.get("dataset", "math500"),
                    "coefficient": coefficient,
                    "token_window": token_window,
                },
                "llm_judge": {
                    "label": "Yes" if correct else "No",
                    "model": "gpt-5.2",
                    "raw_response": raw,
                    "n_repetitions": len(repetitions),
                    "timestamp": now_iso(),
                },
                "_correct": correct,
            })
    return entries


def sample_judge_d(n, seed, existing):
    rng = random.Random(seed)
    existing_keys = get_existing_keys_d(existing)
    n_needed = n - len(existing)
    if n_needed <= 0:
        print(f"  Judge D: already have {len(existing)} >= {n}, skipping.")
        return []

    print("  Loading rolling results...")
    entries_qwq = _flatten_rolling(ROLLING_QWQ, "qwq-32b")
    entries_r1 = _flatten_rolling(ROLLING_R1, "deepseek-r1-distill-qwen-32b")
    all_entries = entries_qwq + entries_r1

    # Filter out existing
    all_entries = [e for e in all_entries
                   if (e["provenance"]["rolling_results_file"],
                       e["provenance"]["question_id"],
                       e["provenance"]["model_type"]) not in existing_keys]

    # Split by correctness
    correct_pool = [e for e in all_entries if e["_correct"]]
    incorrect_pool = [e for e in all_entries if not e["_correct"]]

    n_correct = n_needed // 2
    n_incorrect = n_needed - n_correct

    # Stratified sampling across model_types and source files
    def stratified_sample(pool, target):
        by_stratum = defaultdict(list)
        for e in pool:
            key = (e["provenance"]["rolling_results_file"], e["provenance"]["model_type"])
            by_stratum[key].append(e)

        per_stratum = max(1, target // len(by_stratum)) if by_stratum else 0
        sampled = []
        for key in sorted(by_stratum.keys()):
            items = by_stratum[key]
            rng.shuffle(items)
            sampled.extend(items[:per_stratum])

        # Fill remainder
        remaining = target - len(sampled)
        if remaining > 0:
            already = {id(e) for e in sampled}
            leftover = [e for e in pool if id(e) not in already]
            rng.shuffle(leftover)
            sampled.extend(leftover[:remaining])
        return sampled[:target]

    sampled_correct = stratified_sample(correct_pool, n_correct)
    sampled_incorrect = stratified_sample(incorrect_pool, n_incorrect)

    rows = sampled_correct + sampled_incorrect
    rng.shuffle(rows)

    # Clean up internal field and add row_id/human_labels
    for row in rows:
        row.pop("_correct", None)
        row["row_id"] = -1
        row["human_labels"] = {}

    strata = defaultdict(int)
    for r in rows:
        strata[r["provenance"]["model_type"]] += 1
    print(f"  Judge D: {len(rows)} new entries. Correct: {len(sampled_correct)}, Incorrect: {len(sampled_incorrect)}. Types: {dict(strata)}")
    return rows[:n_needed]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample human evaluation data for LLM judges")
    parser.add_argument("--n", type=int, default=100, help="Target total samples per judge")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--judges", nargs="+", default=["a", "b", "c", "d"],
                        choices=["a", "b", "c", "d"], help="Which judges to sample for")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for judge in args.judges:
        print(f"\n{'='*60}")
        print(f"Judge {judge.upper()}")
        print(f"{'='*60}")

        existing = load_existing(judge)
        print(f"  Existing: {len(existing)} rows")

        if judge == "a":
            new_rows = sample_judge_a(args.n, args.seed, existing)
            new_rows = run_llm_judge_a(new_rows)
        elif judge == "b":
            new_rows = sample_judge_b(args.n, args.seed, existing)
        elif judge == "c":
            new_rows = sample_judge_c(args.n, args.seed, existing)
        elif judge == "d":
            new_rows = sample_judge_d(args.n, args.seed, existing)

        if new_rows:
            assign_row_ids(existing, new_rows)
            all_rows = existing + new_rows
            path = os.path.join(OUTPUT_DIR, f"judge_{judge}.json")
            save_json(path, all_rows)
            print(f"  Saved {len(all_rows)} total rows to {path}")
        else:
            print(f"  No new rows to add.")


if __name__ == "__main__":
    main()
