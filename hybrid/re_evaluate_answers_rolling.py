import argparse
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from tqdm.auto import tqdm
except ImportError as exc:
    raise ImportError("tqdm is required for progress reporting") from exc


MODEL_SPECS: List[Tuple[str, str]] = [
    ("thinking", "Thinking Model"),
    ("base", "Base Model"),
    ("hybrid", "Hybrid Model"),
]

# Qwen models that have OpenReasonerZero counterparts
QWEN_ORZ_MODELS = [
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2.5-7b",
    "qwen2.5-32b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate rolling results with LLM judges using OpenAI Batch API")
    parser.add_argument(
        "--prefix",
        required=False,
        help="Rolling file prefix. Accepts absolute path or name inside the rolling directory.",
    )
    parser.add_argument(
        "--rolling-dir",
        type=str,
        default=None,
        help="Directory containing rolling outputs (defaults to results/rolling next to this script).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2",
        help="Judge model to query via OpenAI Batch API.",
    )
    parser.add_argument(
        "--max-judge-tokens",
        type=int,
        default=100,
        help="Maximum tokens returned by the judge model.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="qwen-orz",
        help="Filter which results to process: 'qwen-orz' (default, only Qwen models with ORZ counterparts), 'all' (all files), or a comma-separated list of model patterns.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Poll interval in seconds for checking batch status (default: 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Maximum requests per batch file (default: 5000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: run the full flow but only evaluate one record per file.",
    )
    return parser.parse_known_args()[0]


def _default_rolling_dir() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "results", "rolling")


def _resolve_prefix(raw_prefix: str, rolling_dir: str) -> str:
    if raw_prefix is None:
        raise ValueError("raw_prefix must not be None")
    if os.path.isabs(raw_prefix):
        prefix = raw_prefix
    else:
        prefix = os.path.join(rolling_dir, raw_prefix)
    if prefix.endswith(".jsonl"):
        prefix = prefix[:-6]
    return prefix


def _split_prefix_parts(path: str) -> Tuple[str, Optional[int]]:
    """Split a rolling file path into its prefix and part number."""
    base = os.path.basename(path)
    parent = os.path.dirname(path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    m = re.match(r"^(.*)_(\d+)$", base)
    if not m:
        return os.path.join(parent, base + ".jsonl"), None
    prefix = os.path.join(parent, m.group(1) + ".jsonl")
    return prefix, int(m.group(2))


def _list_ordered_files(prefix: str) -> List[str]:
    directory = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    assert base, "Prefix must include a filename component"

    files: List[str] = []
    legacy = os.path.join(directory, f"{base}.jsonl")
    if os.path.exists(legacy):
        files.append(legacy)

    part_pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.jsonl$")
    part_paths: List[str] = []
    for name in os.listdir(directory):
        match = part_pattern.match(name)
        if match:
            part_paths.append(os.path.join(directory, name))

    part_paths.sort(key=lambda path: int(part_pattern.match(os.path.basename(path)).group(1)))
    files.extend(part_paths)
    assert files, f"No rolling files found for prefix {prefix}"
    return files


def _extract_model_from_filename(filename: str) -> Optional[str]:
    """Extract the model name from a rolling filename like rolling_qwen2.5-1.5b_gsm8k.jsonl"""
    base = os.path.basename(filename)
    if not base.startswith("rolling_"):
        return None
    # Remove rolling_ prefix and .jsonl suffix
    rest = base[len("rolling_"):]
    if rest.endswith(".jsonl"):
        rest = rest[:-6]
    # Handle part numbers like _0, _1
    rest = re.sub(r"_\d+$", "", rest)
    # Extract model name (everything before the first _ that looks like a dataset)
    # Datasets are: gsm8k, math500, aime, etc.
    parts = rest.split("_")
    # Find where the dataset starts
    datasets = {"gsm8k", "math500", "aime"}
    model_parts = []
    for part in parts:
        if part in datasets:
            break
        model_parts.append(part)
    return "_".join(model_parts) if model_parts else None


def _matches_filter(filename: str, filter_patterns: List[str], exclude_patterns: Optional[List[str]] = None) -> bool:
    """Check if a filename matches any of the filter patterns."""
    model = _extract_model_from_filename(filename)
    if model is None:
        return False
    model_lower = model.lower()
    # Check exclusions first
    if exclude_patterns:
        for pattern in exclude_patterns:
            if pattern.lower() in model_lower:
                return False
    # Check inclusions
    for pattern in filter_patterns:
        if pattern.lower() in model_lower:
            return True
    return False


def _list_all_rollings(
    rolling_dir: str,
    filter_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, List[str]]]:
    files: Dict[str, List[Any]] = {}
    for name in os.listdir(rolling_dir):
        if not name.endswith(".jsonl"):
            continue
        if not name.startswith("rolling_"):
            continue
        # Skip vector_stats files
        if "_vector_stats" in name:
            continue
        # Apply filter if specified
        if filter_patterns is not None:
            if not _matches_filter(name, filter_patterns, exclude_patterns):
                continue
        full_path = os.path.join(rolling_dir, name)
        prefix, part = _split_prefix_parts(full_path)
        files.setdefault(prefix, []).append(full_path if part is None else (part, full_path))

    grouped: Dict[str, List[str]] = {}
    for prefix, entries in files.items():
        sorted_paths: List[str] = []
        parts = [e for e in entries if isinstance(e, tuple)]
        legacy = [e for e in entries if isinstance(e, str)]
        if legacy:
            sorted_paths.extend(sorted(legacy))
        for _, path in sorted(parts, key=lambda x: x[0]):
            sorted_paths.append(path)
        grouped[prefix] = sorted_paths

    if not grouped:
        return []
    return sorted(grouped.items(), key=lambda kv: kv[0])


def clean_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_judge_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    return (
        "Please evaluate whether the following answer to a math problem is correct.\n\n"
        f"Question: {question}\n\n"
        f"Correct answer: {correct_answer}\n\n"
        f"Model's answer: {model_answer}\n\n"
        "First, extract the final numerical answer from both the correct answer and model's answer.\n"
        "Then determine if the model's final numerical answer is equivalent to the correct final numerical answer.\n"
        "Just answer YES if the model's answer is correct, or NO if it's incorrect. Nothing else.\n"
    )


def _get_model_id(model_name: str) -> str:
    """Strip provider prefix if present (e.g., 'openai/gpt-4.1' -> 'gpt-4.1')."""
    for sep in ["/", ":"]:
        if sep in model_name:
            prefix, model_id = model_name.split(sep, 1)
            if prefix.lower() in {"openai", "anthropic", "google", "mistral", "mistralai"}:
                return model_id
    return model_name


def _is_transient_error(e: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    status_code = getattr(e, "status_code", None)
    if isinstance(status_code, int) and status_code >= 500:
        return True
    response = getattr(e, "response", None)
    resp_status = getattr(response, "status_code", None)
    if isinstance(resp_status, int) and resp_status >= 500:
        return True
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    name = type(e).__name__.lower()
    if "timeout" in name or "connect" in name or "connection" in name:
        return True
    return False


class BatchEvaluator:
    """Handles batch submission and polling for OpenAI Batch API."""

    def __init__(
        self,
        judge_model: str,
        max_tokens: int,
        poll_interval: int = 60,
        batch_size: int = 5000,
    ):
        self.judge_model = judge_model
        self.model_id = _get_model_id(judge_model)
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.client = OpenAI()

    def submit_batches(self, prompts: List[str]) -> List[str]:
        """Submit prompts to OpenAI Batch API and return batch IDs."""
        total_items = len(prompts)
        total_batches = (total_items + self.batch_size - 1) // self.batch_size
        print(f"[BatchEvaluator] Submitting {total_items} prompts in {total_batches} batch(es)")

        batch_ids: List[str] = []
        for batch_idx, start_idx in enumerate(range(0, total_items, self.batch_size), start=1):
            end_idx = min(start_idx + self.batch_size, total_items)
            batch_prompts = prompts[start_idx:end_idx]

            # Build JSONL requests
            requests_list: List[Dict[str, Any]] = []
            for i, prompt in enumerate(batch_prompts):
                custom_id = f"req_{start_idx + i}"
                body = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": self.max_tokens,
                }
                requests_list.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                })

            print(f"[BatchEvaluator] Submitting batch {batch_idx}/{total_batches}: {len(requests_list)} requests (idx {start_idx}-{end_idx - 1})")

            # Write JSONL and upload
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                for req in requests_list:
                    json.dump(req, f)
                    f.write("\n")
                input_path = f.name

            try:
                with open(input_path, "rb") as f:
                    input_file = self.client.files.create(file=f, purpose="batch")

                batch = self.client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": "re_evaluate_answers_rolling"},
                )
            finally:
                os.unlink(input_path)

            batch_id = batch.id
            print(f"[BatchEvaluator] OPENAI_BATCH_ID={batch_id}")
            batch_ids.append(batch_id)

        return batch_ids

    def poll_batches(self, batch_ids: List[str], total_items: int) -> Dict[int, str]:
        """Poll batches until completion and return responses indexed by position."""
        print(f"[BatchEvaluator] Polling {len(batch_ids)} batch(es) every {self.poll_interval}s...")

        pending = set(batch_ids)
        responses: Dict[int, str] = {}

        while pending:
            status_counts: Dict[str, int] = {}
            completed_now: List[str] = []

            for batch_id in list(pending):
                try:
                    batch = self.client.batches.retrieve(batch_id)
                except Exception as e:
                    if _is_transient_error(e):
                        print(f"[BatchEvaluator] Transient error retrieving batch {batch_id}: {e}")
                        status_counts["in_progress"] = status_counts.get("in_progress", 0) + 1
                        continue
                    raise

                status = str(batch.status)
                status_counts[status] = status_counts.get(status, 0) + 1

                if status not in {"completed", "failed", "expired", "cancelled"}:
                    continue

                if status != "completed":
                    error_details = ""
                    if hasattr(batch, "errors") and batch.errors:
                        error_details = f", errors={batch.errors}"
                    if batch.error_file_id:
                        try:
                            err_content = self.client.files.content(batch.error_file_id)
                            error_details += f", error_file_content={err_content.text[:2000]}"
                        except Exception:
                            error_details += f", error_file_id={batch.error_file_id}"
                    raise RuntimeError(
                        f"OpenAI batch did not complete successfully: batch_id={batch_id}, "
                        f"status={status}{error_details}"
                    )

                if batch.output_file_id is None:
                    req_counts = getattr(batch, "request_counts", None)
                    all_failed = (
                        req_counts is not None
                        and getattr(req_counts, "completed", 0) == 0
                        and getattr(req_counts, "failed", 0) > 0
                    )
                    if all_failed:
                        raise RuntimeError(
                            f"OpenAI batch failed: all requests failed. batch_id={batch_id}"
                        )
                    print(f"[BatchEvaluator] Batch {batch_id} completed but output_file_id not ready, will retry")
                    continue

                # Process responses
                try:
                    file_response = self.client.files.content(batch.output_file_id)
                    for line in file_response.text.splitlines():
                        obj = json.loads(line)
                        custom_id = obj.get("custom_id")
                        assert isinstance(custom_id, str) and custom_id.startswith("req_")

                        if obj.get("error") is not None:
                            print(f"[BatchEvaluator] Request {custom_id} failed: {obj['error']}")
                            idx = int(custom_id.split("_", 1)[1])
                            responses[idx] = ""
                            continue

                        resp_body = obj.get("response", {}).get("body", {})
                        choices = resp_body.get("choices", [])
                        if not choices or "message" not in choices[0]:
                            print(f"[BatchEvaluator] Request {custom_id} missing choices/message")
                            idx = int(custom_id.split("_", 1)[1])
                            responses[idx] = ""
                            continue

                        idx = int(custom_id.split("_", 1)[1])
                        content = choices[0]["message"].get("content") or ""
                        responses[idx] = content

                    # Also check error file
                    if batch.error_file_id:
                        err_response = self.client.files.content(batch.error_file_id)
                        for line in err_response.text.splitlines():
                            obj = json.loads(line)
                            custom_id = obj.get("custom_id", "?")
                            if custom_id.startswith("req_"):
                                idx = int(custom_id.split("_", 1)[1])
                                if idx not in responses:
                                    responses[idx] = ""
                except Exception as e:
                    if _is_transient_error(e):
                        print(f"[BatchEvaluator] Transient error processing batch {batch_id}: {e}")
                        continue
                    raise

                pending.remove(batch_id)
                completed_now.append(batch_id)

            print(f"[BatchEvaluator] Batches pending={len(pending)} (completed={len(completed_now)}): {status_counts}")
            if pending:
                time.sleep(self.poll_interval)

        # Fill missing responses
        missing = [i for i in range(total_items) if i not in responses]
        if missing:
            print(f"[BatchEvaluator] WARNING: {len(missing)} responses missing, filling with empty strings")
            for idx in missing:
                responses[idx] = ""

        return responses


def collect_all_prompts(
    files: List[Tuple[str, List[str]]],
    max_records_per_file: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, int, str]]]:
    """
    Collect all prompts from all files.

    Args:
        files: List of (prefix, paths) tuples
        max_records_per_file: If set, limit to this many records per file (for debug mode)

    Returns:
        prompts: List of all judge prompts
        prompt_mapping: List of (file_path, record_idx, model_key) for each prompt
    """
    prompts: List[str] = []
    prompt_mapping: List[Tuple[str, int, str]] = []

    for prefix, paths in tqdm(files, desc="Collecting prompts", unit="prefix"):
        for path in paths:
            with open(path, "r", encoding="utf-8") as src:
                records = [json.loads(line) for line in src if line.strip()]

            # Limit records in debug mode
            if max_records_per_file is not None:
                records_to_process = records[:max_records_per_file]
            else:
                records_to_process = records

            for idx, record in enumerate(records_to_process):
                question = str(record["question"])
                gold = str(record["gold_answer"])
                answers = record["answers"]

                for key, _ in MODEL_SPECS:
                    answer_text = clean_answer(str(answers[key]))
                    prompt = _build_judge_prompt(question, gold, answer_text)
                    prompts.append(prompt)
                    prompt_mapping.append((path, idx, key))

    return prompts, prompt_mapping


def update_all_files(
    files: List[Tuple[str, List[str]]],
    prompt_mapping: List[Tuple[str, int, str]],
    responses: Dict[int, str],
) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    """
    Update all files with judge responses.

    Returns:
        total_records: Total number of records processed
        aggregate_changed: Changed counts per model type
        aggregate_correct: Correct counts per model type
    """
    # Group responses by file
    file_updates: Dict[str, Dict[int, Dict[str, str]]] = {}
    for prompt_idx, (path, record_idx, model_key) in enumerate(prompt_mapping):
        response = responses.get(prompt_idx, "")
        file_updates.setdefault(path, {}).setdefault(record_idx, {})[model_key] = response

    total_records = 0
    aggregate_changed: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
    aggregate_correct: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}

    # Process each file
    all_paths = [path for _, paths in files for path in paths]
    for path in tqdm(all_paths, desc="Updating files", unit="file"):
        with open(path, "r", encoding="utf-8") as src:
            records = [json.loads(line) for line in src if line.strip()]

        updates = file_updates.get(path, {})
        file_changed: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
        file_correct: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}

        for idx, record in enumerate(records):
            if idx not in updates:
                continue

            existing_judges = record.get("judges", {})
            record.setdefault("judges", {})

            for key, _ in MODEL_SPECS:
                raw = updates[idx].get(key, "")
                is_correct = "yes" in raw.lower()

                # Get previous value
                prev_entry = existing_judges.get(key)
                prev_correct = None
                if isinstance(prev_entry, dict):
                    val = prev_entry.get("correct")
                    if isinstance(val, bool):
                        prev_correct = val

                record["judges"][key] = {"correct": bool(is_correct), "raw": raw}

                if is_correct:
                    file_correct[key] += 1
                    aggregate_correct[key] += 1

                if prev_correct is None or bool(is_correct) != bool(prev_correct):
                    file_changed[key] += 1
                    aggregate_changed[key] += 1

        total_records += len(records)

        # Write back
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as dst:
            for record in records:
                dst.write(json.dumps(record) + "\n")
        os.replace(temp_path, path)

    return total_records, aggregate_changed, aggregate_correct


def main() -> None:
    args = parse_args()
    rolling_dir = args.rolling_dir or _default_rolling_dir()
    assert os.path.isdir(rolling_dir), f"Rolling directory not found: {rolling_dir}"

    # Parse filter
    exclude_patterns: Optional[List[str]] = None
    if args.filter == "all":
        filter_patterns = None
    elif args.filter == "qwen-orz":
        filter_patterns = QWEN_ORZ_MODELS
        # Exclude variants that use different thinking models (e.g., DeepSeek)
        exclude_patterns = ["deepseek"]
    else:
        filter_patterns = [p.strip() for p in args.filter.split(",")]

    # Get files to process
    if args.prefix:
        prefix = _resolve_prefix(args.prefix, rolling_dir)
        files = [(prefix, _list_ordered_files(prefix))]
        print(f"Found {len(files[0][1])} files for prefix {prefix}")
    else:
        files = _list_all_rollings(rolling_dir, filter_patterns, exclude_patterns)
        if not files:
            print(f"No rolling files found in {rolling_dir} matching filter: {args.filter}")
            return
        total_files = sum(len(paths) for _, paths in files)
        print(f"Found {len(files)} rolling groups ({total_files} files) matching filter: {args.filter}")

    # Show what we're processing
    print("\nFiles to process:")
    for prefix, paths in files:
        print(f"  {os.path.basename(prefix)}: {len(paths)} file(s)")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above files. Exiting.")
        return

    if args.debug:
        print("\n[DEBUG MODE] Processing only 1 record per file.")

    # Collect all prompts
    print("\nPhase 1: Collecting all prompts...")
    max_records = 1 if args.debug else None
    prompts, prompt_mapping = collect_all_prompts(files, max_records_per_file=max_records)
    print(f"Collected {len(prompts)} prompts total")

    if not prompts:
        print("No prompts to process.")
        return

    # Submit batches
    print("\nPhase 2: Submitting to OpenAI Batch API...")
    evaluator = BatchEvaluator(
        judge_model=args.judge_model,
        max_tokens=args.max_judge_tokens,
        poll_interval=args.poll_interval,
        batch_size=args.batch_size,
    )
    batch_ids = evaluator.submit_batches(prompts)

    # Poll until complete
    print("\nPhase 3: Polling for completion...")
    responses = evaluator.poll_batches(batch_ids, len(prompts))
    print(f"Received {len(responses)} responses")

    # Update all files
    print("\nPhase 4: Updating files...")
    total_records, aggregate_changed, aggregate_correct = update_all_files(
        files, prompt_mapping, responses
    )

    # Print summary
    print(f"\nRe-evaluated {total_records} records across {sum(len(paths) for _, paths in files)} files.")
    if total_records > 0:
        print("\n==== Aggregate Summary ====")
        for key, label in MODEL_SPECS:
            changed = aggregate_changed.get(key, 0)
            changed_pct = changed / total_records * 100.0
            accuracy = aggregate_correct.get(key, 0) / total_records * 100.0
            print(f"{label}: changed_correct={changed} ({changed_pct:.1f}%), accuracy={accuracy:.1f}%")


if __name__ == "__main__":
    main()
