"""
annotate.py — Interactive terminal CLI for human annotation of LLM judge data.

Usage:
    cd /Users/ivan/src/base-models-reasoning-interp
    uv run python human_eval/annotate.py --judge a --annotator ivan
"""

import argparse
import json
import os
import sys
import termios
import time
import tty
from datetime import datetime, timezone

DATA_DIR = "human_eval/data"

JUDGE_NAMES = {
    "a": "Taxonomy Consistency",
    "b": "Taxonomy Completeness",
    "c": "Taxonomy Independence",
    "d": "Benchmark Scoring",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_keypress():
    """Read a single keypress without requiring Enter. Consumes escape sequences fully."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # If escape byte, consume the rest of the escape sequence so its
        # trailing characters (which may include digits) aren't mistaken
        # for user input.
        if ch == "\x1b":
            import select
            # Read all bytes that follow within 50ms (the escape sequence)
            while select.select([fd], [], [], 0.05)[0]:
                sys.stdin.read(1)
            # Return a sentinel that no input handler will match
            return "\x1b"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    # Handle Ctrl+C
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def get_binary_input():
    """Get y/n/q input. Returns 'Yes', 'No', or None (quit)."""
    while True:
        ch = get_keypress()
        if ch.lower() == "y":
            return "Yes"
        elif ch.lower() == "n":
            return "No"
        elif ch.lower() == "q":
            return None


def get_rating_input():
    """Get 0-10 rating or q (quit). Returns int or None."""
    while True:
        ch = get_keypress()
        if ch.lower() == "q":
            return None
        if ch.lower() == "t":
            return 10
        if ch in "0123456789":
            return int(ch)


def load_data(judge):
    path = os.path.join(DATA_DIR, f"judge_{judge}.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run sample.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def save_data(judge, data):
    path = os.path.join(DATA_DIR, f"judge_{judge}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def get_terminal_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 120


def wrap_text(text, width=None, indent=2):
    """Simple word-wrap with indent. Uses terminal width by default."""
    if width is None:
        width = get_terminal_width() - 2
    indent_str = " " * indent
    words = text.split()
    lines = []
    current_line = indent_str
    for word in words:
        if len(current_line) + len(word) + 1 > width:
            lines.append(current_line)
            current_line = indent_str + word
        else:
            if current_line == indent_str:
                current_line += word
            else:
                current_line += " " + word
    if current_line.strip():
        lines.append(current_line)
    return "\n".join(lines)


SEP = "\u2500" * 51


def display_judge_a(row, idx, total):
    clear_screen()
    print(f"[{idx}/{total}]\n")
    print(f'Category: "{row["category_title"]}"')
    print(wrap_text(row["category_description"]))
    print()
    print("Sentence:")
    print(wrap_text(f'"{row["sentence"]}"'))
    print()
    print("Does this sentence belong to this category?")
    print("  [y] Yes  [n] No  [q] Quit")
    sys.stdout.write("> ")
    sys.stdout.flush()


def display_judge_b(row, idx, total):
    clear_screen()
    print(f"[{idx}/{total}]\n")
    print(f'Category: "{row["category_title"]}"')
    print(wrap_text(row["category_description"]))
    print()
    print("Sentence:")
    print(wrap_text(f'"{row["sentence"]}"'))
    print()
    print("How well does this sentence fit this category? (0-10)")
    print("  [0-9] Rate  [t] 10  [q] Quit")
    sys.stdout.write("> ")
    sys.stdout.flush()


def display_judge_c(row, idx, total):
    clear_screen()
    print(f"[{idx}/{total}]\n")
    print(f'Category 1: "{row["category_1"]["title"]}"')
    print(wrap_text(row["category_1"]["description"]))
    print()
    print(f'Category 2: "{row["category_2"]["title"]}"')
    print(wrap_text(row["category_2"]["description"]))
    print()
    print("How similar are these two categories? (0-10)")
    print("  0 = completely different functions")
    print("  10 = essentially the same function")
    print("  [0-9] Rate  [t] 10  [q] Quit")
    sys.stdout.write("> ")
    sys.stdout.flush()


def display_judge_d(row, idx, total):
    clear_screen()
    print(f"[{idx}/{total}]\n")
    print("Question:")
    print(wrap_text(row["math_question"]))
    print()
    print(f'Correct answer: {row["correct_answer"]}')
    print()
    response = row["model_response"]
    answer = row["correct_answer"].strip()
    if answer and answer in response:
        response = response.replace(answer, f"\033[1;32m{answer}\033[0m")
    print(f'Model response ({row["provenance"]["model_type"]}):')
    print(wrap_text(response))
    print()
    print("Did the model arrive at the correct answer?")
    print("  [y] Yes  [n] No  [q] Quit")
    sys.stdout.write("> ")
    sys.stdout.flush()


DISPLAY_FNS = {
    "a": display_judge_a,
    "b": display_judge_b,
    "c": display_judge_c,
    "d": display_judge_d,
}


def annotate(judge, annotator):
    data = load_data(judge)
    total = len(data)
    judge_name = JUDGE_NAMES[judge]
    is_binary = judge in ("a", "d")

    # Find first unannotated item
    start_idx = 0
    for i, row in enumerate(data):
        if annotator not in row.get("human_labels", {}):
            start_idx = i
            break
    else:
        print(f"Done! All {total} items already annotated by {annotator}.")
        return

    done_count = sum(1 for r in data if annotator in r.get("human_labels", {}))
    print(f"Judge {judge.upper()}: {judge_name}")
    print(f"Annotator: {annotator}")
    print(f"Progress: {done_count}/{total}")
    print(f"\nStarting from item {start_idx + 1}...")
    time.sleep(1)

    for i in range(start_idx, total):
        row = data[i]
        if annotator in row.get("human_labels", {}):
            continue

        done_count = sum(1 for r in data if annotator in r.get("human_labels", {}))
        display_fn = DISPLAY_FNS[judge]
        display_fn(row, done_count + 1, total)

        if is_binary:
            result = get_binary_input()
        else:
            result = get_rating_input()

        if result is None:
            # Quit
            done_count = sum(1 for r in data if annotator in r.get("human_labels", {}))
            print(f"\n\nProgress: {done_count}/{total} annotated. Resume with same command.")
            return

        # Save annotation
        if "human_labels" not in row:
            row["human_labels"] = {}

        if is_binary:
            row["human_labels"][annotator] = {
                "label": result,
                "timestamp": now_iso(),
            }
        else:
            row["human_labels"][annotator] = {
                "rating": result,
                "timestamp": now_iso(),
            }

        save_data(judge, data)
        print(f"\n\u2713 Saved.")
        time.sleep(0.3)

    done_count = sum(1 for r in data if annotator in r.get("human_labels", {}))
    clear_screen()
    print(f"Done! All {total} items annotated by {annotator}.")


def main():
    parser = argparse.ArgumentParser(description="Human annotation CLI for LLM judge evaluation")
    parser.add_argument("--judge", required=True, choices=["a", "b", "c", "d"],
                        help="Which judge to annotate (a/b/c/d)")
    parser.add_argument("--annotator", required=True,
                        help="Annotator ID string")
    args = parser.parse_args()

    try:
        annotate(args.judge, args.annotator)
    except KeyboardInterrupt:
        data = load_data(args.judge)
        done = sum(1 for r in data if args.annotator in r.get("human_labels", {}))
        total = len(data)
        print(f"\n\nInterrupted. Progress: {done}/{total} annotated. Resume with same command.")


if __name__ == "__main__":
    main()
