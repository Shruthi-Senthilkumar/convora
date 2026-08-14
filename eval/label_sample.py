"""
eval/label_sample.py

Quick, informal accuracy check - NOT a substitute for the real Phase 3
labeled-set protocol (PRD sec 2.4). This pulls a stratified sample of
pause candidates (mixing rule/llm/degraded sources) and lets you mark
each one against what the pipeline predicted, giving a rough sanity-check
accuracy number.

Run:
    python eval/label_sample.py eval/pause_candidates_result.json

Controls at each prompt:
    c = mark as "complete" (your judgment)
    i = mark as "incomplete" (your judgment)
    s = skip (genuinely unsure / not enough context)
    q = quit early and save what you've labeled so far
"""

import sys
import json
import random

SAMPLE_SIZE_PER_SOURCE = 15  # up to this many from each source category


def stratified_sample(candidates: list) -> list:
    by_source = {"rule": [], "llm": [], "degraded": []}
    for c in candidates:
        by_source.setdefault(c["semantic_source"], []).append(c)

    sample = []
    for source, items in by_source.items():
        random.shuffle(items)
        sample.extend(items[:SAMPLE_SIZE_PER_SOURCE])

    random.shuffle(sample)
    return sample


def run_labeling(sample: list):
    labeled = []
    print(f"\n{len(sample)} fragments to review. For each: read the fragment,")
    print("decide what YOU think the correct label is.\n")
    print("Controls: [c]omplete  [i]ncomplete  [s]kip  [q]uit and save\n")
    print("=" * 80)

    for idx, c in enumerate(sample, 1):
        print(f"\n[{idx}/{len(sample)}] Pause duration: {c['pause_duration_s']}s")
        print(f"Fragment: ...{c['fragment']}")
        print(f"(Pipeline said: {c['semantic_label']} via {c['semantic_source']})")

        while True:
            choice = input("Your judgment [c/i/s/q]: ").strip().lower()
            if choice in ("c", "i", "s", "q"):
                break
            print("Please enter c, i, s, or q.")

        if choice == "q":
            print("\nStopping early, saving progress so far...")
            break
        if choice == "s":
            continue

        human_label = "complete" if choice == "c" else "incomplete"
        agree = human_label == c["semantic_label"]

        labeled.append({
            **c,
            "human_label": human_label,
            "agrees_with_pipeline": agree,
        })

        print("MATCH" if agree else "MISMATCH <-- pipeline disagreed with you")

    return labeled


def summarize(labeled: list):
    if not labeled:
        print("\nNo fragments labeled - nothing to summarize.")
        return

    total = len(labeled)
    matches = sum(1 for l in labeled if l["agrees_with_pipeline"])

    print(f"\n{'=' * 80}")
    print(f"=== Rough Accuracy Summary (informal, NOT Phase 3 validation) ===")
    print(f"Labeled: {total}")
    print(f"Pipeline matched your judgment: {matches} ({100*matches/total:.0f}%)")
    print(f"Mismatches: {total - matches} ({100*(total-matches)/total:.0f}%)")

    by_source = {}
    for l in labeled:
        src = l["semantic_source"]
        by_source.setdefault(src, {"total": 0, "matches": 0})
        by_source[src]["total"] += 1
        if l["agrees_with_pipeline"]:
            by_source[src]["matches"] += 1

    print(f"\nBreakdown by source:")
    for src, stats in by_source.items():
        pct = 100 * stats["matches"] / stats["total"] if stats["total"] else 0
        print(f"  {src:<10} {stats['matches']}/{stats['total']} ({pct:.0f}%)")

    mismatches = [l for l in labeled if not l["agrees_with_pipeline"]]
    if mismatches:
        print(f"\nMismatch details (worth reviewing):")
        for m in mismatches:
            print(f"  '{m['fragment'][-50:]}' -> pipeline: {m['semantic_label']}, you: {m['human_label']} ({m['semantic_source']})")

    print("\nNOTE: This is an informal single-annotator spot-check on a small")
    print("sample, not the real Phase 3 protocol (which needs >=200 boundaries,")
    print(">=10 speakers, double-pass labeling for ambiguous cases per PRD sec")
    print("2.4). Treat this number as directional, not a reportable metric.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/label_sample.py <pause_candidates_result.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        candidates = json.load(f)

    sample = stratified_sample(candidates)
    labeled = run_labeling(sample)
    summarize(labeled)

    out_path = "eval/label_sample_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, indent=2)
    print(f"\nLabeled results saved to {out_path}")
