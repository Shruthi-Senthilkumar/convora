"""
eval/label_fusion_sample.py

Manual accuracy spot-check against fusion.py's final decisions (not just
raw semantic labels). Pulls a stratified sample covering:
  - True decisions (predicted end-of-speech)
  - False decisions (predicted continuation)
  - A mix of confidence bands (near-threshold cases are the most
    informative - that's where fusion logic errors would show up)

This is still NOT the real Phase 3 protocol (PRD sec 2.4) - single
annotator, small sample, no double-pass on disagreements. Treat the
resulting number as a real but rough signal, not a reportable metric.

Run:
    python eval/label_fusion_sample.py eval/fusion_result.json

Controls:
    c = you judge this as a genuine end-of-speech point
    n = you judge this as NOT end-of-speech (mid-thought/continuing)
    s = skip (genuinely unclear even with context)
    q = quit early, save progress
"""

import sys
import json
import random

TARGET_SAMPLE_SIZE = 40


def stratified_sample(results: list) -> list:
    true_cases = [r for r in results if r["fusion"]["is_end_of_speech"]]
    false_cases = [r for r in results if not r["fusion"]["is_end_of_speech"]]

    # Also grab near-threshold cases specifically (0.35-0.65 confidence)
    # regardless of which side they landed on - these are the most
    # informative for checking if the 0.5 cutoff is well-placed.
    near_threshold = [
        r for r in results
        if 0.35 <= r["fusion"]["confidence"] <= 0.65
    ]

    random.shuffle(true_cases)
    random.shuffle(false_cases)
    random.shuffle(near_threshold)

    half = TARGET_SAMPLE_SIZE // 2
    quarter = TARGET_SAMPLE_SIZE // 4

    sample = (
        true_cases[:half]
        + false_cases[:half]
        + near_threshold[:quarter]
    )

    # Dedup (a case could appear in both true/false split and near_threshold)
    seen = set()
    deduped = []
    for r in sample:
        key = r["candidate"]["word_index"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    random.shuffle(deduped)
    return deduped[:TARGET_SAMPLE_SIZE]


def run_labeling(sample: list):
    labeled = []
    print(f"\n{len(sample)} fragments to review.")
    print("For each, read the fragment and decide: does this sound like a")
    print("genuine end-of-speech point, or is the speaker clearly continuing?\n")
    print("Controls: [c]omplete/end-of-speech  [n]ot end-of-speech  [s]kip  [q]uit\n")
    print("=" * 90)

    for idx, r in enumerate(sample, 1):
        c = r["candidate"]
        f_res = r["fusion"]

        print(f"\n[{idx}/{len(sample)}] Pause: {c['pause_duration_s']}s | "
              f"Speaker changed: {c['speaker_changed']} | Speaker: {c['speaker']}")
        print(f"Fragment: ...{c['fragment']}")
        print(f"(Pipeline decision: {'END-OF-SPEECH' if f_res['is_end_of_speech'] else 'continuing'} "
              f"@ confidence {f_res['confidence']:.2f}, semantic={c['semantic_label']}/{c['semantic_source']})")

        while True:
            choice = input("Your judgment [c/n/s/q]: ").strip().lower()
            if choice in ("c", "n", "s", "q"):
                break
            print("Please enter c, n, s, or q.")

        if choice == "q":
            print("\nStopping early, saving progress so far...")
            break
        if choice == "s":
            continue

        human_says_eos = (choice == "c")
        pipeline_says_eos = f_res["is_end_of_speech"]
        agree = human_says_eos == pipeline_says_eos

        labeled.append({
            "candidate": c,
            "fusion": f_res,
            "human_says_end_of_speech": human_says_eos,
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

    print(f"\n{'=' * 90}")
    print(f"=== Fusion Accuracy Spot-Check (informal, NOT Phase 3 validation) ===")
    print(f"Labeled: {total}")
    print(f"Pipeline matched your judgment: {matches} ({100*matches/total:.0f}%)")
    print(f"Mismatches: {total - matches} ({100*(total-matches)/total:.0f}%)")

    # Break down by which direction the mismatch went (false positive vs
    # false negative) - these have different real-world costs per PRD sec 2.1
    false_positives = [
        l for l in labeled
        if not l["agrees_with_pipeline"] and l["fusion"]["is_end_of_speech"]
    ]
    false_negatives = [
        l for l in labeled
        if not l["agrees_with_pipeline"] and not l["fusion"]["is_end_of_speech"]
    ]

    print(f"\nFalse positives (pipeline said EOS, you said continuing): {len(false_positives)}")
    print(f"False negatives (pipeline said continuing, you said EOS): {len(false_negatives)}")
    print("(PRD sec 2.1 targets these SEPARATELY: FP <=5%, FN <=10% -")
    print(" this sample is too small to compare against those targets,")
    print(" but the split direction is still informative.)")

    if false_positives:
        print(f"\nFalse positive details:")
        for l in false_positives:
            print(f"  '{l['candidate']['fragment'][-50:]}' (conf={l['fusion']['confidence']:.2f})")

    if false_negatives:
        print(f"\nFalse negative details:")
        for l in false_negatives:
            print(f"  '{l['candidate']['fragment'][-50:]}' (conf={l['fusion']['confidence']:.2f})")

    print("\nNOTE: Informal single-annotator sample, not the Phase 3 protocol.")
    print("Treat as directional signal for whether fusion.py's weights/threshold")
    print("need adjustment before Task 4, not as a final accuracy metric.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/label_fusion_sample.py <fusion_result.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        results = json.load(f)

    sample = stratified_sample(results)
    labeled = run_labeling(sample)
    summarize(labeled)

    out_path = "eval/label_fusion_sample_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, indent=2)
    print(f"\nLabeled results saved to {out_path}")
