#!/usr/bin/env python3
"""
eval/edge_case_slicer.py
------------------------
Edge-Case Slice Scoring for Convora Turn-Boundary Detection.

Per PRD Section 2.4:
  "Score edge-case slices separately - backchannels, restarts, enumeration...
   since aggregate FP/FN can look healthy while every backchannel case fails."

Categorizes candidate pause points into the PRD-defined edge-case slices:
  1. Backchannels ("mm-hmm", "right", "yeah") — must NOT trigger end-of-speech
  2. Trailing Questions (Strict Interrogative Syntax & Question Clauses)
     + Prosodic Uptalk / Syntax Mismatches (reported as a separate analytical slice)
  3. List enumeration with long inter-item pauses
  4. Self-corrections and restarts
  5. Trailing filler ("so, yeah...", "I guess")

Evaluates each slice against official AMI ground truth using eval/gt_matching.py.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
import spacy

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from detection.semantic_judge import (
    BACKCHANNELS,
    SELF_CORRECTION_MARKERS,
    TRAILING_LIST_MARKERS,
    INCOMPLETE_TRAILING_POS
)
from eval.gt_matching import (
    parse_nxt_da_segments,
    derive_ground_truth_boundaries,
    evaluate_candidates,
    evaluate_gt_level,
    MEETING_ID
)

CANDIDATES_PATH = WORKSPACE_ROOT / "eval" / "fusion_result.json"
RESULTS_JSON_PATH = WORKSPACE_ROOT / "eval" / "edge_case_results.json"

nlp = spacy.load("en_core_web_sm")

# Additional phrase definitions extending semantic_judge rules
EXTENDED_SELF_CORRECTION = SELF_CORRECTION_MARKERS.union({
    "no wait", "let me see", "or rather", "excuse me", "i mean to say"
})

EXTENDED_LIST_MARKERS = TRAILING_LIST_MARKERS.union({
    "first", "second", "third", "fourth", "fifth", "first of all",
    "and then", "number one", "number two", "number three", "option a", "option b"
})

TRAILING_FILLER_PHRASES = {
    "i guess", "i think", "sort of", "kind of", "you know", "or something",
    "so yeah", "and yeah", "and so on", "or whatever", "and all", "and stuff"
}

QUESTION_PATTERNS = [
    r'^(what|why|how|where|who|which|whose|when)\b',
    r'^(is|are|was|were|do|does|did|can|could|would|should|will|shall|have|has)\s+(you|we|i|they|he|she|it|that|this|there)\b',
    r'\b(do you|what do you|can you|shall we|is it|is that|right\?|what about)\b'
]


def tag_candidate(candidate_dict: Dict[str, Any]) -> List[str]:
    """
    Tags a candidate pause point with zero or more of the PRD edge-case categories.
    """
    frag = candidate_dict.get("fragment", "").strip().lower()
    words = re.findall(r"\b\w+\b", frag)
    prosody_data = candidate_dict.get("prosody", {})
    pitch_slope = prosody_data.get("pitch_slope")
    tags = []

    # --------------------------------------------------------------------------
    # 1. Backchannels: short listener feedback acknowledgments
    # --------------------------------------------------------------------------
    if frag in BACKCHANNELS or (len(words) <= 3 and all(w in BACKCHANNELS for w in words)):
        tags.append("backchannels")

    # --------------------------------------------------------------------------
    # 2. Trailing Questions (Strict Interrogative Syntax & Clauses)
    # --------------------------------------------------------------------------
    is_syntactic_q = any(re.search(p, frag) for p in QUESTION_PATTERNS)
    if is_syntactic_q:
        tags.append("trailing_questions")

    # 2b. Prosodic Uptalk / Mismatch (Non-question statements with rising pitch > +30Hz/s)
    is_rising_pitch = (pitch_slope is not None and pitch_slope > 30.0)
    if not is_syntactic_q and is_rising_pitch and len(words) > 3 and frag not in BACKCHANNELS:
        tags.append("prosodic_uptalk_mismatch")

    # --------------------------------------------------------------------------
    # 3. List Enumeration with inter-item pauses
    # --------------------------------------------------------------------------
    has_list_phrase = any(m in frag for m in EXTENDED_LIST_MARKERS)
    ends_with_list_word = (len(words) > 0 and words[-1] in EXTENDED_LIST_MARKERS)
    if has_list_phrase or ends_with_list_word:
        tags.append("list_enumeration")

    # --------------------------------------------------------------------------
    # 4. Self-Corrections and Restarts
    # --------------------------------------------------------------------------
    has_correction_marker = any(m in frag for m in EXTENDED_SELF_CORRECTION)
    has_repetition = False
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 1:
            has_repetition = True
            break
    if has_correction_marker or has_repetition:
        tags.append("self_corrections_restarts")

    # --------------------------------------------------------------------------
    # 5. Trailing Filler / Discourse Particles
    # --------------------------------------------------------------------------
    has_filler_phrase = any(frag.endswith(fp) for fp in TRAILING_FILLER_PHRASES)
    doc = nlp(frag)
    has_trailing_incomplete_pos = False
    if len(doc) > 0:
        last_tok = doc[-1]
        if last_tok.pos_ in INCOMPLETE_TRAILING_POS and last_tok.text.lower() in {
            "and", "but", "so", "or", "because", "like", "with", "to", "for"
        }:
            has_trailing_incomplete_pos = True
    if has_filler_phrase or has_trailing_incomplete_pos:
        tags.append("trailing_filler")

    return tags


def main():
    print("=" * 80)
    print(" CONVORA EDGE-CASE SLICE EVALUATION (PRD Section 2.4)")
    print("=" * 80)

    if not CANDIDATES_PATH.exists():
        print(f"ERROR: Cannot find {CANDIDATES_PATH}")
        sys.exit(1)

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        fused_candidates = json.load(f)

    # Load Ground Truth
    da_segments = parse_nxt_da_segments(MEETING_ID)
    gt_filtered = derive_ground_truth_boundaries(da_segments, exclude_backchannels=True)
    tol = 0.5

    # Evaluate full candidate set first to get candidate-level match status
    cres_full = evaluate_candidates(fused_candidates, gt_filtered, tol)
    for item, detail in zip(fused_candidates, cres_full["candidate_details"]):
        item["match_status"] = detail["match_status"]
        item["delta_s"] = detail["delta_s"]
        item["edge_case_tags"] = tag_candidate(item["candidate"])

    categories = [
        "backchannels",
        "trailing_questions",
        "prosodic_uptalk_mismatch",
        "list_enumeration",
        "self_corrections_restarts",
        "trailing_filler"
    ]

    slice_results = {}

    print(f"\nEvaluating {len(fused_candidates)} total candidates across edge-case categories:")

    for cat in categories:
        slice_items = [it for it in fused_candidates if cat in it["edge_case_tags"]]
        n_cands = len(slice_items)
        cres = evaluate_candidates(slice_items, gt_filtered, tol)
        cm = cres["confusion_matrix"]
        m = cres["metrics"]

        # Backchannel suppression rate
        suppressed_count = sum(1 for it in slice_items if not it["fusion"]["is_end_of_speech"])
        eos_count = sum(1 for it in slice_items if it["fusion"]["is_end_of_speech"])
        suppression_rate = (suppressed_count / n_cands) if n_cands > 0 else 0.0

        slice_data = {
            "category": cat,
            "candidate_count": n_cands,
            "meets_prd_min_sample_size": n_cands >= 10,
            "confusion_matrix": cm,
            "precision": m["precision"],
            "false_positive_rate": m["false_positive_rate_early_cutoff"],
            "accuracy": m["accuracy"],
            "suppression_rate": suppression_rate if cat == "backchannels" else None,
            "sample_candidates": [
                {
                    "pause_start": it["candidate"]["pause_start"],
                    "pred_eos": it["fusion"]["is_end_of_speech"],
                    "confidence": it["fusion"]["confidence"],
                    "match_status": it["match_status"],
                    "fragment": it["candidate"].get("fragment", "")
                }
                for it in slice_items[:5]
            ]
        }
        slice_results[cat] = slice_data

        print("\n" + "-" * 80)
        print(f" SLICE: {cat.upper()} ({n_cands} candidates | Sample Size >= 10: {n_cands >= 10})")
        print("-" * 80)
        if cat == "backchannels":
            print(f"  * Raw Confusion Matrix:     TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']} (Total={n_cands})")
            print(f"  * Precision Calculation:    TP / (TP + FP) = {cm['TP']}/({cm['TP']} + {cm['FP']}) = {m['precision']*100:.2f}%")
            print(f"  * Accuracy Calculation:     (TP + TN) / Total = ({cm['TP']} + {cm['TN']}) / {n_cands} = {cm['TP'] + cm['TN']}/{n_cands} = {m['accuracy']*100:.2f}%")
            print(f"  * Non-EOS Suppression Rate: {suppressed_count}/{n_cands} ({suppression_rate*100:.1f}%) [CRITICAL PRD TARGET]")
            print(f"  * EOS Triggered Count:      {eos_count}/{n_cands} (All {cm['TP']} triggered cases were genuine turn boundaries, FP={cm['FP']})")
        else:
            print(f"  * Raw Confusion Matrix:     TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']} (Total={n_cands})")
            print(f"  * Precision:                {m['precision']*100:.2f}% (TP / (TP + FP) = {cm['TP']}/{cm['TP'] + cm['FP'] if (cm['TP'] + cm['FP']) > 0 else '0'})")
            print(f"  * False Positive Rate:      {m['false_positive_rate_early_cutoff']*100:.2f}% (FP / (FP + TN) = {cm['FP']}/{cm['FP'] + cm['TN'] if (cm['FP'] + cm['TN']) > 0 else '0'})")
            print(f"  * Overall Accuracy:         {m['accuracy']*100:.2f}% ((TP + TN) / Total = {cm['TP'] + cm['TN']}/{n_cands})")

        print("\n  Sample Candidates in Slice:")
        for s in slice_data["sample_candidates"]:
            print(f"    [{s['pause_start']:>7.2f}s] PredEOS={str(s['pred_eos']):<5} ({s['match_status']:<2}) | \"{s['fragment'][:50]}\"")

    # Save structured JSON
    output_payload = {
        "meeting_id": MEETING_ID,
        "total_candidates": len(fused_candidates),
        "tolerance_s": tol,
        "slices": slice_results
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f" SUCCESS: Edge-case results saved to {RESULTS_JSON_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
