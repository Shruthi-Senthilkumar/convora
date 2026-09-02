#!/usr/bin/env python3
"""
evaluate_against_ami_ground_truth.py
------------------------------------
Convora Accuracy Evaluation against official AMI Corpus Human Ground-Truth Annotations.

This script evaluates Convora's end-of-speech detector decisions (from eval/fusion_result.json)
against official human-annotated Dialogue Act (DA) turn boundaries from the AMI Meeting Corpus
for meeting ES2002a.

AMI Stand-off Annotation Schema (NXT XML):
- words/ES2002a.[A-D].words.xml: Word-level start/end timestamps.
- dialogueActs/ES2002a.[A-D].dialog-act.xml: DA segment references to word IDs (via <nite:child href="...">).

Ground-Truth Derivation:
- A DA segment's end time marks a genuine turn boundary when the next chronological DA segment
  is from a DIFFERENT speaker.
- Backchannel Filter (Primary Rule): Dialogue acts of type `ami_da_1` (Backchannel / bck, e.g. "mm-hmm", "yeah")
  are non-substantive listener feedback that do NOT transfer conversational floor. Excluding backchannels
  yields 224 true turn-transfer boundaries (compared to 295 unfiltered transitions). Both metrics are reported.

Metrics Computed:
- Precision, Recall, F1 Score
- False Positive Rate (Early cutoff rate on natural intra-turn pauses)
- False Negative Rate (Missed or delayed end-of-speech rate on genuine turn boundaries)
- Sensitivity across tolerance windows (+/-0.3s, +/-0.5s baseline, +/-0.8s)
"""

import os
import sys
import json
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared evaluation logic
from eval.gt_matching import (
    parse_nxt_da_segments,
    derive_ground_truth_boundaries,
    evaluate_candidates,
    evaluate_gt_level,
    MEETING_ID,
    FUSION_RESULT_PATH,
    OUTPUT_JSON_PATH
)


def main():
    print("=" * 80)
    print(" CONVORA ACCURACY EVALUATION AGAINST AMI CORPUS GROUND TRUTH (ES2002a)")
    print("=" * 80)

    # 1. Load pipeline candidates
    if not FUSION_RESULT_PATH.exists():
        print(f"ERROR: Cannot find fusion result file at {FUSION_RESULT_PATH}")
        sys.exit(1)

    with open(FUSION_RESULT_PATH, 'r', encoding='utf-8') as f:
        pipeline_candidates = json.load(f)

    print(f"Loaded {len(pipeline_candidates)} pipeline candidates from {FUSION_RESULT_PATH.name}")

    # 2. Parse AMI NXT DA segments
    da_segments = parse_nxt_da_segments(MEETING_ID)
    bck_count = sum(1 for d in da_segments if d['da_type'] == 'ami_da_1')
    print(f"Parsed {len(da_segments)} DA segments from AMI annotations across 4 speakers (A, B, C, D)")
    print(f"  - Backchannels (ami_da_1 / bck): {bck_count} instances")

    # 3. Derive Ground Truth Turn Boundaries
    gt_filtered = derive_ground_truth_boundaries(da_segments, exclude_backchannels=True)
    gt_unfiltered = derive_ground_truth_boundaries(da_segments, exclude_backchannels=False)

    print(f"Derived Ground-Truth Turn Boundaries:")
    print(f"  - Primary (Filtered - Excl. Backchannels): {len(gt_filtered)} genuine floor-transfer boundaries")
    print(f"  - Baseline (Unfiltered - All DA transitions): {len(gt_unfiltered)} total speaker transitions")

    # 4. Perform Evaluation across tolerances (+/-0.3s, +/-0.5s baseline, +/-0.8s)
    tolerances = [0.3, 0.5, 0.8]
    eval_results_primary = {}       # candidate-level: Precision, FPR, Accuracy
    eval_results_unfiltered = {}    # candidate-level: unfiltered GT
    gt_level_primary = {}           # GT-centric: Recall, FNR, F1 (monotone-correct)
    gt_level_unfiltered = {}        # GT-centric: unfiltered GT

    for tol in tolerances:
        eval_results_primary[str(tol)] = evaluate_candidates(pipeline_candidates, gt_filtered, tol)
        eval_results_unfiltered[str(tol)] = evaluate_candidates(pipeline_candidates, gt_unfiltered, tol)
        gt_level_primary[str(tol)] = evaluate_gt_level(pipeline_candidates, gt_filtered, tol)
        gt_level_unfiltered[str(tol)] = evaluate_gt_level(pipeline_candidates, gt_unfiltered, tol)

    # Primary baseline at +/-0.5s
    res_05 = eval_results_primary["0.5"]
    gtl_05 = gt_level_primary["0.5"]
    m05 = res_05['metrics']
    cm05 = res_05['confusion_matrix']
    gm05 = gtl_05['metrics']

    # 5. Print Formal Summary Tables
    print("\n" + "=" * 80)
    print(" PRIMARY EVALUATION SUMMARY (Excl. Listener Backchannels, Tol +/-0.5s)")
    print("=" * 80)
    print()
    print("  *** CRITICAL FINDING ***")
    print(f"  False Negative Rate (FNR):    {gm05['fnr_gt_centric'] * 100:.2f}% — System MISSES {gtl_05['gt_fn_total']} of {gtl_05['total_gt_boundaries']} genuine turn boundaries")
    print(f"    - FN_FUSION (candidate exists, voted EOS=False): {gtl_05['gt_fn_fusion']}")
    print(f"    - FN_VAD    (no candidate within tol — VAD gap):  {gtl_05['gt_fn_vad']}")
    print(f"  Target FNR per PRD v1.5:      <=10%")
    print(f"  STATUS:                       FAILING (FNR {gm05['fnr_gt_centric'] * 100:.1f}% >> 10% target)")
    print()
    print(f"  Total GT Boundaries (filtered): {gtl_05['total_gt_boundaries']}")
    print(f"  Total Candidates Evaluated:     {res_05['total_candidates']}")
    print(f"  GT-level TP (boundaries hit):   {gtl_05['gt_tp']}")
    print(f"  GT-level FN (boundaries missed): {gtl_05['gt_fn_total']}")
    print()
    print(f"  Candidate-Level Confusion Matrix (for Precision/FPR/Accuracy):")
    print(f"    TP={cm05['TP']}, FP={cm05['FP']}, FN={cm05['FN']}, TN={cm05['TN']}")
    print("-" * 80)
    print(f"  Overall Decision Accuracy:      {m05['accuracy'] * 100:.2f}%")
    print(f"  Precision (Candidate-centric):  {m05['precision'] * 100:.2f}% (TP / (TP + FP) = {cm05['TP']}/{cm05['TP'] + cm05['FP']})")
    print(f"  Precision (GT-nearest matched): {gm05['precision_gt_centric'] * 100:.2f}% ({gtl_05['metrics']['precision_gt_centric']*100:.2f}% of EOS triggers were 1-to-1 nearest matches)")
    print(f"  Recall    (GT-centric):         {gm05['recall_gt_centric'] * 100:.2f}% ({gtl_05['gt_tp']}/{gtl_05['total_gt_boundaries']} genuine boundaries hit)")
    print(f"  F1 Score  (GT-centric):         {gm05['f1_gt_centric']:.4f}")
    print(f"  False Positive Rate (FPR):      {m05['false_positive_rate_early_cutoff'] * 100:.2f}% (Early Cutoff Rate on non-turn pauses: FP/(FP+TN) = {cm05['FP']}/{cm05['FP'] + cm05['TN']})")
    print(f"  False Negative Rate (FNR):      {gm05['fnr_gt_centric'] * 100:.2f}% (GT-centric — MONOTONE CORRECT)")

    print("\n" + "=" * 80)
    print(" TOLERANCE WINDOW SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(" [Recall and FNR use GT-centric matching — monotone guaranteed]")
    print(f" {'Tolerance':<12} | {'Precision':<10} | {'Recall(GT)':<11} | {'F1(GT)':<10} | {'FPR (Early)':<12} | {'FNR(GT)':<10} | {'AccuracyNote':<14}")
    print("-" * 80)
    for tol in tolerances:
        m = eval_results_primary[str(tol)]['metrics']
        gm = gt_level_primary[str(tol)]['metrics']
        acc = m['accuracy']
        print(f" +/-{tol:<8.1f}s | {m['precision']*100:<9.2f}% | {gm['recall_gt_centric']*100:<10.2f}% | {gm['f1_gt_centric']:<10.4f} | {m['false_positive_rate_early_cutoff']*100:<11.2f}% | {gm['fnr_gt_centric']*100:<9.2f}% | acc={acc*100:.1f}%")

    print("\n" + "=" * 80)
    print(" RULE COMPARISON: FILTERED VS UNFILTERED GT (+/-0.5s)")
    print("=" * 80)
    m_unf = eval_results_unfiltered["0.5"]['metrics']
    gm_unf = gt_level_unfiltered["0.5"]['metrics']
    print(f" Primary Filtered (Excl. Backchannels):    F1={gm05['f1_gt_centric']:.4f}, Precision={gm05['precision_gt_centric']*100:.2f}%, Recall={gm05['recall_gt_centric']*100:.2f}%, FNR={gm05['fnr_gt_centric']*100:.2f}%")
    print(f" Baseline Unfiltered (All DA Transitions): F1={gm_unf['f1_gt_centric']:.4f}, Precision={gm_unf['precision_gt_centric']*100:.2f}%, Recall={gm_unf['recall_gt_centric']*100:.2f}%, FNR={gm_unf['fnr_gt_centric']*100:.2f}%")

    # 6. Sample Alignment Rows (Ground Truth vs Prediction)
    print("\n" + "=" * 80)
    print(" REAL ALIGNMENT SAMPLE ROWS (First 10 Candidates, tol=+/-0.5s)")
    print("=" * 80)
    print(f" {'PauseStart':<10} | {'PredEOS':<8} | {'Status':<6} | {'Delta':<7} | {'Fragment / Context':<40}")
    print("-" * 80)
    for row in res_05['candidate_details'][:10]:
        c_ts = f"{row['pause_start']:<10.2f}"
        eos = f"{str(row['pred_eos']):<8}"
        st = f"{row['match_status']:<6}"
        delta = f"{row['delta_s']:<7.3f}" if row['delta_s'] is not None else "N/A    "
        frag = row['fragment'][:38]
        print(f" {c_ts} | {eos} | {st} | {delta} | {frag:<40}")

    print("\n" + "=" * 80)
    print(" GT-LEVEL SAMPLE ROWS (First 10 GT Boundaries, tol=+/-0.5s)")
    print("=" * 80)
    print(f" {'GT Timestamp':<14} | {'Speaker Change':<14} | {'Status':<16} | {'Nearest Delta':<14} | {'EOS Vote':<8}")
    print("-" * 80)
    for row in gtl_05['gt_details'][:10]:
        gt_ts = f"{row['gt_timestamp']:<14.2f}"
        spk = f"{row['speaker_from']}->{row['speaker_to']}"
        st = f"{row['status']:<16}"
        delta = f"{row['nearest_candidate_delta_s']:<14.3f}" if row['nearest_candidate_delta_s'] is not None else "N/A (VAD)     "
        eos = str(row['nearest_candidate_eos']) if row['nearest_candidate_eos'] is not None else 'N/A'
        print(f" {gt_ts} | {spk:<14} | {st} | {delta} | {eos:<8}")

    # 7. Save structured output JSON
    output_data = {
        'meeting_id': MEETING_ID,
        'audio_file': r"C:\Users\shrut\ami-corpus-data\amicorpus\ES2002a\audio\ES2002a.Mix-Headset.wav",
        'methodology_note': (
            'Recall and FNR use GT-centric nearest-neighbor matching (evaluate_gt_level). '
            'Precision and FPR use candidate-centric matching (evaluate_candidates). '
            'Candidate-centric recall is NOT reported — it is non-monotone and mathematically invalid.'
        ),
        'summary': {
            'total_da_segments': len(da_segments),
            'backchannel_da_count': bck_count,
            'primary_gt_turn_boundaries_filtered': len(gt_filtered),
            'baseline_gt_turn_boundaries_unfiltered': len(gt_unfiltered),
            'total_pipeline_candidates': len(pipeline_candidates)
        },
        'primary_evaluation_filtered': eval_results_primary,
        'baseline_evaluation_unfiltered': eval_results_unfiltered,
        'gt_level_evaluation_filtered': gt_level_primary,
        'gt_level_evaluation_unfiltered': gt_level_unfiltered,
        'gt_boundaries_sample': gt_filtered[:20]
    }

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f" SUCCESS: Evaluation results saved to {OUTPUT_JSON_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
