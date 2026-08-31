import json
import sys
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.gt_matching import (
    parse_nxt_da_segments,
    derive_ground_truth_boundaries,
    evaluate_candidates,
    evaluate_gt_level,
    MEETING_ID
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = WORKSPACE_ROOT / "eval" / "pause_candidates_result.json"

def main():
    print("=" * 80)
    print(" SILENCE THRESHOLD BASELINE EVALUATION (300/500/700ms)")
    print("=" * 80)

    # 1. Load GT boundaries
    da_segments = parse_nxt_da_segments(MEETING_ID)
    gt_filtered = derive_ground_truth_boundaries(da_segments, exclude_backchannels=True)
    print(f"Loaded {len(gt_filtered)} primary ground truth boundaries.")

    # 2. Load candidates
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)
    print(f"Loaded {len(raw_candidates)} candidate pause points.")

    thresholds = [0.3, 0.5, 0.7]
    tol = 0.5

    for th in thresholds:
        print("\n" + "=" * 80)
        print(f" EVALUATING SILENCE-ONLY THRESHOLD = {int(th*1000)}ms (tol = +/-{tol}s)")
        print("=" * 80)

        # Generate predictions
        fused_candidates = []
        for c in raw_candidates:
            # Check pause duration
            is_eos = (c["pause_duration_s"] >= th)
            fused_candidates.append({
                "candidate": c,
                "fusion": {
                    "is_end_of_speech": is_eos,
                    "confidence": 1.0 if is_eos else 0.0
                }
            })

        # Evaluate
        gtl = evaluate_gt_level(fused_candidates, gt_filtered, tol)
        cres = evaluate_candidates(fused_candidates, gt_filtered, tol)

        m = cres["metrics"]
        gm = gtl["metrics"]
        cm = cres["confusion_matrix"]

        print(f"Total Candidates:   {len(fused_candidates)}")
        print(f"TP:                 {cm['TP']}")
        print(f"FP:                 {cm['FP']}")
        print(f"FN:                 {cm['FN']}")
        print(f"TN:                 {cm['TN']}")
        print("-" * 80)
        print(f"GT-level TP (hits): {gtl['gt_tp']}")
        print(f"GT-level FN (miss): {gtl['gt_fn_total']}")
        print(f"  - FN_FUSION:      {gtl['gt_fn_fusion']}")
        print(f"  - FN_VAD:         {gtl['gt_fn_vad']}")
        print("-" * 80)
        print(f"Recall (GT):        {gm['recall_gt_centric']*100:.2f}%")
        print(f"FNR:                {gm['fnr_gt_centric']*100:.2f}%")
        print(f"Precision:          {m['precision']*100:.2f}%")
        print(f"F1 Score:           {gm['f1_gt_centric']:.4f}")
        print(f"FPR (Early):        {m['false_positive_rate_early_cutoff']*100:.2f}%")
        print(f"Accuracy:           {m['accuracy']*100:.2f}%")

        # Print first 10 sample alignments
        print("\nSAMPLE ALIGNMENTS (First 10 Candidates):")
        print(f" {'PauseStart':<10} | {'PredEOS':<8} | {'Status':<6} | {'Delta':<7} | {'Fragment / Context':<40}")
        print("-" * 80)
        for row in cres['candidate_details'][:10]:
            c_ts = f"{row['pause_start']:<10.2f}"
            eos = f"{str(row['pred_eos']):<8}"
            st = f"{row['match_status']:<6}"
            delta = f"{row['delta_s']:<7.3f}" if row['delta_s'] is not None else "N/A    "
            frag = row['fragment'][:38]
            print(f" {c_ts} | {eos} | {st} | {delta} | {frag:<40}")

if __name__ == "__main__":
    main()
