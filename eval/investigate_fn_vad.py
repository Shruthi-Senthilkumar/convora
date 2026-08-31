"""
investigate_fn_vad.py
---------------------
Deep investigation of the 113 FN_VAD boundaries — GT turn boundaries where
the pause detector generated NO candidate within +/-0.5s.

For each FN_VAD boundary this script computes:
  1. The silence gap in the Deepgram word stream at that exact timestamp
     (gap = time between end of last word before boundary and start of first
      word after boundary in the transcript).
  2. Whether Deepgram's diarization assigned different speaker IDs to the
     words immediately before and after the boundary (i.e., did Deepgram
     "see" the speaker change?).
  3. The nearest pause candidate (any distance), to see whether the trigger
     radius simply missed it or whether NO candidate exists nearby at all.
  4. The Deepgram speaker IDs in the 2s window around the boundary.

Outputs:
  - Console: summary statistics + 10 detailed example rows
  - eval/fn_vad_investigation.json: full per-boundary data
"""

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from eval.evaluate_against_ami_ground_truth import (
    parse_nxt_da_segments,
    derive_ground_truth_boundaries,
    evaluate_gt_level,
    FUSION_RESULT_PATH,
)

TRANSCRIBE_PATH = WORKSPACE_ROOT / "eval" / "transcribe_batch_test_result.json"
OUTPUT_PATH = WORKSPACE_ROOT / "eval" / "fn_vad_investigation.json"
TOL = 0.5  # primary baseline

# ── helpers ──────────────────────────────────────────────────────────────────

def silence_gap_at(words, ts, context_s=2.0):
    """
    Return the inter-word silence gap in the Deepgram word stream
    at timestamp `ts`.

    Strategy: find the last word that ends at or before `ts`, and the first
    word that starts at or after `ts`. The gap is max(0, next_start - prev_end).

    Also return the speaker IDs of those two words.
    """
    prev_word = None
    next_word = None

    for w in words:
        w_end = w["end"]
        w_start = w["start"]
        if w_end <= ts:
            if prev_word is None or w_end > prev_word["end"]:
                prev_word = w
        if w_start >= ts:
            if next_word is None or w_start < next_word["start"]:
                next_word = w

    gap = None
    if prev_word and next_word:
        gap = round(max(0.0, next_word["start"] - prev_word["end"]), 4)

    # collect all words within context_s of the boundary
    context_words = [
        w for w in words
        if abs((w["start"] + w["end"]) / 2 - ts) <= context_s
    ]
    speakers_in_window = sorted(set(w.get("speaker", -1) for w in context_words))

    return {
        "gap_s": gap,
        "prev_word": prev_word["word"] if prev_word else None,
        "prev_end": prev_word["end"] if prev_word else None,
        "prev_speaker": prev_word.get("speaker") if prev_word else None,
        "next_word": next_word["word"] if next_word else None,
        "next_start": next_word["start"] if next_word else None,
        "next_speaker": next_word.get("speaker") if next_word else None,
        "deepgram_detected_speaker_change": (
            prev_word is not None
            and next_word is not None
            and prev_word.get("speaker") != next_word.get("speaker")
        ),
        "speakers_in_2s_window": speakers_in_window,
    }


def nearest_candidate(all_candidates, ts):
    """Return (delta_s, candidate_dict) of the nearest candidate to ts."""
    if not all_candidates:
        return None, None
    nearest = min(all_candidates, key=lambda c: abs(c["candidate"]["pause_start"] - ts))
    delta = round(abs(nearest["candidate"]["pause_start"] - ts), 4)
    return delta, nearest


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print(" FN_VAD DEEP INVESTIGATION")
    print(f" Tolerance: +/-{TOL}s  |  Meeting: ES2002a")
    print("=" * 80)

    # 1. Load data
    with open(FUSION_RESULT_PATH, "r", encoding="utf-8") as f:
        pipeline_candidates = json.load(f)

    with open(TRANSCRIBE_PATH, "r", encoding="utf-8") as f:
        transcribe = json.load(f)

    words = transcribe["words"]
    print(f"Loaded {len(pipeline_candidates)} fusion candidates")
    print(f"Loaded {len(words)} Deepgram words  "
          f"(span: {words[0]['start']:.1f}s – {words[-1]['end']:.1f}s)")

    # 2. Derive GT boundaries
    da_segs = parse_nxt_da_segments("ES2002a")
    gt_filtered = derive_ground_truth_boundaries(da_segs, exclude_backchannels=True)

    # 3. Run GT-level evaluation to identify FN_VAD set
    gtl = evaluate_gt_level(pipeline_candidates, gt_filtered, TOL)
    fn_vad_boundaries = [
        row for row in gtl["gt_details"] if row["status"] == "GT_FN_VAD"
    ]
    print(f"\nGT boundaries total:  {gtl['total_gt_boundaries']}")
    print(f"GT_TP:                {gtl['gt_tp']}")
    print(f"GT_FN_FUSION:         {gtl['gt_fn_fusion']}")
    print(f"GT_FN_VAD:            {gtl['gt_fn_vad']}  <-- investigating these")

    # 4. For each FN_VAD boundary, compute diagnostics
    results = []
    for row in fn_vad_boundaries:
        g_ts = row["gt_timestamp"]

        # silence gap + Deepgram speaker info
        gap_info = silence_gap_at(words, g_ts)

        # nearest candidate (any EOS, any distance)
        delta_any, nearest_cand = nearest_candidate(pipeline_candidates, g_ts)
        nearest_cand_summary = None
        if nearest_cand:
            nearest_cand_summary = {
                "pause_start": nearest_cand["candidate"]["pause_start"],
                "pause_duration_s": nearest_cand["candidate"].get("pause_duration_s"),
                "speaker_changed_flag": nearest_cand["candidate"].get("speaker_changed"),
                "is_end_of_speech": nearest_cand["fusion"]["is_end_of_speech"],
                "delta_s": delta_any,
            }

        results.append({
            "gt_timestamp": g_ts,
            "speaker_from": row["speaker_from"],
            "speaker_to": row["speaker_to"],
            "silence_gap_s": gap_info["gap_s"],
            "prev_word": gap_info["prev_word"],
            "prev_end_s": gap_info["prev_end"],
            "prev_deepgram_speaker": gap_info["prev_speaker"],
            "next_word": gap_info["next_word"],
            "next_start_s": gap_info["next_start"],
            "next_deepgram_speaker": gap_info["next_speaker"],
            "deepgram_detected_speaker_change": gap_info["deepgram_detected_speaker_change"],
            "speakers_in_2s_window": gap_info["speakers_in_2s_window"],
            "nearest_candidate": nearest_cand_summary,
        })

    # 5. Aggregate silence gap distribution
    gaps = [r["silence_gap_s"] for r in results if r["silence_gap_s"] is not None]
    gaps_sorted = sorted(gaps)
    deepgram_saw_change = sum(1 for r in results if r["deepgram_detected_speaker_change"])
    deepgram_missed = sum(1 for r in results if not r["deepgram_detected_speaker_change"])

    buckets = {"<0.1s": 0, "0.1-0.3s": 0, "0.3-0.5s": 0, "0.5-1.0s": 0, ">1.0s": 0}
    for g in gaps:
        if g < 0.1:
            buckets["<0.1s"] += 1
        elif g < 0.3:
            buckets["0.1-0.3s"] += 1
        elif g < 0.5:
            buckets["0.3-0.5s"] += 1
        elif g < 1.0:
            buckets["0.5-1.0s"] += 1
        else:
            buckets[">1.0s"] += 1

    # nearest-candidate distance distribution for FN_VADs
    near_dist = [r["nearest_candidate"]["delta_s"] for r in results if r["nearest_candidate"]]
    near_buckets = {"<1s": 0, "1-2s": 0, "2-5s": 0, ">5s": 0}
    for d in near_dist:
        if d < 1:
            near_buckets["<1s"] += 1
        elif d < 2:
            near_buckets["1-2s"] += 1
        elif d < 5:
            near_buckets["2-5s"] += 1
        else:
            near_buckets[">5s"] += 1

    # 6. Print summary
    print("\n" + "=" * 80)
    print(" SILENCE GAP DISTRIBUTION AT FN_VAD BOUNDARIES")
    print(" (gap = time between last Deepgram word before boundary and first after)")
    print("=" * 80)
    print(f" Total FN_VAD boundaries: {len(results)}")
    print(f" Boundaries with measurable gap: {len(gaps)}")
    if gaps:
        print(f" Min gap:    {min(gaps):.3f}s")
        print(f" Median gap: {gaps_sorted[len(gaps_sorted)//2]:.3f}s")
        print(f" Mean gap:   {sum(gaps)/len(gaps):.3f}s")
        print(f" Max gap:    {max(gaps):.3f}s")
    print()
    print(" Gap size buckets:")
    for k, v in buckets.items():
        bar = "#" * v
        pct = v / len(results) * 100 if results else 0
        print(f"   {k:>10}  {v:3d} ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 80)
    print(" DEEPGRAM DIARIZATION AT FN_VAD BOUNDARIES")
    print("=" * 80)
    pct_saw = deepgram_saw_change / len(results) * 100 if results else 0
    pct_missed = deepgram_missed / len(results) * 100 if results else 0
    print(f" Deepgram assigned DIFFERENT speaker IDs across boundary: "
          f"{deepgram_saw_change} ({pct_saw:.1f}%)")
    print(f" Deepgram assigned SAME speaker ID across boundary:      "
          f"{deepgram_missed} ({pct_missed:.1f}%)")
    print()
    print(" Interpretation:")
    print(f"   -> If Deepgram SAME ({pct_missed:.0f}%): diarization failure - it didn't detect")
    print(f"     the turn change, so speaker_changed=False, no trigger fired.")
    print(f"   -> If Deepgram DIFFERENT ({pct_saw:.0f}%): diarization OK but pause_duration_s")
    print(f"     was too short (< threshold) to generate a VAD candidate.")

    print("\n" + "=" * 80)
    print(" NEAREST CANDIDATE DISTANCE (how far is the closest candidate)")
    print(" [confirms no candidate was within the +/-0.5s window]")
    print("=" * 80)
    for k, v in near_buckets.items():
        bar = "#" * v
        pct = v / len(results) * 100 if results else 0
        print(f"   {k:>6}  {v:3d} ({pct:5.1f}%)  {bar}")

    # 7. Print 10 real examples
    print("\n" + "=" * 80)
    print(" 10 REAL FN_VAD EXAMPLES (sorted by GT timestamp)")
    print("=" * 80)
    hdr = (f" {'GT ts':>7} | {'AMI chg':>7} | {'Gap(s)':>6} | "
           f"{'DG chg':>6} | {'DGspk':>5} | "
           f"{'Prev word':>12} | {'Next word':>12} | "
           f"{'NearCand':>8} | {'NearDelta':>9}")
    print(hdr)
    print("-" * 120)

    sample = sorted(results, key=lambda r: r["gt_timestamp"])[:10]
    for r in sample:
        gt_ts = f"{r['gt_timestamp']:7.2f}"
        ami_chg = f"{r['speaker_from']}->{r['speaker_to']}"
        gap = f"{r['silence_gap_s']:6.3f}" if r['silence_gap_s'] is not None else "  N/A "
        dg_chg = "YES" if r["deepgram_detected_speaker_change"] else "NO "
        dg_spk = (f"{r['prev_deepgram_speaker']}->"
                  f"{r['next_deepgram_speaker']}")
        prev_w = (r["prev_word"] or "")[:12]
        next_w = (r["next_word"] or "")[:12]
        if r["nearest_candidate"]:
            nc = r["nearest_candidate"]
            nc_eos = "EOS=T" if nc["is_end_of_speech"] else "EOS=F"
            nc_delta = f"{nc['delta_s']:9.3f}"
        else:
            nc_eos = "NONE"
            nc_delta = "      N/A"
        print(f" {gt_ts} | {ami_chg:>7} | {gap} | {dg_chg} | "
              f"{dg_spk:>5} | {prev_w:>12} | {next_w:>12} | "
              f"{nc_eos:>8} | {nc_delta}")

    print()
    print(" Column guide:")
    print("   GT ts     = AMI ground-truth boundary timestamp")
    print("   AMI chg   = speaker transition per AMI annotation")
    print("   Gap(s)    = silence in Deepgram word stream at that point")
    print("   DG chg    = did Deepgram assign different speaker IDs?")
    print("   DGspk     = Deepgram speaker IDs before->after boundary")
    print("   NearCand  = EOS vote of nearest pipeline candidate (any distance)")
    print("   NearDelta = distance to that nearest candidate")

    # 8. Sub-analysis: silence gap for DG-same vs DG-different
    print("\n" + "=" * 80)
    print(" SILENCE GAP BY DIARIZATION OUTCOME")
    print("=" * 80)
    gaps_dg_same = [r["silence_gap_s"] for r in results
                    if not r["deepgram_detected_speaker_change"]
                    and r["silence_gap_s"] is not None]
    gaps_dg_diff = [r["silence_gap_s"] for r in results
                    if r["deepgram_detected_speaker_change"]
                    and r["silence_gap_s"] is not None]

    def stats(label, g_list):
        if not g_list:
            print(f" {label}: no data")
            return
        s = sorted(g_list)
        print(f" {label} (n={len(g_list)}):")
        print(f"   min={min(g_list):.3f}s  median={s[len(s)//2]:.3f}s  "
              f"mean={sum(g_list)/len(g_list):.3f}s  max={max(g_list):.3f}s")
        sub = {"<0.1s": 0, "0.1-0.3s": 0, "0.3-0.5s": 0, ">0.5s": 0}
        for g in g_list:
            if g < 0.1: sub["<0.1s"] += 1
            elif g < 0.3: sub["0.1-0.3s"] += 1
            elif g < 0.5: sub["0.3-0.5s"] += 1
            else: sub[">0.5s"] += 1
        for k, v in sub.items():
            print(f"   {k}: {v}")

    stats("Deepgram SAME speaker (diarization missed the change)", gaps_dg_same)
    print()
    stats("Deepgram DIFFERENT speaker (diarization OK, gap too small)", gaps_dg_diff)

    # 9. Save full results
    output = {
        "tolerance_s": TOL,
        "total_fn_vad": len(results),
        "summary": {
            "silence_gap": {
                "min": min(gaps) if gaps else None,
                "median": gaps_sorted[len(gaps_sorted)//2] if gaps else None,
                "mean": round(sum(gaps)/len(gaps), 4) if gaps else None,
                "max": max(gaps) if gaps else None,
                "buckets": buckets,
            },
            "deepgram_diarization": {
                "detected_speaker_change": deepgram_saw_change,
                "missed_speaker_change": deepgram_missed,
                "pct_detected": round(pct_saw, 2),
                "pct_missed": round(pct_missed, 2),
            },
            "nearest_candidate_distance_buckets": near_buckets,
        },
        "per_boundary": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
