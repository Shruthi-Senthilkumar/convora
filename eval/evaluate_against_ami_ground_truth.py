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
import zipfile
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = WORKSPACE_ROOT / "eval"
FUSION_RESULT_PATH = EVAL_DIR / "fusion_result.json"
OUTPUT_JSON_PATH = EVAL_DIR / "ami_ground_truth_evaluation.json"

AMI_DATA_DIR = Path(r"C:\Users\shrut\ami-corpus-data")
AMI_ZIP_PATH = AMI_DATA_DIR / "ami_public_manual_1.6.2.zip"
AMI_UNZIPPED_DIR = AMI_DATA_DIR / "ami_public_manual_1.6.2"

MEETING_ID = "ES2002a"
SPEAKERS = ["A", "B", "C", "D"]

# DA Aspect Map
DA_TYPES_MAP = {
    "ami_da_1": ("bck", "Backchannel"),
    "ami_da_2": ("stl", "Stall"),
    "ami_da_3": ("fra", "Fragment"),
    "ami_da_4": ("inf", "Inform"),
    "ami_da_5": ("el.inf", "Elicit-Inform"),
    "ami_da_6": ("sug", "Suggest"),
    "ami_da_7": ("off", "Offer"),
    "ami_da_8": ("el.sug", "Elicit-Offer"),
    "ami_da_9": ("ass", "Assess"),
    "ami_da_11": ("el.ass", "Elicit-Assess"),
    "ami_da_12": ("und", "Comment-Understand"),
    "ami_da_13": ("el.und", "Elicit-Understand"),
    "ami_da_14": ("be.pos", "Be-Positive"),
    "ami_da_15": ("be.neg", "Be-Negative"),
    "ami_da_16": ("oth", "Other")
}


def ensure_ami_data():
    """Ensure AMI NXT XML files are extracted and accessible."""
    if not AMI_UNZIPPED_DIR.exists():
        if not AMI_ZIP_PATH.exists():
            print(f"ERROR: Cannot find AMI zip at {AMI_ZIP_PATH}")
            sys.exit(1)
        print(f"Unzipping {AMI_ZIP_PATH} to {AMI_UNZIPPED_DIR}...")
        with zipfile.ZipFile(AMI_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(AMI_UNZIPPED_DIR)
        print("Unzip complete.")


def parse_nxt_da_segments(meeting_id=MEETING_ID):
    """
    Parse NXT XML words and dialogue acts for the given meeting ID across all 4 speakers.
    Returns sorted list of DA segment dictionaries.
    """
    ensure_ami_data()
    all_da_segments = []

    for speaker in SPEAKERS:
        words_file = AMI_UNZIPPED_DIR / "words" / f"{meeting_id}.{speaker}.words.xml"
        da_file = AMI_UNZIPPED_DIR / "dialogueActs" / f"{meeting_id}.{speaker}.dialog-act.xml"

        if not words_file.exists() or not da_file.exists():
            print(f"ERROR: Missing annotation files for speaker {speaker}: {words_file} or {da_file}")
            sys.exit(1)

        # 1. Parse words.xml into indexed list & ID map
        w_tree = ET.parse(words_file)
        w_root = w_tree.getroot()
        elem_list = []
        elem_map = {}

        for idx, el in enumerate(w_root):
            wid = el.attrib.get('{http://nite.sourceforge.net/}id')
            st = el.attrib.get('starttime')
            et = el.attrib.get('endtime')
            st_f = float(st) if st is not None else None
            et_f = float(et) if et is not None else None
            text = el.text or ''
            elem_info = {'id': wid, 'st': st_f, 'et': et_f, 'text': text, 'idx': idx}
            elem_list.append(elem_info)
            elem_map[wid] = elem_info

        # 2. Parse dialog-act.xml and resolve word ID ranges
        da_tree = ET.parse(da_file)
        da_root = da_tree.getroot()

        for dact in da_root.findall('dact'):
            da_id = dact.attrib.get('{http://nite.sourceforge.net/}id')
            child = dact.find('{http://nite.sourceforge.net/}child')
            if child is None:
                continue

            href = child.attrib.get('href', '')
            m_range = re.search(r'id\(([^)]+)\)\.\.id\(([^)]+)\)', href)
            m_single = re.search(r'id\(([^)]+)\)', href)

            if m_range:
                w_start_id, w_end_id = m_range.group(1), m_range.group(2)
                idx1 = elem_map[w_start_id]['idx']
                idx2 = elem_map[w_end_id]['idx']
                sub = elem_list[idx1:idx2+1]
            elif m_single:
                w_id = m_single.group(1)
                sub = [elem_map[w_id]]
            else:
                sub = []

            st_vals = [e['st'] for e in sub if e['st'] is not None]
            et_vals = [e['et'] for e in sub if e['et'] is not None]
            words_text = ' '.join([e['text'] for e in sub if e['text']])

            da_type = None
            ptr = dact.find('{http://nite.sourceforge.net/}pointer')
            if ptr is not None:
                raw_da = ptr.attrib.get('href', '').split('#')[-1]
                m_da = re.search(r'ami_da_\d+', raw_da)
                if m_da:
                    da_type = m_da.group(0)

            if st_vals and et_vals:
                all_da_segments.append({
                    'da_id': da_id,
                    'speaker': speaker,
                    'start_time': min(st_vals),
                    'end_time': max(et_vals),
                    'text': words_text,
                    'da_type': da_type,
                    'da_type_label': DA_TYPES_MAP.get(da_type, ('unk', 'Unknown'))[0]
                })

    all_da_segments.sort(key=lambda x: (x['start_time'], x['end_time']))
    return all_da_segments


def derive_ground_truth_boundaries(da_segments, exclude_backchannels=True):
    """
    Derive ground-truth turn boundaries from sorted DA segments.
    If exclude_backchannels=True (primary rule), ami_da_1 (Backchannels) do not transfer floor.
    """
    if exclude_backchannels:
        eval_segments = [d for d in da_segments if d['da_type'] != 'ami_da_1']
    else:
        eval_segments = da_segments

    gt_boundaries = []
    for i in range(len(eval_segments) - 1):
        curr = eval_segments[i]
        nxt = eval_segments[i+1]
        if curr['speaker'] != nxt['speaker']:
            gt_boundaries.append({
                'timestamp': curr['end_time'],
                'speaker_from': curr['speaker'],
                'speaker_to': nxt['speaker'],
                'da_id': curr['da_id'],
                'da_type': curr['da_type'],
                'text': curr['text'],
                'next_start_time': nxt['start_time']
            })

    return gt_boundaries


def evaluate_candidates(pipeline_candidates, gt_boundaries, tol):
    """
    Evaluate candidate-level predictions against ground truth boundaries within tolerance window tol.

    NOTE on Precision / FPR:
    These are correctly computed candidate-centrically: each candidate checks whether ANY GT
    boundary falls within +/-tol. As tolerance widens, FP can only decrease (never increase),
    so Precision is monotonically non-decreasing.

    NOTE on Recall / FNR (REMOVED from this function):
    Candidate-centric recall (TP / (TP+FN)) is NON-MONOTONE as tolerance widens because
    EOS=False candidates near a GT boundary shift TN→FN, inflating the denominator faster
    than TP grows. Recall and FNR are now computed GT-centrically in evaluate_gt_level().
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    cand_details = []

    for item in pipeline_candidates:
        cand = item['candidate']
        pred_eos = item['fusion']['is_end_of_speech']
        c_ts = cand['pause_start']

        # Find closest GT boundary within tolerance
        matching_gts = [g for g in gt_boundaries if abs(g['timestamp'] - c_ts) <= tol]
        gt_is_turn = len(matching_gts) > 0
        closest_gt = min(matching_gts, key=lambda g: abs(g['timestamp'] - c_ts)) if gt_is_turn else None

        if pred_eos and gt_is_turn:
            tp += 1
            status = "TP"
        elif pred_eos and not gt_is_turn:
            fp += 1
            status = "FP"
        elif not pred_eos and gt_is_turn:
            fn += 1
            status = "FN"
        else:
            tn += 1
            status = "TN"

        cand_details.append({
            'pause_start': c_ts,
            'pause_end': cand.get('pause_end'),
            'pause_duration_s': cand.get('pause_duration_s'),
            'speaker': cand.get('speaker'),
            'pred_eos': pred_eos,
            'confidence': item['fusion'].get('confidence'),
            'gt_is_turn': gt_is_turn,
            'match_status': status,
            'fragment': cand.get('fragment', ''),
            'closest_gt_timestamp': closest_gt['timestamp'] if closest_gt else None,
            'closest_gt_speaker_change': f"{closest_gt['speaker_from']}->{closest_gt['speaker_to']}" if closest_gt else None,
            'delta_s': round(abs(closest_gt['timestamp'] - c_ts), 4) if closest_gt else None
        })

    total = len(pipeline_candidates)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Early cutoff rate on non-turn pauses
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        'tolerance_s': tol,
        'total_candidates': total,
        'positives_predicted': tp + fp,
        'negatives_predicted': tn + fn,
        'confusion_matrix': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn},
        'metrics': {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'false_positive_rate_early_cutoff': round(fpr, 4),
        },
        'candidate_details': cand_details
    }


def evaluate_gt_level(pipeline_candidates, gt_boundaries, tol):
    """
    GT-centric evaluation: for each ground-truth turn boundary, find its single NEAREST
    candidate within +/-tol and check whether that candidate predicted EOS=True.

    This is the CORRECT way to compute recall and FNR:
    - Recall = fraction of GT boundaries whose nearest candidate predicted EOS=True.
    - FNR    = fraction of GT boundaries whose nearest candidate predicted EOS=False (or had
               no candidate within tol at all — VAD gap).

    Monotonicity guarantee: as tolerance widens, every GT boundary that was matched at a
    narrower window is still matched (same or closer candidate), so TP can only stay the same
    or increase. Recall is therefore strictly non-decreasing.

    Two FN breakdown sub-types are reported:
    - FN_FUSION: a candidate exists within tol but voted EOS=False (pipeline logic error).
    - FN_VAD:    no candidate exists within tol at all (VAD/pause-detector missed the pause).
    """
    all_candidates = [
        (c['candidate']['pause_start'], c['fusion']['is_end_of_speech'])
        for c in pipeline_candidates
    ]
    eos_true_timestamps = [ts for ts, eos in all_candidates if eos]

    gt_tp = 0            # GT boundaries with nearest candidate EOS=True
    gt_fn_fusion = 0     # GT boundaries with nearest candidate EOS=False
    gt_fn_vad = 0        # GT boundaries with no candidate within tol
    matched_eos_indices = set()
    gt_details = []

    for g in gt_boundaries:
        g_ts = g['timestamp']

        # All candidates (EOS=True or False) within tolerance
        cands_in_tol = [
            (abs(ts - g_ts), idx, eos)
            for idx, (ts, eos) in enumerate(all_candidates)
            if abs(ts - g_ts) <= tol
        ]

        if cands_in_tol:
            # Pick single nearest candidate
            cands_in_tol.sort()
            nearest_delta, nearest_idx, nearest_eos = cands_in_tol[0]
            if nearest_eos:
                gt_tp += 1
                matched_eos_indices.add(nearest_idx)
                status = 'GT_TP'
            else:
                gt_fn_fusion += 1
                status = 'GT_FN_FUSION'
        else:
            gt_fn_vad += 1
            nearest_delta = None
            nearest_idx = None
            nearest_eos = None
            status = 'GT_FN_VAD'

        gt_details.append({
            'gt_timestamp': g_ts,
            'speaker_from': g['speaker_from'],
            'speaker_to': g['speaker_to'],
            'status': status,
            'nearest_candidate_delta_s': round(nearest_delta, 4) if nearest_delta is not None else None,
            'nearest_candidate_eos': nearest_eos
        })

    # EOS=True candidates that were NOT matched as the nearest-to any GT
    gt_fn_total = gt_fn_fusion + gt_fn_vad
    total_gt = len(gt_boundaries)
    recall = gt_tp / total_gt if total_gt > 0 else 0.0
    fnr = gt_fn_total / total_gt if total_gt > 0 else 0.0

    # Precision: among EOS=True candidates, how many were the nearest match to some GT?
    total_eos_true = len(eos_true_timestamps)
    gt_precision_tp = len(matched_eos_indices)
    gt_fp = total_eos_true - gt_precision_tp  # EOS=True candidates not matched as nearest to any GT
    precision = gt_precision_tp / total_eos_true if total_eos_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tolerance_s': tol,
        'total_gt_boundaries': total_gt,
        'gt_tp': gt_tp,
        'gt_fn_total': gt_fn_total,
        'gt_fn_fusion': gt_fn_fusion,
        'gt_fn_vad': gt_fn_vad,
        'gt_fp_eos_unmatched': gt_fp,
        'total_eos_true_candidates': total_eos_true,
        'metrics': {
            'recall_gt_centric': round(recall, 4),
            'fnr_gt_centric': round(fnr, 4),
            'precision_gt_centric': round(precision, 4),
            'f1_gt_centric': round(f1, 4),
        },
        'gt_details': gt_details
    }


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
    print(f"  Overall Decision Accuracy:    {m05['accuracy'] * 100:.2f}%  [WARNING: misleading — see FNR above]")
    print(f"  Precision (GT-centric):       {gm05['precision_gt_centric'] * 100:.2f}%")
    print(f"  Recall    (GT-centric):       {gm05['recall_gt_centric'] * 100:.2f}%")
    print(f"  F1 Score  (GT-centric):       {gm05['f1_gt_centric']:.4f}")
    print(f"  False Positive Rate (FPR):    {m05['false_positive_rate_early_cutoff'] * 100:.2f}% (Early Cutoff Rate on non-turn pauses)")
    print(f"  False Negative Rate (FNR):    {gm05['fnr_gt_centric'] * 100:.2f}% (GT-centric — MONOTONE CORRECT)")

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
