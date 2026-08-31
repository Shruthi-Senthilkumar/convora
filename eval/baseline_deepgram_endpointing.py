import json
import string
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
TRANSCRIBE_PATH = WORKSPACE_ROOT / "eval" / "transcribe_batch_test_result.json"
CANDIDATES_PATH = WORKSPACE_ROOT / "eval" / "pause_candidates_result.json"

def get_deepgram_sentence_ends():
    """
    Parse the punctuated transcript and align it with the words list.
    Returns a set of word indices that end a sentence (ending with '.', '?', '!').
    """
    with open(TRANSCRIBE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript = data["transcript"]
    words = data["words"]
    tokens = transcript.split()

    w_idx = 0
    sentence_end_indices = set()

    for tok in tokens:
        if w_idx >= len(words):
            break
        clean_tok = tok.lower().translate(str.maketrans("", "", string.punctuation))
        
        # Lookahead up to 5 words to keep alignment robust
        matched = False
        for offset in range(5):
            if w_idx + offset >= len(words):
                break
            clean_word = words[w_idx + offset]["word"].lower().translate(str.maketrans("", "", string.punctuation))
            if clean_tok == clean_word or clean_word in clean_tok or clean_tok in clean_word:
                w_idx += offset
                matched = True
                break
        
        if matched:
            stripped_tok = tok.rstrip(')"\'')
            if stripped_tok.endswith((".", "?", "!")):
                sentence_end_indices.add(w_idx)
            w_idx += 1
        else:
            pass

    return sentence_end_indices

def main():
    print("=" * 80)
    print(" DEEPGRAM NATIVE ENDPOINTING BASELINE EVALUATION")
    print("=" * 80)

    # 1. Load GT boundaries
    da_segments = parse_nxt_da_segments(MEETING_ID)
    gt_filtered = derive_ground_truth_boundaries(da_segments, exclude_backchannels=True)
    print(f"Loaded {len(gt_filtered)} primary ground truth boundaries.")

    # 2. Get sentence ends
    sentence_end_indices = get_deepgram_sentence_ends()
    print(f"Identified {len(sentence_end_indices)} Deepgram sentence boundaries.")

    # 3. Load baseline candidates
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)

    # 4. Generate predictions
    fused_candidates = []
    for c in raw_candidates:
        wi = c["word_index"]
        is_eos = (wi in sentence_end_indices)
        fused_candidates.append({
            "candidate": c,
            "fusion": {
                "is_end_of_speech": is_eos,
                "confidence": 1.0 if is_eos else 0.0
            }
        })

    # 5. Evaluate at +/-0.5s tolerance
    tol = 0.5
    gtl = evaluate_gt_level(fused_candidates, gt_filtered, tol)
    cres = evaluate_candidates(fused_candidates, gt_filtered, tol)

    m = cres["metrics"]
    gm = gtl["metrics"]
    cm = cres["confusion_matrix"]

    print("\n" + "=" * 80)
    print(f" RESULTS AT +/-{tol}s TOLERANCE")
    print("=" * 80)
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
    print("=" * 80)

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
