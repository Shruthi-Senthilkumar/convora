import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from huggingface_hub import hf_hub_download

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from eval.gt_matching import (
    parse_nxt_da_segments,
    derive_ground_truth_boundaries,
    evaluate_candidates,
    evaluate_gt_level,
    MEETING_ID
)

WAV_PATH = r"C:\Users\shrut\ami-corpus-data\amicorpus\ES2002a\audio\ES2002a.Mix-Headset.wav"
CANDIDATES_PATH = WORKSPACE_ROOT / "eval" / "pause_candidates_with_prosody.json"
MODEL_DIR = WORKSPACE_ROOT / "models" / "smart_turn"
MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
OUTPUT_RESULTS_PATH = WORKSPACE_ROOT / "eval" / "smart_turn_results.json"


def ensure_model_downloaded() -> Path:
    """Ensure Smart Turn v3 ONNX model is available locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / MODEL_FILENAME
    if not local_path.exists():
        print(f"Downloading {MODEL_FILENAME} from Hugging Face (pipecat-ai/smart-turn-v3)...")
        hf_hub_download(
            repo_id="pipecat-ai/smart-turn-v3",
            filename=MODEL_FILENAME,
            local_dir=str(MODEL_DIR)
        )
    return local_path


def main():
    print("=" * 80)
    print(" PIPECAT SMART TURN V3 REFERENCE CEILING BENCHMARK")
    print("=" * 80)

    # 1. Ensure Model & Load ONNX Session
    onnx_path = ensure_model_downloaded()
    print(f"Loaded ONNX Model: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=so)
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    # 2. Load Audio
    if not os.path.exists(WAV_PATH):
        raise FileNotFoundError(f"Audio file not found at {WAV_PATH}")
    print(f"Loading meeting audio from {WAV_PATH}...")
    audio_full, sr = sf.read(WAV_PATH, dtype="float32")
    print(f"Loaded {len(audio_full)} samples ({len(audio_full)/sr:.2f}s, {sr}Hz mono)")

    # 3. Load Candidates
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)
    print(f"Loaded {len(raw_candidates)} candidate pause points.")

    # 4. Warm up ONNX session
    dummy_audio = np.random.randn(8 * 16000).astype(np.float32)
    dummy_inputs = feature_extractor(
        dummy_audio,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True
    )
    dummy_feat = np.expand_dims(dummy_inputs.input_features.squeeze(0).astype(np.float32), axis=0)
    for _ in range(5):
        session.run(None, {"input_features": dummy_feat})

    # 5. Run inference on each candidate
    print("\nRunning Smart Turn v3 inference over candidate pause points...")
    inference_latencies_ms = []
    smart_turn_scored_candidates = []

    for item in raw_candidates:
        cand = item.get("candidate", item)
        ts_pause = cand["pause_start"]

        # Preceding 8-second speech window up to pause_start
        start_sec = max(0.0, ts_pause - 8.0)
        end_sec = ts_pause
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        segment = audio_full[start_sample:end_sample]
        max_samples = 8 * 16000
        if len(segment) < max_samples:
            segment = np.pad(segment, (max_samples - len(segment), 0), mode="constant")
        elif len(segment) > max_samples:
            segment = segment[-max_samples:]

        t0 = time.perf_counter()
        inputs = feature_extractor(
            segment,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)
        outputs = session.run(None, {"input_features": input_features})
        prob = float(outputs[0][0].item())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        inference_latencies_ms.append(elapsed_ms)

        smart_turn_scored_candidates.append({
            "candidate": cand,
            "smart_turn_probability": prob,
            "latency_ms": elapsed_ms
        })

    # 6. Load Ground Truth
    da_segments = parse_nxt_da_segments(MEETING_ID)
    gt_filtered = derive_ground_truth_boundaries(da_segments, exclude_backchannels=True)
    print(f"Loaded {len(gt_filtered)} primary ground truth boundaries.")

    # 7. Evaluate across tolerance and thresholds
    tolerances = [0.3, 0.5, 0.8]
    eval_results_by_tol = {}

    for tol in tolerances:
        # Standard threshold 0.50
        fused_candidates = []
        for item in smart_turn_scored_candidates:
            prob = item["smart_turn_probability"]
            is_eos = (prob > 0.50)
            fused_candidates.append({
                "candidate": item["candidate"],
                "fusion": {
                    "is_end_of_speech": is_eos,
                    "confidence": prob,
                    "smart_turn_probability": prob
                }
            })

        cres = evaluate_candidates(fused_candidates, gt_filtered, tol)
        gtl = evaluate_gt_level(fused_candidates, gt_filtered, tol)

        eval_results_by_tol[f"tol_{tol}s"] = {
            "candidate_eval": cres,
            "gt_eval": gtl
        }

    # Focus print on primary standard (+/-0.5s tolerance, threshold 0.50)
    primary_cres = eval_results_by_tol["tol_0.5s"]["candidate_eval"]
    primary_gtl = eval_results_by_tol["tol_0.5s"]["gt_eval"]
    cm = primary_cres["confusion_matrix"]
    m = primary_cres["metrics"]
    gm = primary_gtl["metrics"]

    print("\n" + "=" * 80)
    print(" PRIMARY EVALUATION SUMMARY (tol = +/-0.50s, threshold = 0.50)")
    print("=" * 80)
    print(f"Total Candidates:             {len(smart_turn_scored_candidates)}")
    print(f"Total GT Boundaries:          {len(gt_filtered)}")
    print(f"Confusion Matrix:             TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}")
    print("-" * 80)
    print(f"GT-level TP (Hits):           {primary_gtl['gt_tp']}")
    print(f"GT-level FN (Misses):         {primary_gtl['gt_fn_total']}")
    print(f"  - FN_FUSION:                {primary_gtl['gt_fn_fusion']}")
    print(f"  - FN_VAD:                   {primary_gtl['gt_fn_vad']}")
    print("-" * 80)
    print(f"Recall (GT-centric):          {gm['recall_gt_centric']*100:.2f}%")
    print(f"False Negative Rate (FNR):    {gm['fnr_gt_centric']*100:.2f}%")
    print(f"Precision (Candidate-level):  {m['precision']*100:.2f}%")
    print(f"False Positive Rate (FPR):    {m['false_positive_rate_early_cutoff']*100:.2f}%")
    print(f"F1 Score:                     {gm['f1_gt_centric']:.4f}")
    print(f"Overall Accuracy:             {m['accuracy']*100:.2f}%")

    print("\n" + "=" * 80)
    print(" INFERENCE LATENCY PROFILE (Intel Core i3-1315U, 4 ONNX threads)")
    print("=" * 80)
    print(f"Mean Latency:                 {np.mean(inference_latencies_ms):.2f} ms")
    print(f"Median (p50):                 {np.median(inference_latencies_ms):.2f} ms")
    print(f"p90 Latency:                  {np.percentile(inference_latencies_ms, 90):.2f} ms")
    print(f"p95 Latency:                  {np.percentile(inference_latencies_ms, 95):.2f} ms")
    print(f"p99 Latency:                  {np.percentile(inference_latencies_ms, 99):.2f} ms")
    print(f"Min / Max:                    {np.min(inference_latencies_ms):.2f} ms / {np.max(inference_latencies_ms):.2f} ms")

    # Sample alignments
    print("\n" + "=" * 80)
    print(" SAMPLE ALIGNMENTS (First 10 Candidates)")
    print("=" * 80)
    print(f" {'PauseStart':<10} | {'Prob':<6} | {'PredEOS':<8} | {'Status':<6} | {'Delta':<7} | {'Fragment / Context':<36}")
    print("-" * 80)
    for i, row in enumerate(primary_cres["candidate_details"][:10]):
        cand_meta = smart_turn_scored_candidates[i]
        prob = cand_meta["smart_turn_probability"]
        c_ts = f"{row['pause_start']:<10.2f}"
        p_str = f"{prob:<6.3f}"
        eos = f"{str(row['pred_eos']):<8}"
        st = f"{row['match_status']:<6}"
        delta = f"{row['delta_s']:<7.3f}" if row["delta_s"] is not None else "N/A    "
        frag = row["fragment"][:34]
        print(f" {c_ts} | {p_str} | {eos} | {st} | {delta} | {frag:<36}")

    # Save full results
    save_data = {
        "model": "pipecat-ai/smart-turn-v3.2-cpu",
        "onnx_model_file": MODEL_FILENAME,
        "meeting_id": MEETING_ID,
        "audio_file": WAV_PATH,
        "num_candidates": len(smart_turn_scored_candidates),
        "num_gt_boundaries": len(gt_filtered),
        "latency_stats_ms": {
            "mean": float(np.mean(inference_latencies_ms)),
            "median_p50": float(np.median(inference_latencies_ms)),
            "p90": float(np.percentile(inference_latencies_ms, 90)),
            "p95": float(np.percentile(inference_latencies_ms, 95)),
            "p99": float(np.percentile(inference_latencies_ms, 99)),
            "min": float(np.min(inference_latencies_ms)),
            "max": float(np.max(inference_latencies_ms)),
        },
        "tolerances": eval_results_by_tol,
        "scored_candidates": smart_turn_scored_candidates
    }

    with open(OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved benchmark results to {OUTPUT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
