"""
detection/fusion.py

Combines the semantic judgment, pause duration, speaker change, and prosodic
features into a single confidence score for turn-boundary detection.

Phase 1, Task 3 & Phase 3, Task 2 per PRD sec 2.2, sec 2.5, and sec 4 (weighted scoring).
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from detection.prosody import get_prosody_completion_value

# ==============================================================================
# CONFIGURATION & WEIGHTS
# ==============================================================================

# Master toggle for 3-signal prosodic fusion
USE_PROSODY = True

# 2-Signal Tuned Weights (Phase 3 Baseline)
WEIGHT_SEMANTIC_2SIG = 0.30
WEIGHT_PAUSE_2SIG = 0.25
WEIGHT_SPEAKER_CHANGE_2SIG = 0.45
DECISION_THRESHOLD_2SIG = 0.55

# 3-Signal Tuned Weights (Phase 3 Prosody Ablation)
WEIGHT_PROSODY = 0.20
WEIGHT_SEMANTIC_3SIG = 0.20
WEIGHT_PAUSE_3SIG = 0.20
WEIGHT_SPEAKER_CHANGE_3SIG = 0.40
DECISION_THRESHOLD_3SIG = 0.55

# Active defaults (match USE_PROSODY setting)
WEIGHT_SEMANTIC = WEIGHT_SEMANTIC_3SIG if USE_PROSODY else WEIGHT_SEMANTIC_2SIG
WEIGHT_PAUSE = WEIGHT_PAUSE_3SIG if USE_PROSODY else WEIGHT_PAUSE_2SIG
WEIGHT_SPEAKER_CHANGE = WEIGHT_SPEAKER_CHANGE_3SIG if USE_PROSODY else WEIGHT_SPEAKER_CHANGE_2SIG
DECISION_THRESHOLD = DECISION_THRESHOLD_3SIG if USE_PROSODY else DECISION_THRESHOLD_2SIG

# Pause mapping constants (PRD sec 1.1)
PAUSE_FLOOR_S = 0.3
PAUSE_CEILING_S = 3.0


@dataclass
class FusionResult:
    is_end_of_speech: bool
    confidence: float
    contributing_signals: Dict[str, Any]


def get_semantic_value(label: str, conf: float, source: str) -> float:
    """
    Map semantic label and confidence to a 0.0 - 1.0 scale.
    """
    if source == "degraded":
        # Fallback guess. Neutral/low prior so it doesn't push decision alone.
        return 0.5
    
    if label == "complete":
        return conf
    else:
        # Confident incomplete pushes value towards 0.
        return 1.0 - conf


def get_pause_value(duration_s: float) -> float:
    """
    Monotonic mapping for pause duration. Capped linear scale from floor to ceiling.
    """
    if duration_s <= PAUSE_FLOOR_S:
        return 0.0
    if duration_s >= PAUSE_CEILING_S:
        return 1.0
    return (duration_s - PAUSE_FLOOR_S) / (PAUSE_CEILING_S - PAUSE_FLOOR_S)


def fuse(
    candidate: dict,
    use_prosody: Optional[bool] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
) -> FusionResult:
    """
    Combines signals into a final end-of-speech decision.

    Parameters
    ----------
    candidate : dict
        Candidate pause data including semantic, pause, speaker, and prosody info.
    use_prosody : bool, optional
        Explicit toggle to enable/disable prosody for this call (defaults to global USE_PROSODY).
    weights : dict, optional
        Custom weight dictionary with keys {'semantic', 'pause', 'speaker_change', 'prosody'}.
    threshold : float, optional
        Custom decision threshold (defaults to corresponding baseline/3sig threshold).
    """
    enable_prosody = USE_PROSODY if use_prosody is None else use_prosody

    # 1. Determine weights and threshold
    if weights is not None:
        w_sem = weights.get("semantic", 0.25)
        w_pause = weights.get("pause", 0.20)
        w_spk = weights.get("speaker_change", 0.40)
        w_pros = weights.get("prosody", 0.15) if enable_prosody else 0.0
        thresh = threshold if threshold is not None else DECISION_THRESHOLD
    elif enable_prosody:
        w_sem = WEIGHT_SEMANTIC_3SIG
        w_pause = WEIGHT_PAUSE_3SIG
        w_spk = WEIGHT_SPEAKER_CHANGE_3SIG
        w_pros = WEIGHT_PROSODY
        thresh = threshold if threshold is not None else DECISION_THRESHOLD_3SIG
    else:
        w_sem = WEIGHT_SEMANTIC_2SIG
        w_pause = WEIGHT_PAUSE_2SIG
        w_spk = WEIGHT_SPEAKER_CHANGE_2SIG
        w_pros = 0.0
        thresh = threshold if threshold is not None else DECISION_THRESHOLD_2SIG

    # 2. Semantic Signal
    semantic_label = candidate.get("semantic_label", "incomplete")
    semantic_conf = candidate.get("semantic_confidence", 0.0)
    semantic_source = candidate.get("semantic_source", "unknown")
    raw_semantic_val = get_semantic_value(semantic_label, semantic_conf, semantic_source)
    semantic_contrib = raw_semantic_val * w_sem

    # 3. Pause Signal
    pause_duration = candidate.get("pause_duration_s", 0.0)
    raw_pause_val = get_pause_value(pause_duration)
    pause_contrib = raw_pause_val * w_pause

    # 4. Speaker Change Signal
    speaker_changed = candidate.get("speaker_changed", False)
    raw_speaker_val = 1.0 if speaker_changed else 0.0
    speaker_contrib = raw_speaker_val * w_spk

    # 5. Prosody Signal (Fusion Signal 3)
    prosody_data = candidate.get("prosody", {})
    raw_prosody_val = get_prosody_completion_value(prosody_data)
    prosody_contrib = raw_prosody_val * w_pros

    # Total weighted confidence
    total_confidence = semantic_contrib + pause_contrib + speaker_contrib + prosody_contrib
    is_eos = total_confidence >= thresh

    # Extract prosody fields safely for inspectability
    if isinstance(prosody_data, dict):
        p_avail = prosody_data.get("is_available", False)
        p_status = prosody_data.get("status", "unavailable")
        p_slope = prosody_data.get("pitch_slope")
        i_slope = prosody_data.get("intensity_slope")
        rel_dur = prosody_data.get("final_syllable_relative_duration")
    else:
        p_avail = getattr(prosody_data, "is_available", False)
        p_status = getattr(prosody_data, "status", "unavailable")
        p_slope = getattr(prosody_data, "pitch_slope", None)
        i_slope = getattr(prosody_data, "intensity_slope", None)
        rel_dur = getattr(prosody_data, "final_syllable_relative_duration", None)

    signals_breakdown = {
        "semantic": {
            "label": semantic_label,
            "source": semantic_source,
            "raw_confidence": semantic_conf,
            "mapped_value": raw_semantic_val,
            "weighted_contribution": semantic_contrib,
            "weight": w_sem
        },
        "pause": {
            "duration_s": pause_duration,
            "mapped_value": raw_pause_val,
            "weighted_contribution": pause_contrib,
            "weight": w_pause
        },
        "speaker_change": {
            "changed": speaker_changed,
            "mapped_value": raw_speaker_val,
            "weighted_contribution": speaker_contrib,
            "weight": w_spk
        },
        "prosody": {
            "is_available": p_avail,
            "status": p_status,
            "pitch_slope": p_slope,
            "intensity_slope": i_slope,
            "final_syllable_relative_duration": rel_dur,
            "mapped_value": raw_prosody_val,
            "weighted_contribution": prosody_contrib,
            "weight": w_pros
        }
    }

    return FusionResult(
        is_end_of_speech=is_eos,
        confidence=total_confidence,
        contributing_signals=signals_breakdown
    )


if __name__ == "__main__":
    input_path = "eval/pause_candidates_with_prosody.json"
    output_path = "eval/fusion_result.json"
    
    print(f"Loading candidates from {input_path}...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except FileNotFoundError:
        input_path = "eval/pause_candidates_result.json"
        print(f"Candidates with prosody not found, falling back to {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    
    results = []
    eos_count = 0
    total_conf = 0.0
    
    print(f"{'Pause(s)':<9} | {'Spkr':<5} | {'Semantic (Label/Src)':<24} | {'Prosody (p/i)':<18} | {'Conf':<6} | {'Decision'}")
    print("-" * 80)
    
    for c in candidates:
        fusion_res = fuse(c)
        results.append({
            "candidate": c,
            "fusion": {
                "is_end_of_speech": fusion_res.is_end_of_speech,
                "confidence": fusion_res.confidence,
                "contributing_signals": fusion_res.contributing_signals
            }
        })
        
        if fusion_res.is_end_of_speech:
            eos_count += 1
        total_conf += fusion_res.confidence
        
        # Print table row
        p_dur = f"{c.get('pause_duration_s', 0):.2f}"
        spkr = "Yes" if c.get("speaker_changed") else "No"
        sem_str = f"{c.get('semantic_label')} ({c.get('semantic_source')})"
        
        p_info = fusion_res.contributing_signals["prosody"]
        if p_info["is_available"] and p_info["pitch_slope"] is not None:
            pros_str = f"{p_info['pitch_slope']:+.0f}Hz, {p_info['intensity_slope']:+.0f}dB"
        else:
            pros_str = f"[{p_info['status']}]"
            
        conf_str = f"{fusion_res.confidence:.2f}"
        decision = "EOS" if fusion_res.is_end_of_speech else "Cont"
        
        print(f"{p_dur:<9} | {spkr:<5} | {sem_str:<24} | {pros_str:<18} | {conf_str:<6} | {decision}")
        
    print("-" * 80)
    print("SUMMARY STATS:")
    print(f"Total candidates: {len(candidates)}")
    print(f"Resolved to EOS : {eos_count}")
    print(f"Resolved to Cont: {len(candidates) - eos_count}")
    print(f"Avg Confidence  : {total_conf / len(candidates):.3f}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved fused results to {output_path}")
