"""
detection/fusion.py

Combines the semantic judgment, pause duration, and speaker change signals
into a single confidence score for turn-boundary detection.

Phase 1, Task 3 per PRD sec 2.2 and sec 4 (weighted scoring).
"""

import json
from dataclasses import dataclass
from typing import Dict, Any

# Initial defaults, tune in Phase 3.
# Why these values:
# - We want a highly confident semantic 'complete' (like rule-based conf=0.9) to cross 
#   the 0.5 threshold on its own (0.9 * 0.60 = 0.54 > 0.5).
# - An LLM 'complete' (conf=0.75) contributes 0.45, requiring either a slight pause 
#   or a speaker change to push it over 0.5, preventing false positives on brief hesitations.
# - A degraded semantic signal gives a neutral base score (0.5), requiring other signals 
#   like speaker_change to cross the threshold.
# - A confident 'incomplete' (conf=0.9) pushes the score so low that even max pause + 
#   speaker change cannot force an 'end of speech' decision.
WEIGHT_SEMANTIC = 0.60
WEIGHT_PAUSE = 0.15
WEIGHT_SPEAKER_CHANGE = 0.25

DECISION_THRESHOLD = 0.5

# Pause mapping constants (PRD sec 1.1)
PAUSE_FLOOR_S = 0.4
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


def fuse(candidate: dict) -> FusionResult:
    """
    Combines signals into a final end-of-speech decision.
    """
    # 1. Semantic Signal
    semantic_label = candidate.get("semantic_label", "incomplete")
    semantic_conf = candidate.get("semantic_confidence", 0.0)
    semantic_source = candidate.get("semantic_source", "unknown")
    
    raw_semantic_val = get_semantic_value(semantic_label, semantic_conf, semantic_source)
    semantic_contrib = raw_semantic_val * WEIGHT_SEMANTIC

    # 2. Pause Signal
    pause_duration = candidate.get("pause_duration_s", 0.0)
    raw_pause_val = get_pause_value(pause_duration)
    pause_contrib = raw_pause_val * WEIGHT_PAUSE

    # 3. Speaker Change Signal
    speaker_changed = candidate.get("speaker_changed", False)
    raw_speaker_val = 1.0 if speaker_changed else 0.0
    speaker_contrib = raw_speaker_val * WEIGHT_SPEAKER_CHANGE

    # Total weighted confidence
    total_confidence = semantic_contrib + pause_contrib + speaker_contrib
    is_eos = total_confidence >= DECISION_THRESHOLD

    signals_breakdown = {
        "semantic": {
            "label": semantic_label,
            "source": semantic_source,
            "raw_confidence": semantic_conf,
            "mapped_value": raw_semantic_val,
            "weighted_contribution": semantic_contrib
        },
        "pause": {
            "duration_s": pause_duration,
            "mapped_value": raw_pause_val,
            "weighted_contribution": pause_contrib
        },
        "speaker_change": {
            "changed": speaker_changed,
            "mapped_value": raw_speaker_val,
            "weighted_contribution": speaker_contrib
        }
    }

    return FusionResult(
        is_end_of_speech=is_eos,
        confidence=total_confidence,
        contributing_signals=signals_breakdown
    )


if __name__ == "__main__":
    input_path = "eval/pause_candidates_result.json"
    output_path = "eval/fusion_result.json"
    
    print(f"Loading candidates from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    
    results = []
    eos_count = 0
    total_conf = 0.0
    
    print(f"{'Pause(s)':<10} | {'Spkr Chg':<10} | {'Semantic (Label/Src)':<25} | {'Conf':<6} | {'Decision'}")
    print("-" * 75)
    
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
        conf_str = f"{fusion_res.confidence:.2f}"
        decision = "EOS" if fusion_res.is_end_of_speech else "Cont"
        
        print(f"{p_dur:<10} | {spkr:<10} | {sem_str:<25} | {conf_str:<6} | {decision}")
        
    print("-" * 75)
    print("SUMMARY STATS:")
    print(f"Total candidates: {len(candidates)}")
    print(f"Resolved to EOS : {eos_count}")
    print(f"Resolved to Cont: {len(candidates) - eos_count}")
    print(f"Avg Confidence  : {total_conf / len(candidates):.3f}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved fused results to {output_path}")
