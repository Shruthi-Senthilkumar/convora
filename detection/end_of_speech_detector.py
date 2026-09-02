"""
detection/end_of_speech_detector.py

Orchestrates the entire end-of-speech detection pipeline.
Phase 1, Task 4 (per PRD sec 3, 5, and 9).

Pipeline:
1. Transcribe audio with deepgram batch (word timestamps + diarization)
2. Find candidate pause points (>= 400ms or speaker change)
3. Judge semantic completeness of trailing fragments
4. Fuse semantic, pause duration, and speaker change signals into a confidence score
"""

import time
import json
import sys
import os

from detection.transcribe_batch import transcribe_file
from detection.find_pause_candidates import find_pause_candidates, judge_candidates
from detection.semantic_judge import SemanticJudge
from detection.fusion import fuse


class EndOfSpeechDetector:
    def __init__(self, semantic_judge: SemanticJudge = None):
        """
        Initializes the detector. Allows injecting a custom SemanticJudge,
        defaults to the real SemanticJudge if none provided.
        """
        self.semantic_judge = semantic_judge or SemanticJudge()

    def process_file(self, audio_path: str) -> dict:
        """
        Runs the full detection pipeline on a given audio file.
        Returns a spec-compliant JSON dictionary.
        """
        start_time = time.perf_counter()

        # 1. Transcribe
        transcription_result = transcribe_file(audio_path)
        words = transcription_result.get("words", [])
        duration = transcription_result.get("duration", 0.0)

        # 2. Find Candidates
        candidates = find_pause_candidates(words)
        
        # 3. Judge Candidates
        if candidates:
            candidates = judge_candidates(candidates, self.semantic_judge)

        # 4. Fuse and Filter
        events = []
        all_candidates = []
        rule_count = 0
        llm_count = 0
        degraded_count = 0

        for candidate in candidates:
            # Tally semantic sources for metadata
            source = candidate.get("semantic_source")
            if source == "rule":
                rule_count += 1
            elif source == "llm":
                llm_count += 1
            elif source == "degraded":
                degraded_count += 1

            fusion_res = fuse(candidate)

            candidate_entry = {
                "timestamp_s": candidate.get("pause_start"),
                "is_end_of_speech": fusion_res.is_end_of_speech,
                "confidence": fusion_res.confidence,
                "speaker": candidate.get("speaker"),
                "speaker_changed": candidate.get("speaker_changed", False),
                "fragment": candidate.get("fragment"),
                "contributing_signals": fusion_res.contributing_signals
            }
            all_candidates.append(candidate_entry)

            if fusion_res.is_end_of_speech:
                events.append({
                    "event_type": "end_of_speech",
                    "timestamp_s": candidate.get("pause_start"),
                    "confidence": fusion_res.confidence,
                    "speaker": candidate.get("speaker"),
                    "fragment": candidate.get("fragment"),
                    "contributing_signals": fusion_res.contributing_signals
                })

        processing_time_s = time.perf_counter() - start_time

        output = {
            "audio_file": audio_path,
            "duration_s": duration,
            "events": events,
            "all_candidates": all_candidates,
            "metadata": {
                "total_pause_candidates": len(candidates),
                "resolved_to_end_of_speech": len(events),
                "processing_time_s": processing_time_s,
                "semantic_source_breakdown": {
                    "rule": rule_count,
                    "llm": llm_count,
                    "degraded": degraded_count
                }
            }
        }
        return output


if __name__ == "__main__":
    audio_file = r"C:\Users\shrut\ami-corpus-data\amicorpus\ES2002a\audio\ES2002a.Mix-Headset.wav"
    output_path = "eval/end_of_speech_detector_result.json"

    if not os.path.exists(audio_file):
        print(f"Error: audio file not found at {audio_file}")
        sys.exit(1)

    print(f"Running pipeline on {audio_file}...")
    detector = EndOfSpeechDetector()
    result = detector.process_file(audio_file)

    meta = result["metadata"]
    print(f"\n--- Pipeline Summary ---")
    print(f"Total Audio Duration : {result['duration_s']:.1f}s")
    print(f"Total Candidates     : {meta['total_pause_candidates']}")
    print(f"Resolved to EOS      : {meta['resolved_to_end_of_speech']}")
    print(f"Processing Time      : {meta['processing_time_s']:.1f}s")
    print(f"Source Breakdown     : {meta['semantic_source_breakdown']}")

    print(f"\n--- First 5 Events ---")
    for i, event in enumerate(result["events"][:5], 1):
        print(f"Event {i}:")
        print(f"  Timestamp : {event['timestamp_s']}s")
        print(f"  Confidence: {event['confidence']:.2f}")
        print(f"  Speaker   : {event['speaker']}")
        print(f"  Fragment  : {event['fragment']}")
        print(f"  Signals   : {event['contributing_signals']}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full JSON output to {output_path}")

    # Basic Schema Validation
    print("\n--- Validating Schema ---")
    is_valid = True
    errors = []

    if "audio_file" not in result: errors.append("Missing 'audio_file'")
    if "duration_s" not in result: errors.append("Missing 'duration_s'")
    if "events" not in result or not isinstance(result["events"], list):
        errors.append("Missing or invalid 'events' list")
    else:
        for i, ev in enumerate(result["events"]):
            if "event_type" not in ev or ev["event_type"] != "end_of_speech":
                errors.append(f"Event {i} missing/wrong 'event_type'")
            if "timestamp_s" not in ev: errors.append(f"Event {i} missing 'timestamp_s'")
            if "confidence" not in ev: errors.append(f"Event {i} missing 'confidence'")
            if "speaker" not in ev: errors.append(f"Event {i} missing 'speaker'")
            if "fragment" not in ev: errors.append(f"Event {i} missing 'fragment'")
            if "contributing_signals" not in ev or not isinstance(ev["contributing_signals"], dict):
                errors.append(f"Event {i} missing/invalid 'contributing_signals'")

    if "all_candidates" not in result or not isinstance(result["all_candidates"], list):
        errors.append("Missing or invalid 'all_candidates' list")
    else:
        for i, cand in enumerate(result["all_candidates"]):
            if "timestamp_s" not in cand: errors.append(f"Candidate {i} missing 'timestamp_s'")
            if "is_end_of_speech" not in cand: errors.append(f"Candidate {i} missing 'is_end_of_speech'")
            if "confidence" not in cand: errors.append(f"Candidate {i} missing 'confidence'")
            if "speaker" not in cand: errors.append(f"Candidate {i} missing 'speaker'")
            if "speaker_changed" not in cand: errors.append(f"Candidate {i} missing 'speaker_changed'")
            if "fragment" not in cand: errors.append(f"Candidate {i} missing 'fragment'")
            if "contributing_signals" not in cand or not isinstance(cand["contributing_signals"], dict):
                errors.append(f"Candidate {i} missing/invalid 'contributing_signals'")

    if "metadata" not in result or not isinstance(result["metadata"], dict):
        errors.append("Missing or invalid 'metadata' dict")
    else:
        m = result["metadata"]
        if "total_pause_candidates" not in m: errors.append("Metadata missing 'total_pause_candidates'")
        if "resolved_to_end_of_speech" not in m: errors.append("Metadata missing 'resolved_to_end_of_speech'")
        if "processing_time_s" not in m: errors.append("Metadata missing 'processing_time_s'")
        if "semantic_source_breakdown" not in m or not isinstance(m["semantic_source_breakdown"], dict):
            errors.append("Metadata missing/invalid 'semantic_source_breakdown'")
        else:
            sb = m["semantic_source_breakdown"]
            if "rule" not in sb: errors.append("Source breakdown missing 'rule'")
            if "llm" not in sb: errors.append("Source breakdown missing 'llm'")
            if "degraded" not in sb: errors.append("Source breakdown missing 'degraded'")

    if errors:
        print("SCHEMA INVALID:")
        for err in errors:
            print(f" - {err}")
    else:
        print("SCHEMA VALID")
