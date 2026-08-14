"""
detection/find_pause_candidates.py

Phase 1, Task 2: Semantic completeness check on transcript.

Scans a Deepgram word-timestamp+diarization transcript for candidate
pause points - gaps between consecutive words >= 400ms, per PRD sec 2.4.
For each candidate, runs SemanticJudge against the trailing fragment.

SPEAKER-AWARE (fix applied after Phase 1 manual spot-check revealed
cross-speaker fragments were largely unjudgeable crosstalk on AMI's
multi-party audio):
  - Fragments never span a speaker change - only words from the SAME
    speaker as the word immediately before the pause are included.
  - A speaker change is itself always treated as a pause candidate
    (someone else starting to talk is a strong turn-boundary signal,
    independent of silence duration).

Run standalone against a saved transcription result:
    python detection/find_pause_candidates.py eval/transcribe_batch_test_result.json
"""

import sys
import json
from detection.semantic_judge import SemanticJudge

PAUSE_THRESHOLD_S = 0.4  # PRD sec 2.4: pauses >= 400ms are scoreable
TRAILING_WORD_WINDOW = 12  # how many words of context to feed the judge


def find_pause_candidates(words: list) -> list:
    """
    Returns a list of dicts, one per candidate pause point. A candidate
    is triggered by EITHER a gap >= PAUSE_THRESHOLD_S, OR a speaker
    change (even with zero/tiny gap) - both are real turn-boundary
    signals. Fragments are built ONLY from words matching the speaker
    of the word immediately before the pause - never crossing speakers.
    """
    candidates = []
    diarization_available = any(w.get("speaker") is not None for w in words)

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        gap = next_word["start"] - current_word["end"]

        speaker_changed = (
            diarization_available
            and current_word.get("speaker") is not None
            and next_word.get("speaker") is not None
            and current_word["speaker"] != next_word["speaker"]
        )

        if gap >= PAUSE_THRESHOLD_S or speaker_changed:
            current_speaker = current_word.get("speaker")

            # Walk backward from i, only including words from the SAME
            # speaker as current_word - stop at a speaker change even
            # if we haven't hit TRAILING_WORD_WINDOW yet.
            fragment_words = []
            for j in range(i, -1, -1):
                if len(fragment_words) >= TRAILING_WORD_WINDOW:
                    break
                w = words[j]
                if diarization_available and w.get("speaker") != current_speaker:
                    break
                fragment_words.insert(0, w)

            if not fragment_words:
                continue  # nothing usable (e.g. very first word already a boundary)

            fragment = " ".join(w["word"] for w in fragment_words)

            candidates.append({
                "pause_start": current_word["end"],
                "pause_end": next_word["start"],
                "pause_duration_s": round(gap, 3),
                "speaker_changed": speaker_changed,
                "speaker": current_speaker,
                "fragment": fragment,
                "word_index": i,
            })

    return candidates


def judge_candidates(candidates: list, judge: SemanticJudge) -> list:
    for c in candidates:
        result = judge.judge(c["fragment"])
        c["semantic_label"] = result.label
        c["semantic_source"] = result.source
        c["semantic_confidence"] = result.confidence
        c["semantic_latency_ms"] = result.latency_ms
    return candidates


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detection/find_pause_candidates.py <transcription_result.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data["words"]
    print(f"Loaded transcript: {len(words)} words, {data['duration']:.1f}s duration")

    diarization_available = any(w.get("speaker") is not None for w in words)
    print(f"Diarization available: {diarization_available}")
    if not diarization_available:
        print("WARNING: no speaker labels found - re-run transcribe_batch.py "
              "with the updated diarize=True version first.")

    candidates = find_pause_candidates(words)
    speaker_change_count = sum(1 for c in candidates if c["speaker_changed"])
    print(f"Found {len(candidates)} candidate pause points "
          f"({speaker_change_count} are speaker changes)")

    if not candidates:
        print("No candidates found - nothing to judge.")
        sys.exit(0)

    print(f"\nRunning SemanticJudge on all {len(candidates)} candidates...\n")

    judge = SemanticJudge()
    candidates = judge_candidates(candidates, judge)

    rule_count = sum(1 for c in candidates if c["semantic_source"] == "rule")
    llm_count = sum(1 for c in candidates if c["semantic_source"] == "llm")
    degraded_count = sum(1 for c in candidates if c["semantic_source"] == "degraded")

    print(f"{'#':<4}{'Pause(s)':<10}{'Spk':<8}{'Label':<12}{'Source':<10}{'Fragment (trailing)'}")
    print("-" * 110)
    for i, c in enumerate(candidates[:30], 1):
        frag_display = c["fragment"][-55:] if len(c["fragment"]) > 55 else c["fragment"]
        spk_flag = f"{c['speaker']}*" if c["speaker_changed"] else str(c["speaker"])
        print(f"{i:<4}{c['pause_duration_s']:<10}{spk_flag:<8}{c['semantic_label']:<12}{c['semantic_source']:<10}...{frag_display}")

    if len(candidates) > 30:
        print(f"... ({len(candidates) - 30} more not shown)")

    print(f"\n=== Summary ===")
    print(f"Total candidates: {len(candidates)}")
    print(f"Speaker-change triggered: {speaker_change_count} ({100*speaker_change_count/len(candidates):.0f}%)")
    print(f"Resolved by rule: {rule_count} ({100*rule_count/len(candidates):.0f}%)")
    print(f"Escalated to LLM: {llm_count} ({100*llm_count/len(candidates):.0f}%)")
    print(f"Degraded fallback: {degraded_count} ({100*degraded_count/len(candidates):.0f}%)")

    out_path = "eval/pause_candidates_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    print(f"\nFull results saved to {out_path}")
