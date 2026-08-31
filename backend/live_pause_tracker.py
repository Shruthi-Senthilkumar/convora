"""
backend/live_pause_tracker.py

Phase 2, Tasks 3-4: Live pause detection, rolling buffer, semantic judgment, and fusion.
PRD - 5, Phase 2, Tasks 3-4.

This module provides the LivePauseTracker class, which tracks word-level transcription
results, identifies candidate pause points incrementally, and performs semantic and
fusion evaluation.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from detection.semantic_judge import SemanticJudge, JudgeResult
from detection.fusion import fuse, FusionResult

logger = logging.getLogger(__name__)


class LivePauseTracker:
    """
    Tracks streaming transcription words and evaluates turn boundaries
    at candidate pause points.
    """

    def __init__(self, semantic_judge: Optional[SemanticJudge] = None, window_size: int = 12):
        self.semantic_judge = semantic_judge or SemanticJudge()
        self.window_size = window_size
        
        # rolling buffer of finalized words
        # each word: {"word": str, "start": float, "end": float, "speaker": int | None}
        self.words: List[Dict[str, Any]] = []
        
        # Simple per-session cache: stores the last fragment judged and its result.
        # This optimizes latency by preventing multiple LLM calls for the same utterance
        # during consecutive triggers (e.g. speech_final -> utterance_end).
        self.last_fragment_text: Optional[str] = None
        self.last_judge_result: Optional[JudgeResult] = None

        # Tracks evaluated boundaries to prevent redundant evaluations.
        # Key: boundary timestamp_s (float), Value: {"resolved_eos": bool, "max_pause_duration": float}
        self.boundary_states: Dict[float, Dict[str, Any]] = {}

    def _should_evaluate_candidate(self, timestamp_s: float, pause_duration_s: float, source: str) -> bool:
        """
        Deduplicates candidate checks. Returns True if:
          1. This boundary hasn't been evaluated yet.
          2. The boundary hasn't resolved to EOS yet.
          3. The pause duration hasn't already reached the 3.0s ceiling.
          4. It is not a redundant speech_final trigger during ongoing silence.
          5. The pause duration has grown by at least 0.2s since the last check.
        """
        state = self.boundary_states.get(timestamp_s)
        if state is None:
            # First time seeing this boundary
            self.boundary_states[timestamp_s] = {
                "resolved_eos": False,
                "max_pause_duration": pause_duration_s
            }
            logger.debug("Boundary at %.2fs registered via %s (pause: %.2fs)", timestamp_s, source, pause_duration_s)
            return True

        if state["resolved_eos"]:
            return False

        # If the pause has already reached the ceiling (3.0s), further evaluations are redundant
        if state["max_pause_duration"] >= 3.0:
            return False

        # Skip subsequent speech_final triggers on the same boundary
        # (Deepgram sends empty final segments with speech_final during silence)
        if source == "speech_final":
            return False

        # Only check again if the pause duration has increased significantly (>= 200ms)
        if pause_duration_s >= state["max_pause_duration"] + 0.2:
            logger.debug(
                "Boundary at %.2fs re-evaluating via %s: pause grew from %.2fs to %.2fs",
                timestamp_s, source, state["max_pause_duration"], pause_duration_s
            )
            state["max_pause_duration"] = pause_duration_s
            return True

        return False

    def mark_boundary_resolved(self, timestamp_s: float) -> None:
        """
        Marks a specific boundary timestamp as resolved to EOS, preventing any further checks.
        """
        if timestamp_s in self.boundary_states:
            self.boundary_states[timestamp_s]["resolved_eos"] = True
            logger.debug("Boundary at %.2fs marked as RESOLVED (EOS)", timestamp_s)

    def build_fragment(self, last_word_index: int) -> Tuple[str, Optional[int]]:
        """
        Builds a trailing text fragment ending at last_word_index, up to window_size words.
        Does not cross speaker boundaries, matching Phase 1 batch mode.
        """
        if last_word_index < 0 or last_word_index >= len(self.words):
            return "", None
            
        last_word = self.words[last_word_index]
        current_speaker = last_word.get("speaker")
        
        fragment_words = []
        for j in range(last_word_index, -1, -1):
            if len(fragment_words) >= self.window_size:
                break
            w = self.words[j]
            if w.get("speaker") != current_speaker:
                break
            fragment_words.insert(0, w)
            
        fragment = " ".join(w["word"] for w in fragment_words)
        return fragment, current_speaker

    def register_words_and_detect_candidates(self, message: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Processes an incoming ListenV1Results final transcript chunk.
        Appends words to the rolling buffer and detects candidate pause points.

        A candidate pause point is triggered by:
          1. A gap >= 0.4s between the last word of the previous chunk and the first word of this chunk.
          2. A speaker change between the previous chunk and this chunk.
          3. Deepgram's speech_final flag indicating built-in endpointing has occurred.

        Returns:
          Tuple of (new_words_added: list, candidates: list)
        """
        alt = message.channel.alternatives[0]
        new_words = []
        if alt.words:
            for w in alt.words:
                new_words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "speaker": getattr(w, "speaker", None)
                })

        candidates = []
        
        # 1 & 2: Check for a gap or speaker change at the boundary between current buffer and new words.
        if self.words and new_words:
            last_word = self.words[-1]
            first_new_word = new_words[0]
            gap = first_new_word["start"] - last_word["end"]
            
            has_speaker = (last_word.get("speaker") is not None and first_new_word.get("speaker") is not None)
            speaker_changed = has_speaker and (last_word["speaker"] != first_new_word["speaker"])
            
            if gap >= 0.3 or speaker_changed:
                ts = last_word["end"]
                pause_dur = max(0.0, round(gap, 3))
                if self._should_evaluate_candidate(ts, pause_dur, "gap_or_speaker_change"):
                    candidates.append({
                        "last_word_index": len(self.words) - 1,
                        "pause_duration_s": pause_dur,
                        "speaker_changed": speaker_changed,
                        "timestamp_s": ts,
                        "source": "gap_or_speaker_change"
                    })

        # Append the new words to the session buffer
        self.words.extend(new_words)

        # 3: Check Deepgram's own endpointing signal (speech_final)
        if message.speech_final and self.words:
            last_word = self.words[-1]
            segment_end = message.start + message.duration
            pause_duration = segment_end - last_word["end"]
            if pause_duration < 0.0:
                pause_duration = 0.3  # Fallback to the minimum threshold
                
            ts = last_word["end"]
            pause_dur = max(0.0, round(pause_duration, 3))
            if self._should_evaluate_candidate(ts, pause_dur, "speech_final"):
                candidates.append({
                    "last_word_index": len(self.words) - 1,
                    "pause_duration_s": pause_dur,
                    "speaker_changed": False,
                    "timestamp_s": ts,
                    "source": "speech_final"
                })

        return new_words, candidates

    def detect_utterance_end_candidate(self, last_word_end: float) -> List[Dict[str, Any]]:
        """
        Detects a candidate pause point triggered by a Deepgram UtteranceEnd message.
        """
        if not self.words:
            return []
            
        last_word = self.words[-1]
        pause_duration = last_word_end - last_word["end"]
        if pause_duration < 0.0:
            pause_duration = 1.0  # default silence duration for UtteranceEnd
            
        ts = last_word["end"]
        pause_dur = max(0.0, round(pause_duration, 3))
        if self._should_evaluate_candidate(ts, pause_dur, "utterance_end"):
            return [{
                "last_word_index": len(self.words) - 1,
                "pause_duration_s": pause_dur,
                "speaker_changed": False,
                "timestamp_s": ts,
                "source": "utterance_end"
            }]
        return []

    def evaluate_candidate_sync(self, fragment: str, pause_duration_s: float, speaker_changed: bool) -> Tuple[FusionResult, float]:
        """
        Runs SemanticJudge and Fusion synchronously.
        Measures the CPU/network latency of the judge + fuse execution itself using time.perf_counter().
        Uses simple per-session caching to avoid redundant LLM calls if the fragment is identical.
        """
        start_perf = time.perf_counter()
        
        # 1. Semantic Judge
        if fragment == self.last_fragment_text and self.last_judge_result is not None:
            judge_res = self.last_judge_result
            logger.debug("Reusing cached semantic judgment for: %r", fragment)
        else:
            try:
                judge_res = self.semantic_judge.judge(fragment)
            except Exception as e:
                logger.error("Error running SemanticJudge: %s", e, exc_info=True)
                judge_res = JudgeResult(label="incomplete", confidence=0.3, source="degraded", latency_ms=0.0)
            self.last_fragment_text = fragment
            self.last_judge_result = judge_res

        # 2. Fusion
        fuse_payload = {
            "pause_duration_s": pause_duration_s,
            "speaker_changed": speaker_changed,
            "fragment": fragment,
            "semantic_label": judge_res.label,
            "semantic_source": judge_res.source,
            "semantic_confidence": judge_res.confidence
        }
        
        try:
            fusion_res = fuse(fuse_payload)
        except Exception as e:
            logger.error("Error in fusion layer: %s", e, exc_info=True)
            fusion_res = FusionResult(
                is_end_of_speech=False,
                confidence=0.0,
                contributing_signals={}
            )

        elapsed_ms = (time.perf_counter() - start_perf) * 1000
        return fusion_res, elapsed_ms
