# Phase 2 - Live Microphone Test - Verified (2026-08-18)

## Summary

The live microphone path (backend/live_mic_test.py -> /ws/transcribe
-> Deepgram streaming -> LivePauseTracker -> SemanticJudge -> fusion)
has been tested across four real runs, progressing from basic
plumbing checks to a full real two-person conversation. This is the
most complete end-to-end verification this project has produced.

---

## Run 1 - single speaker, casual/fragmented speech (20s)

All candidates resolved to CONT (confidence 0.15-0.32). No genuine
complete sentences were spoken - correctly reflects that rapid
informal back-and-forth ("Hello? Hello?", "It's okay, it's okay")
doesn't contain clean turn-boundaries. Not a bug - correct
conservative behavior under ambiguous input.

## Run 2 - single speaker, one deliberate complete sentence (20s)

"a project my friends and also we are going to a trip" -> EOS,
confidence 0.55, latency 1.0ms (rule-gate, cached).
"and" (trailing conjunction alone) -> CONT, confidence 0.07 -
correctly caught by the rule gate as strongly incomplete.

Confirmed the pipeline correctly distinguishes genuine sentence
completion from casual fragmented talk, on live, unscripted speech.

## Run 3 - speaker labeling added, ambient silence check (15s)

Added speaker/speaker_changed fields to the end_of_speech_candidate
WebSocket payload (previously computed internally but not sent to
the client). Updated live_mic_test.py and test_client.py to print
spk<id> tags, with a `*` marker when speaker_changed is True.
Verified against silence: 0 transcripts, 0 candidates, as expected -
confirms the updated code doesn't crash and behaves correctly with
no input.

## Run 4 - REAL TWO-PERSON CONVERSATION over a single microphone (45s)

Two real people (self + a friend) had a genuine unscripted
conversation into the same physical microphone.

### Confirmed working:
- **Live diarization correctly distinguished two different human
  voices on a single mic in real time.** Speaker tracked as
  spk0 -> spk1 -> spk0 through the conversation, matching the actual
  turn-taking between the two speakers.
- **speaker_changed correctly fed into fusion scoring** - candidates
  at genuine speaker-change boundaries were appropriately weighted.
- **Multiple correct EOS detections** across both speakers on
  genuinely complete sentences, e.g.:
  - "hi how are you what is your name" -> EOS (conf 0.75)
  - "...i'm going to my home this weekend" -> EOS (conf 0.85, highest
    of the run - appropriately so, as this was the most clearly
    complete, standalone statement spoken)
  - "how is the project is going what is the project condition now"
    -> EOS (conf 0.60)
- **CONT correctly maintained** through incomplete fragments,
  mid-sentence partials, and trailing words like "and" across both
  speakers.
- Latency remained strong throughout: rule-gate/cached hits at
  0.7-17ms, LLM escalations at 130-325ms - consistent with every
  prior latency measurement in this project.

Full raw output saved to eval/live_mic_test_run.txt.

---

## Overall conclusion

The live pipeline is confirmed working end-to-end on real,
unscripted, MULTI-SPEAKER conversation over a single microphone:
transcription, diarization, pause detection, semantic judgment, and
signal fusion all verified working together correctly on genuine
two-person speech, not just pre-processed batch files or synthetic
test data.

This closes Phase 2's final task (live mic test, script-based,
pre-UI) per the PRD's Phase 2 task list.

## Known limitations / honest gaps

- Real accuracy validation (PRD sec 2.1 FP/FN targets) still requires
  the formal Phase 3 labeled-evaluation protocol (PRD sec 2.4) -
  everything verified so far is spot-checking and plausibility review,
  not statistically rigorous measurement.
- The 250ms p95 semantic-judgment latency budget (PRD sec 2.2) is
  still not strictly met on the hosted Groq path in worst-case
  scenarios (see docs/latency_spike_findings.md) - accepted as a
  known, documented tradeoff per project decision, mitigated by the
  rule-gate reducing how often the LLM path is hit at all.
- Diarization has only been tested with two speakers so far; behavior
  with 3+ simultaneous/overlapping speakers is untested.
