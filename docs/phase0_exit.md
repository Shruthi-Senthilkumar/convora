# Phase 0 Exit Checklist

[x] Repo, venv, deps, GitHub setup (Step 1)
[x] Corpus decision: self-recorded + AMI, Switchboard/Fisher excluded on
    cost grounds ($3000 non-member fee confirmed) (Step 2)
[x] API keys: Deepgram, Groq, ElevenLabs acquired, scoped, verified (Step 3)
[x] Latency spike: extensively tested - 4 hosted vendors (Groq x3 models,
    Cerebras, Gemini), all exceeded 250ms p95 budget. Local proxy testing
    (Colab T4) showed rule-gate + Qwen3-1.7B near budget (~290ms p95).
    Decision: proceed with hosted Groq + 56%-coverage rule gate, latency
    budget accepted as "decent" rather than strictly met. Documented as
    a conscious, non-silent tradeoff (Step 4)
[x] SemanticJudge implemented and smoke-tested (detection/semantic_judge.py)
[~] Test audio: AMI verified working, self-recorded deferred to Phase 3
    (Step 5)
[x] Parselmouth: confirmed working on real AMI audio - 1272.64s duration,
    127261 pitch frames extracted successfully (Step 6)
[ ] Smart Turn v3 reference-ceiling setup: not run - deferred, belongs
    with Phase 3 benchmarking per PRD sec 5, not blocking Phase 1 start

## Phase 0 status
COMPLETE for the purposes of proceeding to Phase 1. Deferred items
(self-recorded audio, Smart Turn v3 benchmark) are explicitly tracked,
not silently dropped, and scheduled for their correct phases per the
PRD timeline.


