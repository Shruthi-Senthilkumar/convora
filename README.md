# Convora

**Your voice AI doesn't need to guess when you're done talking. It should *understand*.**

Most voice assistants decide you're finished speaking the same way a stopwatch does - count some milliseconds of silence, then jump in. That's why they cut you off mid-thought when you pause to think, and why they leave you hanging after you're clearly done. Convora replaces the stopwatch with judgment: an AI-powered pipeline that listens for *meaning*, not just silence, to know exactly when your turn is over - and reacts instantly if you start talking again.

---

## Why this exists

Every production voice AI system - LiveKit, Pipecat, Deepgram Flux - already solves this problem, and solves it better than a solo weekend project ever will. So why build it again?

Because using someone else's turn-detector teaches you nothing about *how* turn detection actually works. This project exists to open the black box: real ASR integration, real semantic judgment calls, real signal fusion, real interruption handling - engineered, measured, and honestly reported, gaps included.

> **This is a learning project, not a production pitch.** Every architecture decision, every latency number, every dead end is documented in-repo. If Convora is slower than the tools it's compared against, that's expected and stated up front - see [`docs/`](./docs) for the receipts.

---

## What it actually does

```
  Audio in
     |
     v
+------------------+     +-------------------+     +------------------+
|   Deepgram ASR    | --> |  Semantic Judge    | --> |  Signal Fusion   |
|  (transcript +    |     | (rule-gate + LLM   |     | (semantic + pause|
|   diarization)    |     |   completeness)    |     |  + speaker-turn) |
+------------------+     +-------------------+     +------------------+
                                                              |
                                                              v
                                                 [ end-of-speech detected ]
```

Plus a parallel path that listens *while the AI is talking*, so it can stop dead the instant you interrupt - no waiting for a round trip to a server to notice you started speaking.

---

## Status: actively being built, in the open

| Phase | What it covers | Status |
|---|---|---|
| **0 - Setup** | Env, API keys, latency validation, corpus sourcing | Done |
| **1 - Batch Pipeline** | Transcription, semantic judgment, fusion, `EndOfSpeechDetector` | Done |
| **2 - Streaming Mode** | Live WebSocket audio, real-time detection | In progress |
| **3 - Evaluation** | Labeled benchmark vs. silence-threshold & industry baselines | Not started |
| **4-8** | Demo UI, interruption handling, final packaging | Not started |

This isn't a polished 1.0 pretending otherwise - it's a working system being built one verified step at a time, and the repo history shows every real bug, wrong turn, and fix along the way.

---

## The stack

| Layer | Tool |
|---|---|
| ASR + diarization | Deepgram Nova-3 |
| Semantic judgment | Rule-based gate (spaCy) -> hosted LLM (Groq) fallback |
| Fusion / decision logic | Custom weighted scoring, fully inspectable |
| Streaming | FastAPI + WebSockets |
| Interruption handling | WebRTC AEC + Silero VAD (client-side) |
| Voice output | ElevenLabs (streaming TTS) |
| Evaluation | pandas, matplotlib, a labeled conversational dataset |

---

## Getting started

```bash
git clone https://github.com/Shruthi-Senthilkumar/convora.git
cd convora
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and add your own keys:
```
DEEPGRAM_API_KEY=
GROQ_API_KEY=
ELEVENLABS_API_KEY=
```

Run the batch pipeline against a sample file:
```bash
python -m detection.end_of_speech_detector
```

Or spin up the live streaming server:
```bash
uvicorn backend.main:app --reload
```

Test with file streaming:
```bash
python backend/test_client.py
```

Test with your own microphone:
```bash
# List available audio input devices
python backend/live_mic_test.py --list-devices

# Stream live mic input to the server
python backend/live_mic_test.py
```

---

## Dig deeper

- [`PRD_Convora_v1_5.docx`](./PRD_Convora_v1_5.docx) - the full product spec, success criteria, and phase-by-phase plan
- [`docs/`](./docs) - latency investigations, model comparisons, and every decision's paper trail, including the ones that didn't pan out
- [`eval/`](./eval) - real test results against real conversational audio

---

*Built solo, end to end - architecture, backend, AI integration, and evaluation - as a portfolio project in public.*
