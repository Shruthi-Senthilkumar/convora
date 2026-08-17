\# Phase 1 Task 4 - EndOfSpeechDetector.process\_file() - Verified



\## First successful full run (2026-08-15)

\- Duration: 1272.6s audio, 260 pause candidates, 59 resolved to EOS

\- Processing time: 589.4s (\~10 min) for a 21-min file - batch mode,

&#x20; no live latency constraint, but this duration increases risk of

&#x20; mid-request connection drops (see below)

\- Semantic source breakdown: rule=92 (35%), llm=143 (55%), degraded=25 (10%)

\- Degraded rate (10%) is higher than earlier standalone pipeline runs

&#x20; (5-7%) - worth investigating before Phase 3, possibly due to

&#x20; cumulative API load/rate-limit pressure across \~150 sequential calls

&#x20; in one long-running process

\- Schema validation: PASSED

\- Manual spot-check of first 5 events: all 5 look correct on inspection



\## Known reliability issue

First run attempt failed with httpx.ReadError (WinError 10054,

connection forcibly closed) during the Deepgram upload/transcription

step. Retry succeeded. Given the long single-request duration (\~10 min

total pipeline, transcription alone likely several minutes for

diarized long-form audio), this class of failure is a real risk.

TODO before Phase 2: consider Deepgram's async/callback pattern for

long files instead of one long-held synchronous request, and/or retry

logic around the transcription call.

