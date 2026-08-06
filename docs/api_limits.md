# API Free-Tier Limits — Phase 0

## Groq (Qwen3.6-27B, Preview)
- Requests: 30/minute, 1,000/day
- Tokens: 8,000/minute, 200,000/day
- Release stage: Preview, released 2026-05-09
- gpt-oss-20b also confirmed available on account as fallback model
- Confirmed via console.groq.com/playground, 2026-08-06

## Implication for latency spike / rate limits (see PRD sec 2.2, sec 9)
30 req/min = roughly 1 semantic call every 2 seconds sustained max. This is
a hard ceiling on Phase 0's 50-call latency spike (spike must be paced, not
fired in a tight burst) and on later live-testing volume. Reinforces the
sec 9 open question on whether every pause needs a semantic call or whether
a cheaper local pre-filter gate is required to stay inside this limit.

## Deepgram
- Plan: Pay As You Go
- Starting credit: $200.00
- Auto-reload: OFF (intentional - avoids unexpected charges during dev/test)
- Confirmed via console.deepgram.com billing overview, 2026-08-06

## ElevenLabs
- API key scoped to Text-to-Speech only (least-privilege; no User, Voices,
  or account-management permissions granted)
- Auto-disable if leaked: ON
- Confirmed working via direct TTS call (eleven_turbo_v2_5, Rachel voice),
  2026-08-06

## Key acquisition status
All three keys acquired, scoped appropriately, and verified working via
eval/smoke_test_keys.py as of 2026-08-06.
