\# Phase 0 - Step 4: Latency Spike Findings (in progress)



\## Status: INCOMPLETE - decision gate not yet resolved. Resume here.



\## Runs completed (2026-08-06)



\### Qwen3.6-27B, reasoning\_effort="none", 2.2s pacing

\- p50: 171ms, p95: 1142ms, max: 2843ms

\- 9/50 (18%) over 400ms deadline



\### Qwen3.6-27B, reasoning\_effort="none", 3.5s pacing (re-run to rule out

&#x20; rate-limit-window interaction)

\- p50: 227ms, p95: 7406ms, max: 44940ms

\- 14/50 (28%) over 400ms deadline

\- Wider pacing made the tail WORSE, not better - rules out rate-limit

&#x20; interaction as the cause. Tail latency appears to be genuine, unpredictable

&#x20; instability on the preview endpoint itself.



\### gpt-oss-20b, reasoning\_effort="low", 3.5s pacing

\- p50: 404ms, p95: 1910ms, max: 3917ms

\- 26/50 (52%) over 400ms deadline

\- WORSE than Qwen on both p50 and miss rate - contradicts PRD's assumption

&#x20; that gpt-oss-20b is the safe low-latency fallback.

\- BUG: all responses came back empty. Suspect max\_tokens=5 is being consumed

&#x20; by internal reasoning tokens even at "low" effort, leaving nothing for the

&#x20; visible answer. Need to re-test with higher max\_tokens (50-100) before

&#x20; gpt-oss-20b can be fairly judged - current numbers may not reflect a

&#x20; working configuration.



\## Both models currently FAIL the 250ms p95 budget in PRD sec 2.2.



\## Next steps (resume here)

1\. Fix gpt-oss-20b max\_tokens and re-run for a fair comparison (empty

&#x20;  responses currently invalidate its numbers)

2\. Re-run Qwen once more at a different time of day to check if tail

&#x20;  latency is time-dependent congestion or a stable characteristic

3\. Based on results, decide between:

&#x20;  a. Accept the degraded-fallback architecture (PRD sec 2.2) as-is and

&#x20;     report the real degraded-rate as a headline Phase 3 metric

&#x20;  b. Split usage by phase - Qwen for Phase 1 batch (no latency

&#x20;     constraint), faster option for Phase 2/3 streaming

&#x20;  c. Investigate a local small model for the semantic step (PRD sec 2.2

&#x20;     option 3), avoiding hosted-API tail latency entirely

4\. Once resolved, this becomes the final "Latency Spike / decision gate"

&#x20;  entry for Phase 0 exit (see docs/phase0\_exit.md)

## Network latency isolation test (2026-08-11)



Raw ICMP ping RTT (no model inference involved):

\- api.groq.com: 21-69ms (avg \~36ms)

\- generativelanguage.googleapis.com: 26-33ms (avg \~29ms)



FINDING: Network distance hypothesis is FALSIFIED. Raw RTT to both

providers is fast and unremarkable (\~20-70ms) - nowhere near enough to

explain the 400-1500ms+ latency measured across all model tests.



REVISED CONCLUSION: The consistent latency floor across 4 vendors/3

architectures is most likely explained by INFERENCE-SIDE factors -

actual compute time plus free-tier request queuing/priority on shared

multi-tenant infrastructure - not network transport.



This strengthens rather than weakens the case for local hosting: local

inference on dedicated hardware (RTX 5070) has zero queuing contention

and no shared-tenant priority deprioritization, directly addressing the

now-identified bottleneck rather than a discarded one.

## SemanticJudge implementation note (2026-08-11)



Smoke test confirms rule gate + Groq LLM path both work correctly:

\- Rule-resolved cases: 0-9ms, correct labels

\- LLM-escalated case: 827ms, correct label ("complete")



KNOWN GAP: client-level `timeout=0.4` on the Groq SDK does not appear to

strictly enforce a 400ms hard cutoff (observed 827ms completion instead

of abort+fallback). Not blocking for Phase 0 - deferred to Phase 2 when

proper streaming timeout enforcement becomes load-bearing rather than

a nice-to-have.

