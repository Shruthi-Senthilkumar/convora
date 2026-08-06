"""
Phase 0 - Step 4: Latency spike.

Validates the 250ms p95 budget allocated to the semantic judgment call in
PRD sec 2.2, BEFORE any pipeline code is written. Fires 50 representative
end-of-speech judgment prompts against Groq (Qwen3.6-27B, reasoning
disabled), paced to respect the free-tier limit of 30 requests/minute
confirmed in docs/api_limits.md.

Decision gate (per PRD sec 2.2):
  - If p95 <= 250ms: budget confirmed, proceed to Phase 1 as planned.
  - If p95 > 250ms: budget must be rebalanced, OR the semantic step moves
    to gpt-oss-20b (already confirmed available) or a local model, BEFORE
    Phase 1 begins.

Run from the convora project root with the venv activated:
    python eval/latency_spike.py

Outputs:
    - Console summary (p50, p95, min, max, mean)
    - eval/latency_spike_results.csv (raw per-call timing + response)
"""

import os
import time
import csv
import statistics
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.6-27b"
SECONDS_BETWEEN_CALLS = 3.5  # widened from 2.2s to rule out rate-limit-window
                              # interaction as the cause of tail latency
DEADLINE_MS = 400  # matches the hard pipeline deadline in PRD sec 2.2
# reasoning_effort accepted values differ by model family:
#   - Qwen models on Groq accept "none" (fully disables reasoning)
#   - gpt-oss models only accept "low" / "medium" / "high" - "low" is the
#     closest equivalent to disabled reasoning for this model family
REASONING_EFFORT = "none" if "qwen" in MODEL.lower() else "low"
WARMUP_CALLS = 1  # first call often includes connection cold-start; run
                   # and discard this many calls before the timed sample

SYSTEM_PROMPT = (
    "You are judging whether a speaker has finished their conversational "
    "turn based on a transcript fragment. Respond with ONLY one word: "
    "'complete' if the turn sounds finished, or 'incomplete' if the "
    "speaker seems likely to continue. No explanation."
)

# 50 representative fragments spanning the edge cases called out in
# PRD sec 2.4: clear completions, mid-thought pauses, backchannels,
# trailing questions, list enumeration, self-corrections, trailing filler.
TEST_PROMPTS = [
    "I think we should go with option B",
    "So I think... we should go with",
    "Can you send me the report by Friday",
    "mm-hmm",
    "right",
    "yeah",
    "I went to the store and bought milk, eggs, and",
    "The meeting is scheduled for next Tuesday at",
    "I guess we could try that approach, so, yeah",
    "Is that the right way to do it, or",
    "We need to finish the first phase before we",
    "Actually, I meant to say the second option, not the",
    "Let's go over the budget numbers for this quarter",
    "First we need to check the server, then the database, and finally",
    "I was thinking maybe we should",
    "That sounds like a good plan to me",
    "Wait, I don't think that's",
    "The results came back positive for all three",
    "Honestly I'm not sure what to",
    "Let me pull up the document real quick",
    "So basically what happened was",
    "Thanks for your help today",
    "Can we reschedule the call to",
    "I completed the task yesterday afternoon",
    "There were a few issues with the deployment, namely",
    "So, yeah",
    "I guess",
    "It's due next week I believe",
    "We should probably talk to the client about",
    "The presentation went well overall",
    "Do you have any questions about the",
    "I'll follow up with you tomorrow morning",
    "Considering the budget constraints we discussed earlier",
    "That's exactly what I was thinking",
    "I need to check with the team before",
    "The flight leaves at 6am so we should",
    "Everything looks good on my end",
    "Give me a second to think about",
    "We finished the audit last week",
    "I'm going to need more time on",
    "Sounds good, talk soon",
    "The numbers don't quite add up, specifically",
    "Let's circle back to this after",
    "I really appreciate you taking the time to",
    "That should cover most of the requirements",
    "I'm not fully convinced that",
    "We can revisit this decision once",
    "Thanks, that answers my question",
    "The client wants three changes: first, second, and",
    "I think that wraps up everything for today",
]


def run_spike():
    results = []
    print(f"Model under test: {MODEL}")
    print(f"reasoning_effort: {REASONING_EFFORT}")
    print(f"Running latency spike: {len(TEST_PROMPTS)} calls, "
          f"~{SECONDS_BETWEEN_CALLS}s apart (widened pacing to rule out "
          f"rate-limit-window interaction)\n")

    if WARMUP_CALLS > 0:
        print(f"Running {WARMUP_CALLS} warmup call(s) to absorb connection "
              f"cold-start (discarded from stats)...")
        for _ in range(WARMUP_CALLS):
            try:
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": 'Transcript fragment: "warmup"'},
                    ],
                    reasoning_effort=REASONING_EFFORT,
                    max_tokens=5,
                    temperature=0,
                )
            except Exception as e:
                print(f"  warmup call failed (non-fatal): {e}")
            time.sleep(SECONDS_BETWEEN_CALLS)
        print("Warmup done.\n")

    for i, prompt in enumerate(TEST_PROMPTS, start=1):
        start = time.perf_counter()
        error = None
        response_text = None
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Transcript fragment: "{prompt}"'},
                ],
                reasoning_effort=REASONING_EFFORT,
                max_tokens=5,
                temperature=0,
            )
            response_text = completion.choices[0].message.content.strip()
        except Exception as e:
            error = str(e)

        elapsed_ms = (time.perf_counter() - start) * 1000
        results.append({
            "index": i,
            "prompt": prompt,
            "elapsed_ms": round(elapsed_ms, 1),
            "response": response_text,
            "error": error,
            "over_deadline": elapsed_ms > DEADLINE_MS,
        })

        status = f"{elapsed_ms:.0f}ms" if not error else f"ERROR: {error[:60]}"
        flag = " [OVER 400ms DEADLINE]" if elapsed_ms > DEADLINE_MS else ""
        print(f"[{i:2d}/{len(TEST_PROMPTS)}] {status}{flag}  -> {response_text}")

        if i < len(TEST_PROMPTS):
            time.sleep(SECONDS_BETWEEN_CALLS)

    return results


def summarize(results):
    successful = [r["elapsed_ms"] for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    print("\n=== Latency Spike Summary ===")
    print(f"Total calls: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors/timeouts: {len(errors)}")

    if successful:
        p50 = statistics.median(successful)
        p95 = statistics.quantiles(successful, n=100)[94] if len(successful) >= 20 else max(successful)
        mean = statistics.mean(successful)
        print(f"\nLatency (ms) over {len(successful)} successful calls:")
        print(f"  min:  {min(successful):.1f}")
        print(f"  p50:  {p50:.1f}")
        print(f"  p95:  {p95:.1f}")
        print(f"  max:  {max(successful):.1f}")
        print(f"  mean: {mean:.1f}")

        print("\n=== DECISION GATE (PRD sec 2.2: 250ms p95 budget) ===")
        if p95 <= 250:
            print(f"PASS - p95 ({p95:.1f}ms) is within the 250ms budget.")
            print("Proceed to Phase 1 as planned.")
        else:
            print(f"FAIL - p95 ({p95:.1f}ms) EXCEEDS the 250ms budget.")
            print("Per PRD sec 2.2, this requires one of:")
            print("  1. Rebalance the sec 2.2 latency budget (e.g. reduce")
            print("     other stages' allocations if they have headroom)")
            print("  2. Switch the semantic step to gpt-oss-20b (already")
            print("     confirmed available) and re-run this spike")
            print("  3. Move the semantic step to a local small model")
            print("This decision MUST be made before Phase 1 begins.")
    else:
        print("\nNo successful calls - cannot compute latency stats.")
        print("Check API key, network, and error messages above.")

    if errors:
        print(f"\n{len(errors)} call(s) failed - review errors above.")

    return successful


def save_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "prompt", "elapsed_ms", "response", "error", "over_deadline"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRaw results saved to {path}")


if __name__ == "__main__":
    results = run_spike()
    summarize(results)
    safe_model_name = MODEL.replace("/", "_")
    save_csv(results, f"eval/latency_spike_results_{safe_model_name}.csv")
    print(f"\nRun completed at {datetime.now().isoformat()}")
