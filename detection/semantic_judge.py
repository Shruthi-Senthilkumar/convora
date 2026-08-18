"""
detection/semantic_judge.py

Semantic end-of-speech judgment: rule-based gate first (near-zero latency,
56% coverage per Phase 0 testing), escalates to hosted LLM (Groq) only for
ambiguous fragments. Latency budget (PRD sec 2.2, 250ms p95) is NOT fully
met by the LLM path - accepted as a known limitation for now, see
docs/latency_spike_findings.md. Hard deadline + degraded fallback protect
against worst-case stalls.
"""

import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq
import spacy

load_dotenv()

nlp = spacy.load("en_core_web_sm")


@dataclass
class JudgeResult:
    label: str          # "complete" | "incomplete"
    confidence: float
    source: str          # "rule" | "llm" | "degraded"
    latency_ms: float


# --- Rule-based gate (Phase 0 validated: 56% coverage on test set) ---

BACKCHANNELS = {
    "mm-hmm", "mhm", "uh-huh", "right", "yeah", "yep", "yup", "ok", "okay",
    "sure", "got it", "i see", "gotcha", "true", "totally", "for sure",
    "i guess",
}
INCOMPLETE_TRAILING_POS = {"CCONJ", "SCONJ", "ADP", "DET", "PART"}
SELF_CORRECTION_MARKERS = {"actually", "wait", "sorry", "i mean"}
TRAILING_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those",
                         "my", "your", "his", "her", "its", "our", "their"}
TRAILING_AUX_MODAL = {
    "should", "would", "could", "was", "were", "is", "are", "will",
    "must", "might", "can", "do", "does", "did", "has", "have", "had",
    "shall", "may",
}
TRAILING_LIST_MARKERS = {"namely", "specifically", "finally", "lastly", "firstly"}


def _normalize(text: str) -> str:
    return text.strip().strip(".,!?").lower()


def _rule_based_classify(fragment: str):
    """Returns (label, confident: bool)."""
    norm = _normalize(fragment)
    if norm in BACKCHANNELS:
        return "incomplete", True

    doc = nlp(fragment.strip())
    if len(doc) == 0:
        return "ambiguous", False

    last_token = doc[-1]
    last_text_lower = last_token.text.lower()

    if last_text_lower in BACKCHANNELS:
        return "incomplete", True
    if last_text_lower in TRAILING_AUX_MODAL:
        return "incomplete", True
    if last_text_lower in TRAILING_LIST_MARKERS:
        return "incomplete", True
    if last_token.pos_ in INCOMPLETE_TRAILING_POS:
        return "incomplete", True
    if last_text_lower in TRAILING_DETERMINERS:
        return "incomplete", True
    if last_text_lower in SELF_CORRECTION_MARKERS:
        return "incomplete", True

    root = [t for t in doc if t.dep_ == "ROOT"]
    if root:
        root_tok = root[0]
        has_subject = any(c.dep_ in ("nsubj", "nsubjpass") for c in root_tok.children)
        if root_tok.pos_ == "VERB" and not has_subject and len(doc) > 2:
            return "ambiguous", False

    if last_token.pos_ == "PUNCT" and last_token.text == ".":
        return "complete", True

    return "ambiguous", False


# --- LLM escalation path (Groq, hosted) ---

SYSTEM_PROMPT = (
    "You are judging whether a speaker has finished their conversational "
    "turn based on a transcript fragment. Respond with ONLY one word: "
    "'complete' if the turn sounds finished, or 'incomplete' if the "
    "speaker seems likely to continue. No explanation."
)

HARD_DEADLINE_S = 0.4  # PRD sec 2.2: 400ms hard deadline before fallback


class SemanticJudge:
    def __init__(self, model="qwen/qwen3.6-27b"):
        # timeout set on the CLIENT itself so it actually aborts the
        # request at the deadline, rather than letting create() run to
        # completion and only checking the elapsed time afterward
        # max_retries=0 prevents the client from retrying on timeout/rate-limit,
        # ensuring the fallback is returned within the hard deadline.
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=HARD_DEADLINE_S,
            max_retries=0,
        )
        self.model = model

    def judge(self, transcript_fragment: str) -> JudgeResult:
        start = time.perf_counter()

        # 1. Try the rule gate first - zero latency if confident
        label, confident = _rule_based_classify(transcript_fragment)
        if confident:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return JudgeResult(label=label, confidence=0.9, source="rule", latency_ms=elapsed_ms)

        # 2. Escalate to LLM, with hard deadline + degraded fallback
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Transcript fragment: "{transcript_fragment}"'},
                ],
                reasoning_effort="none",
                max_tokens=10,
                temperature=0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            llm_label = completion.choices[0].message.content.strip().lower()
            if llm_label not in ("complete", "incomplete"):
                llm_label = "incomplete"  # safe default if the model rambles
            return JudgeResult(label=llm_label, confidence=0.75, source="llm", latency_ms=elapsed_ms)
        except Exception as e:
            # Hit the client-level timeout, or another API error.
            # Degraded fallback: pause-duration-only decision would plug in
            # here in the real fusion layer. For now, safe default.
            elapsed_ms = (time.perf_counter() - start) * 1000
            return JudgeResult(label="incomplete", confidence=0.3, source="degraded", latency_ms=elapsed_ms)
