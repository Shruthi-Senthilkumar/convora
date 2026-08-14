"""
detection/transcribe_batch.py

Phase 1, Task 1: Transcribe an audio file with word-level timestamps
using Deepgram's batch (prerecorded) API.

NOTE: Written against deepgram-sdk v7.x, which is a full rewrite of the
SDK's API surface compared to earlier v3-era examples (no more
PrerecordedOptions/FileSource/listen.rest.v("1") - now
client.listen.v1.media.transcribe_file(request=bytes, **kwargs)).

This is the first building block of EndOfSpeechDetector.process_file().
Output includes word-level start/end timestamps, needed later for:
  - pause duration calculation (fusion signal 1, PRD sec 2.2)
  - aligning semantic judgment calls to specific transcript boundaries

Run standalone to test against a sample file:
    python detection/transcribe_batch.py "path/to/audio.wav"
"""

import os
import sys
import json
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()


def transcribe_file(audio_path: str) -> dict:
    """
    Transcribes an audio file via Deepgram's batch API.

    Returns a dict with:
        - transcript: full text
        - words: list of {word, start, end, confidence} dicts
        - duration: audio duration in seconds
    """
    client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        smart_format=True,
        punctuate=True,
        # word-level timestamps are included by default in the response
    )

    result = response.results.channels[0].alternatives[0]

    words = [
        {
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "confidence": w.confidence,
        }
        for w in result.words
    ]

    return {
        "transcript": result.transcript,
        "words": words,
        "duration": response.metadata.duration,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detection/transcribe_batch.py <audio_file_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    print(f"Transcribing: {audio_path}")
    result = transcribe_file(audio_path)

    print(f"\nDuration: {result['duration']:.1f}s")
    print(f"Word count: {len(result['words'])}")
    print(f"\nFirst 200 chars of transcript:\n{result['transcript'][:200]}...")

    print(f"\nFirst 10 words with timestamps:")
    for w in result["words"][:10]:
        print(f"  {w['word']:<15} {w['start']:.2f}s - {w['end']:.2f}s  (conf: {w['confidence']:.2f})")

    out_path = "eval/transcribe_batch_test_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to {out_path}")
