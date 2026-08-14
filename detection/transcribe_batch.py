"""
detection/transcribe_batch.py

Phase 1, Task 1: Transcribe an audio file with word-level timestamps
AND speaker diarization using Deepgram's batch (prerecorded) API.

Diarization added after Phase 1 Task 2 testing revealed that AMI's
multi-party audio produces nonsensical cross-speaker fragments when
pause-candidate windows aren't speaker-aware.

Extended timeout added after a WriteTimeout on a ~39MB file upload -
the SDK's default timeout is too short for large file bodies on
typical upload speeds.

Written against deepgram-sdk v7.x.

Run standalone to test against a sample file:
    python detection/transcribe_batch.py "path/to/audio.wav"
"""

import os
import sys
import json
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

UPLOAD_TIMEOUT_S = 300  # 5 minutes - generous for large file uploads
                          # on typical connections; batch transcription
                          # is not latency-sensitive like the live path


def transcribe_file(audio_path: str) -> dict:
    """
    Transcribes an audio file via Deepgram's batch API, with diarization.

    Returns a dict with:
        - transcript: full text
        - words: list of {word, start, end, confidence, speaker} dicts
        - duration: audio duration in seconds
    """
    client = DeepgramClient(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        timeout=UPLOAD_TIMEOUT_S,
    )

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    file_size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"Uploading {file_size_mb:.1f}MB (timeout set to {UPLOAD_TIMEOUT_S}s)...")

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        smart_format=True,
        punctuate=True,
        diarize=True,
    )

    result = response.results.channels[0].alternatives[0]

    words = [
        {
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "confidence": w.confidence,
            "speaker": getattr(w, "speaker", None),
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

    print(f"Transcribing (with diarization): {audio_path}")
    result = transcribe_file(audio_path)

    print(f"\nDuration: {result['duration']:.1f}s")
    print(f"Word count: {len(result['words'])}")

    speakers_seen = set(w["speaker"] for w in result["words"] if w["speaker"] is not None)
    print(f"Speakers detected: {sorted(speakers_seen) if speakers_seen else 'NONE - diarization may not be available'}")

    print(f"\nFirst 200 chars of transcript:\n{result['transcript'][:200]}...")

    print(f"\nFirst 15 words with timestamps + speaker:")
    for w in result["words"][:15]:
        print(f"  spk{w['speaker']}  {w['word']:<15} {w['start']:.2f}s - {w['end']:.2f}s")

    out_path = "eval/transcribe_batch_test_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to {out_path}")
