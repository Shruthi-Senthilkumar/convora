"""
backend/test_client.py

Phase 2, Task 2 - Standalone test client for the /ws/transcribe endpoint.
PRD - 5, Phase 2, Task 2.

Streams a WAV file in realistic-sized chunks (~30 ms of audio per send)
to the local FastAPI server and prints every transcript message received.

This verifies the full loop (browser client -> FastAPI -> Deepgram -> back)
without needing a real microphone or a frontend.

Usage (run after starting the server with uvicorn):
    python backend/test_client.py [path/to/audio.wav]

Default audio file if none is given:
    C:/Users/shrut/ami-corpus-data/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
"""

import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URI = "ws://localhost:8000/ws/transcribe"

DEFAULT_AUDIO = (
    r"C:\Users\shrut\ami-corpus-data\amicorpus\ES2002a\audio\ES2002a.Mix-Headset.wav"
)

# Chunk size: ~30 ms of audio at 16 kHz, 16-bit mono = 960 samples = 1920 bytes.
# This mimics real mic-buffer sizes and prevents the server being flooded with
# one giant blob.
SAMPLE_RATE_HZ = 16000
CHUNK_DURATION_MS = 30
SAMPLES_PER_CHUNK = (SAMPLE_RATE_HZ * CHUNK_DURATION_MS) // 1000   # 480
BYTES_PER_SAMPLE = 2  # 16-bit PCM
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE              # 960

# Stop after this many seconds of audio to keep the test run short.
# Set to None to stream the entire file.
MAX_AUDIO_SECONDS: float | None = 60.0

# How long to wait for final transcripts to arrive after the audio ends.
DRAIN_WAIT_S = 5.0


async def stream_audio(audio_path: str) -> None:
    """
    Open the WAV file, connect to the WebSocket server, stream audio in chunks,
    and print every transcript message received.
    """
    wav_path = Path(audio_path)
    if not wav_path.exists():
        print(f"ERROR: Audio file not found: {wav_path}")
        sys.exit(1)

    print(f"Audio file : {wav_path.name}")

    with wave.open(str(wav_path), "rb") as wf:
        file_sample_rate = wf.getframerate()
        file_channels = wf.getnchannels()
        file_sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        duration_s = total_frames / file_sample_rate

        print(f"WAV info   : {file_sample_rate} Hz, {file_channels}ch, "
              f"{file_sampwidth*8}-bit, {duration_s:.1f}s total")

        if file_sample_rate != SAMPLE_RATE_HZ or file_channels != 1 or file_sampwidth != 2:
            print(
                f"WARNING: WAV parameters ({file_sample_rate} Hz, {file_channels}ch, "
                f"{file_sampwidth*8}-bit) differ from expected "
                f"({SAMPLE_RATE_HZ} Hz, 1ch, 16-bit). "
                "Transcription quality may suffer. "
                "Convert with ffmpeg if needed: "
                "ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )

        # How many chunks to send before stopping.
        if MAX_AUDIO_SECONDS is not None:
            max_chunks = int((MAX_AUDIO_SECONDS * file_sample_rate) / SAMPLES_PER_CHUNK)
            print(f"Streaming  : first {MAX_AUDIO_SECONDS}s ({max_chunks} chunks of {CHUNK_DURATION_MS}ms)")
        else:
            max_chunks = None
            print(f"Streaming  : entire file ({total_frames // SAMPLES_PER_CHUNK} chunks)")

        print(f"Server     : {SERVER_URI}")
        print("-" * 60)

        async with websockets.connect(SERVER_URI) as ws:
            print("Connected to server.")
            msg_count = 0
            start_wall = time.monotonic()

            async def send_audio() -> None:
                chunk_index = 0
                while True:
                    if max_chunks is not None and chunk_index >= max_chunks:
                        break
                    raw = wf.readframes(SAMPLES_PER_CHUNK)
                    if not raw:
                        break
                    await ws.send(raw)
                    chunk_index += 1
                    # Real-time pacing: sleep for the duration of each chunk.
                    # This simulates live microphone input and prevents
                    # saturating Deepgram's receive buffer.
                    await asyncio.sleep(CHUNK_DURATION_MS / 1000)

                # Signal end-of-stream.
                await ws.send(json.dumps({"type": "stop"}))
                print(f"\n[Sender done - sent {chunk_index} chunks, "
                      f"waiting {DRAIN_WAIT_S}s for remaining transcripts...]")

            async def receive_transcripts() -> None:
                nonlocal msg_count
                deadline = None
                async for raw_msg in ws:
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type", "unknown")
                    text = msg.get("text", "")
                    ts = msg.get("timestamp_s", 0.0)
                    wall = time.monotonic() - start_wall

                    if msg_type in ("partial_transcript", "final_transcript"):
                        label = "FINAL  " if msg_type == "final_transcript" else "partial"
                        print(f"[{wall:6.1f}s wall | audio @{ts:6.2f}s] "
                              f"{label}  {text!r}")
                        msg_count += 1
                    elif msg_type == "end_of_speech_candidate":
                        is_eos = msg.get("is_end_of_speech", False)
                        confidence = msg.get("confidence", 0.0)
                        fragment = msg.get("fragment", "")
                        lat = msg.get("detection_latency_ms", 0.0)
                        spk = msg.get("speaker")
                        spk_chg = msg.get("speaker_changed", False)

                        if spk is not None:
                            spk_tag = f"spk{spk}*" if spk_chg else f"spk{spk} "
                        else:
                            spk_tag = "spk?*" if spk_chg else "spk? "

                        decision = "EOS" if is_eos else "CONT"
                        print(f"[{wall:6.1f}s wall | audio @{ts:6.2f}s] "
                              f"CANDIDATE: {decision:<4} {spk_tag:<5} (conf: {confidence:.2f}, latency: {lat:.1f}ms) "
                              f"fragment: {fragment!r}")
                        msg_count += 1
                    elif msg_type == "error":
                        print(f"[ERROR from server] {msg.get('detail')}")
                    else:
                        print(f"[{msg_type}] {raw_msg[:120]}")

            # Run send and receive concurrently; wait for both to finish.
            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_transcripts())

            # Wait for sender to finish, then give receiver time to drain.
            await send_task

            # Give Deepgram time to flush final transcripts.
            try:
                await asyncio.wait_for(recv_task, timeout=DRAIN_WAIT_S)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass  # expected - recv loop stays alive until server closes
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                    pass

            elapsed = time.monotonic() - start_wall
            print("-" * 60)
            print(f"Done. Received {msg_count} transcript messages in {elapsed:.1f}s.")


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    asyncio.run(stream_audio(audio_file))

