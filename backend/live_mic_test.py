"""
backend/live_mic_test.py

Phase 2, Live Microphone Test - Diagnostic client capturing live mic audio.
PRD - 5, Phase 2, Live mic test.

Captures audio from the default or specified microphone using `sounddevice`
and streams raw 16kHz 16-bit mono PCM chunks (~30ms each) over WebSockets
to the /ws/transcribe endpoint, printing partial/final transcripts and
end_of_speech_candidate events in real time.

Usage:
    # List available input devices
    python backend/live_mic_test.py --list-devices

    # Run with default microphone until Ctrl+C
    python backend/live_mic_test.py

    # Run for a fixed duration (e.g. 10 seconds) with a specific device index
    python backend/live_mic_test.py --device 1 --duration 10
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

import sounddevice as sd
import websockets

# ---------------------------------------------------------------------------
# Configuration - matching backend/main.py Deepgram expectations exactly
# ---------------------------------------------------------------------------
DEFAULT_SERVER_URI = "ws://localhost:8000/ws/transcribe"

SAMPLE_RATE_HZ = 16000    # 16 kHz
CHANNELS = 1              # Mono
DTYPE = "int16"           # 16-bit PCM integer

# Chunk size: ~30 ms of audio at 16 kHz = 480 samples = 960 bytes
CHUNK_DURATION_MS = 30
SAMPLES_PER_CHUNK = (SAMPLE_RATE_HZ * CHUNK_DURATION_MS) // 1000  # 480

DRAIN_WAIT_S = 3.0


def list_audio_devices() -> None:
    """Print available audio input/output devices and exit."""
    print("Available Audio Devices:")
    print("=" * 60)
    print(sd.query_devices())
    print("=" * 60)
    print("Use --device <index_or_name> to select a specific input device.")


async def capture_and_stream(
    server_uri: str,
    device: Optional[int | str] = None,
    duration: Optional[float] = None,
) -> None:
    """
    Captures live audio from the microphone using sounddevice.RawInputStream,
    streams raw PCM chunks over WebSockets, and prints received transcripts.
    """
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def audio_callback(indata: bytes, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback executed by sounddevice in an audio thread for each block."""
        if status:
            print(f"[sounddevice warning] {status}", file=sys.stderr)
        # Thread-safe put of raw bytes into the asyncio Queue
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    device_info = sd.query_devices(device, "input")
    device_name = device_info.get("name", "Default")
    print(f"Input Device : [{device if device is not None else 'default'}] {device_name}")
    print(f"Audio Format : {SAMPLE_RATE_HZ} Hz, {CHANNELS}ch, 16-bit PCM")
    print(f"Chunking     : {CHUNK_DURATION_MS} ms ({SAMPLES_PER_CHUNK} samples per chunk)")
    print(f"Server URI   : {server_uri}")
    if duration is not None:
        print(f"Duration     : {duration:.1f} seconds max")
    else:
        print("Duration     : Continuous (press Ctrl+C to stop)")
    print("-" * 60)

    try:
        async with websockets.connect(server_uri) as ws:
            print("Connected to WebSocket server.")
            print("Listening for speech... (speak into your microphone)")
            print("-" * 60)

            msg_count = 0
            chunks_sent = 0
            start_wall = time.monotonic()
            stop_event = asyncio.Event()

            # Open RawInputStream for direct 16-bit PCM byte capture
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE_HZ,
                blocksize=SAMPLES_PER_CHUNK,
                device=device,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=audio_callback,
            )

            async def send_audio() -> None:
                nonlocal chunks_sent
                with stream:
                    while not stop_event.is_set():
                        if duration is not None:
                            elapsed = time.monotonic() - start_wall
                            if elapsed >= duration:
                                print(f"\n[Reached duration limit of {duration:.1f}s]")
                                break

                        try:
                            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue

                        await ws.send(chunk)
                        chunks_sent += 1

                # Signal end-of-stream cleanly to server
                try:
                    await ws.send(json.dumps({"type": "stop"}))
                except Exception:
                    pass
                print(f"\n[Mic capture stopped - sent {chunks_sent} chunks ({chunks_sent * CHUNK_DURATION_MS / 1000:.1f}s audio), draining...]")

            async def receive_transcripts() -> None:
                nonlocal msg_count
                try:
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
                except websockets.exceptions.ConnectionClosed:
                    pass

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_transcripts())

            try:
                await send_task
            except asyncio.CancelledError:
                stop_event.set()

            # Drain time for final events
            try:
                await asyncio.wait_for(recv_task, timeout=DRAIN_WAIT_S)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                    pass

            elapsed = time.monotonic() - start_wall
            print("-" * 60)
            print(f"Done. Sent {chunks_sent} chunks, received {msg_count} transcript messages in {elapsed:.1f}s.")

    except websockets.exceptions.ConnectionClosed:
        pass
    except websockets.exceptions.WebSocketException as e:
        print(f"\nERROR: WebSocket connection failed: {e}")
        print("Ensure the server is running (`uvicorn backend.main:app --port 8000`)")
    except Exception as e:
        print(f"\nERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convora Live Microphone Diagnostic Client (Phase 2)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Audio input device index or name substring (default: system default).",
    )
    parser.add_argument(
        "-t",
        "--duration",
        type=float,
        default=None,
        help="Maximum recording duration in seconds (default: run until Ctrl+C).",
    )
    parser.add_argument(
        "-s",
        "--server",
        type=str,
        default=DEFAULT_SERVER_URI,
        help=f"Server WebSocket URI (default: {DEFAULT_SERVER_URI}).",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # Convert device arg to int if it's digit string
    device_val = args.device
    if device_val is not None and device_val.isdigit():
        device_val = int(device_val)

    try:
        asyncio.run(capture_and_stream(args.server, device_val, args.duration))
    except KeyboardInterrupt:
        print("\n[Interrupted by user (Ctrl+C)]")


if __name__ == "__main__":
    main()
