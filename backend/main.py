"""
backend/main.py

Phase 2, Task 1 - FastAPI WebSocket streaming endpoint.
PRD - 5, Phase 2, Tasks 1-2.

Architecture (this file only - no pause detection yet):
  - Client connects to /ws/transcribe and sends binary audio chunks.
  - Each chunk is forwarded to Deepgram's live-transcription WebSocket via
    AsyncDeepgramClient.listen.v1.connect().
  - Deepgram returns ListenV1Results messages; we relay them back to the
    client as JSON:
        {"type": "partial_transcript", "text": "...", "timestamp_s": <float>}
        {"type": "final_transcript",   "text": "...", "timestamp_s": <float>}
  - Client disconnect is caught and Deepgram connection closed cleanly.
  - Per-connection Deepgram errors are logged and surfaced to the client;
    the server itself keeps running for other connections.

deepgram-sdk v7.x streaming API pattern confirmed:
    async with client.listen.v1.connect(model=..., interim_results=True, ...)
        as socket:           # socket is AsyncV1SocketClient
        await socket.send_media(chunk)
        async for msg in socket:   # yields ListenV1Results | ListenV1Metadata | ...
            ...

Pause detection / SemanticJudge wiring is intentionally absent here;
that is Phase 2 Tasks 3-4.
"""

import asyncio
import json
import logging
import os
import time
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from backend.file_upload_api import router as file_upload_router
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types import ListenV1Results, ListenV1UtteranceEnd
from backend.live_pause_tracker import LivePauseTracker

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deepgram configuration
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY: str = os.environ["DEEPGRAM_API_KEY"]

# Nova-3 is the latest production model (same as batch path).
# interim_results=True gives us partial transcripts as audio arrives.
# endpointing controls how aggressively Deepgram finalises segments;
# 200 ms is a good default for conversational audio.
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_SAMPLE_RATE = 16000    # 16 kHz - matches AMI corpus WAV
DEEPGRAM_ENCODING = "linear16"  # Raw PCM 16-bit little-endian
DEEPGRAM_CHANNELS = 1

app = FastAPI(title="Convora Streaming Backend")
app.include_router(file_upload_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket):
    """
    WebSocket endpoint for live audio transcription and turn-detection.

    Protocol (client -> server):
        Binary frames: raw PCM audio chunks (16-bit, 16 kHz, mono).
          Send them as fast as they are produced (mic latency or faster).
        Text frame containing {"type": "stop"} to signal end-of-stream
          gracefully (optional - plain disconnect also works).

    Protocol (server -> client):
        {"type": "partial_transcript", "text": "...", "timestamp_s": <float>}
        {"type": "final_transcript",   "text": "...", "timestamp_s": <float>}
        {"type": "end_of_speech_candidate",
         "timestamp_s": <float>,
         "is_end_of_speech": <bool>,
         "confidence": <float>,
         "fragment": <str>,
         "contributing_signals": <dict>,
         "detection_latency_ms": <float>}
        {"type": "error",              "detail": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    dg_client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
    tracker = LivePauseTracker()
    ws_lock = asyncio.Lock()

    try:
        async with dg_client.listen.v1.connect(
            model=DEEPGRAM_MODEL,
            sample_rate=DEEPGRAM_SAMPLE_RATE,
            encoding=DEEPGRAM_ENCODING,
            channels=DEEPGRAM_CHANNELS,
            interim_results=True,
            endpointing=200,
            utterance_end_ms=1000,
            diarize=True,
            smart_format=True,
            punctuate=True,
        ) as dg_socket:
            logger.info("Deepgram streaming connection established (diarization enabled)")

            # Run the receiver loop and the sender loop concurrently.
            # _receive_from_client: pulls audio chunks from the client and
            #   forwards them to Deepgram.
            # _receive_from_deepgram: pulls messages from Deepgram and
            #   sends them back to the browser client.
            # We cancel whichever task is still running when the other ends.
            sender_task = asyncio.create_task(
                _receive_from_client(websocket, dg_socket)
            )
            receiver_task = asyncio.create_task(
                _receive_from_deepgram(websocket, dg_socket, tracker, ws_lock)
            )

            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the remaining task and drain it cleanly.
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # Re-raise any unexpected exception from the completed task.
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    raise exc

            # Cleanly signal end-of-stream to Deepgram.
            try:
                await dg_socket.send_close_stream()
            except Exception:
                pass  # already handled by context manager exit

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:
        logger.error("Error in /ws/transcribe: %s", exc, exc_info=True)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "detail": str(exc)})
            )
        except Exception:
            pass
    finally:
        logger.info("WebSocket handler exiting")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _receive_from_client(websocket: WebSocket, dg_socket) -> None:
    """
    Pump audio chunks from the FastAPI WebSocket into Deepgram.

    Exits when the client disconnects or sends the stop control message.
    """
    while True:
        try:
            data = await websocket.receive()
        except WebSocketDisconnect:
            logger.info("Client disconnected (receive loop)")
            return

        if "bytes" in data and data["bytes"]:
            await dg_socket.send_media(data["bytes"])
        elif "text" in data and data["text"]:
            try:
                msg = json.loads(data["text"])
                if msg.get("type") == "stop":
                    logger.info("Client sent stop signal")
                    await dg_socket.send_finalize()
                    return
            except json.JSONDecodeError:
                logger.warning("Received non-JSON text frame: %s", data["text"][:80])


async def _receive_from_deepgram(
    websocket: WebSocket,
    dg_socket,
    tracker: LivePauseTracker,
    ws_lock: asyncio.Lock
) -> None:
    """
    Pump Deepgram transcript messages back to the FastAPI WebSocket client.
    Appends finalized words to rolling buffer and evaluates turn-detection candidates.
    """
    async for message in dg_socket:
        recv_time = time.perf_counter()

        if isinstance(message, ListenV1Results):
            transcript_text: str = message.channel.alternatives[0].transcript

            # 1. Relay transcript immediately so downstream logic never blocks transcription
            if transcript_text.strip():
                msg_type = "final_transcript" if message.is_final else "partial_transcript"
                payload = {
                    "type": msg_type,
                    "text": transcript_text,
                    "timestamp_s": message.start,
                }
                try:
                    async with ws_lock:
                        await websocket.send_text(json.dumps(payload))
                except WebSocketDisconnect:
                    logger.info("Client disconnected (send loop)")
                    return
                except Exception as exc:
                    logger.error("Failed to send transcript to client: %s", exc)
                    return

            # 2. Check for pause candidates on final transcript chunks
            if message.is_final:
                _, candidates = tracker.register_words_and_detect_candidates(message)
                for candidate in candidates:
                    # Run heavy CPU/network-bound semantic judgment and fusion in background task
                    asyncio.create_task(
                        _evaluate_and_send_candidate(candidate, tracker, websocket, ws_lock, recv_time)
                    )

        elif isinstance(message, ListenV1UtteranceEnd):
            logger.info("Received ListenV1UtteranceEnd signal (last_word_end=%.2fs)", message.last_word_end)
            candidates = tracker.detect_utterance_end_candidate(message.last_word_end)
            for candidate in candidates:
                asyncio.create_task(
                    _evaluate_and_send_candidate(candidate, tracker, websocket, ws_lock, recv_time)
                )


async def _evaluate_and_send_candidate(
    candidate: Dict[str, Any],
    tracker: LivePauseTracker,
    websocket: WebSocket,
    ws_lock: asyncio.Lock,
    recv_time: float
) -> None:
    """
    Worker task to evaluate a pause candidate and send the result back to the client.
    Uses asyncio.to_thread to keep the event loop non-blocking.
    """
    last_word_idx = candidate["last_word_index"]
    fragment, speaker = tracker.build_fragment(last_word_idx)
    if not fragment:
        return

    try:
        # Offload CPU/network-bound spacy and LLM calls to thread pool
        fusion_res, judge_fuse_ms = await asyncio.to_thread(
            tracker.evaluate_candidate_sync,
            fragment,
            candidate["pause_duration_s"],
            candidate["speaker_changed"]
        )
    except Exception as exc:
        logger.error("Error in background candidate evaluation: %s", exc, exc_info=True)
        return

    # If it is resolved to EOS, mark it in the tracker to stop further checks on this boundary
    if fusion_res.is_end_of_speech:
        tracker.mark_boundary_resolved(candidate["timestamp_s"])

    # Real end-to-end latency: from receiving the Deepgram message to right before sending
    detection_latency_ms = (time.perf_counter() - recv_time) * 1000

    payload = {
        "type": "end_of_speech_candidate",
        "timestamp_s": candidate["timestamp_s"],
        "is_end_of_speech": fusion_res.is_end_of_speech,
        "confidence": fusion_res.confidence,
        "speaker": speaker,
        "speaker_changed": candidate["speaker_changed"],
        "fragment": fragment,
        "contributing_signals": fusion_res.contributing_signals,
        "detection_latency_ms": detection_latency_ms
    }

    logger.info(
        "Candidate evaluated [Source: %s]: %r -> is_eos=%s (conf=%.2f, latency: %.1fms, judge_fuse: %.1fms)",
        candidate["source"],
        fragment,
        fusion_res.is_end_of_speech,
        fusion_res.confidence,
        detection_latency_ms,
        judge_fuse_ms
    )

    try:
        async with ws_lock:
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("Failed to send candidate event to client: %s", exc)


