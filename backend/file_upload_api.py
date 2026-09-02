"""
backend/file_upload_api.py

Phase 4 - File upload mode (batch testing via UI), PRD sec 5 Phase 4.

A separate, standalone FastAPI router for the frontend's file-upload
mode. Kept deliberately separate from backend/main.py (which owns the
live WebSocket streaming endpoint) so this can be reviewed and wired
in without touching the already-verified live pipeline.

Wraps the existing, already-tested batch pipeline
(detection/end_of_speech_detector.py) - no new detection logic here,
purely an HTTP interface over what Phase 1 already built and verified.

TO WIRE THIS IN, add to backend/main.py:

    from backend.file_upload_api import router as file_upload_router
    app.include_router(file_upload_router)

(Do this manually - this file intentionally does not modify main.py.)
"""

import os
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from detection.end_of_speech_detector import EndOfSpeechDetector

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm",
    ".aac", ".amr", ".wma", ".mp4", ".mov", ".3gp",
}
MAX_FILE_SIZE_MB = 100


@router.post("/api/process-file")
async def process_file(file: UploadFile = File(...)):
    """
    Accepts an uploaded audio file, runs it through the existing batch
    EndOfSpeechDetector pipeline, and returns the same spec-compliant
    JSON schema documented in detection/end_of_speech_detector.py.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Stream to a temp file rather than loading the whole upload into
    # memory - matters for anything beyond very short clips.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.1f}MB, max {MAX_FILE_SIZE_MB}MB).",
            )

        detector = EndOfSpeechDetector()
        result = detector.process_file(tmp_path)
        # process_file returns the audio path it was given (a temp
        # path here) - swap in the user's original filename so the
        # response is meaningful to them.
        result["audio_file"] = file.filename

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
