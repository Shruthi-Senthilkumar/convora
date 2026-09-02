# Convora frontend (Phase 4 - Demo UI v1)

## What's genuinely verified vs. what needs your machine

Built and verified in a sandboxed environment with no access to your
actual backend or a real microphone. Here's exactly what that means:

**Verified for real:**
- `npx tsc --noEmit` - clean, zero type errors
- `npm run build` - full production build succeeds (fonts stubbed
  temporarily to work around this sandbox's lack of internet access
  to fonts.googleapis.com, then restored - the font-fetch itself
  couldn't be tested here, but it's a standard public Google Fonts
  CDN call that works from any normal internet connection)
- All WebSocket message types match `backend/main.py`'s real,
  already-tested protocol exactly (checked by reading that file, not
  guessed)
- Audio resampling math (44.1/48kHz float32 -> 16kHz int16 PCM) is
  implemented correctly per the format `backend/live_mic_test.py`
  already proved works, but has NOT been tested against a real
  microphone or a real running backend from here - that can only
  happen on your machine.

**You need to verify:**
1. Live mic capture actually works and streams real audio
2. The backend actually receives and transcribes it correctly
3. The file-upload endpoint actually processes a real file end to end

## A known, honest tradeoff: dependency vulnerabilities

`npm audit` reports vulnerabilities across the Next.js 14.x line
(image optimizer DoS, RSC caching issues, etc.) - the only full fix
is a major-version jump to Next.js 16, a real breaking change not
made silently here. For a local demo tool this isn't running with
untrusted public traffic, so the practical risk is low, but this is
noted rather than hidden. Run `npm audit` yourself and decide whether
to upgrade before any real deployment.

## Setup

```bash
cd frontend
npm install
```

Create `.env.local` (not committed):
```
NEXT_PUBLIC_CONVORA_WS_URL=ws://localhost:8000/ws/transcribe
NEXT_PUBLIC_CONVORA_API_URL=http://localhost:8000
```

## Wiring in the file-upload backend endpoint

`file_upload_api.py` (in this delivery) is a SEPARATE FastAPI router,
deliberately not merged into `backend/main.py` automatically. To wire
it in:

1. Copy `file_upload_api.py` into your `backend/` folder.
2. In `backend/main.py`, add:
   ```python
   from backend.file_upload_api import router as file_upload_router
   app.include_router(file_upload_router)
   ```
3. You'll also need `python-multipart` installed (FastAPI's file
   upload dependency):
   ```powershell
   pip install python-multipart
   ```

## Running it

Terminal 1 - backend (with the router wired in per above):
```powershell
uvicorn backend.main:app --port 8000
```

Terminal 2 - frontend:
```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`. Click "Start listening," allow
microphone access, and talk - the signal instrument should light up
with real transcript and end-of-speech events. Switch to "Upload a
file" to test batch mode against a recording.

## Responsive design

Real breakpoints applied throughout (Tailwind `sm:`/`md:`), verified
via `tsc --noEmit` and a full `next build` after the changes, not
just eyeballed:
- Layout padding/spacing scales down on mobile
- The signal instrument (the radial confidence ring) shrinks from
  224px desktop to 160px mobile, using the SVG's `viewBox` scaling
  so nothing needs separate mobile artwork
- Signal-breakdown bars and stat rows wrap instead of overflowing on
  narrow screens
- All interactive controls (start/stop button, mode toggle) meet the
  44px minimum touch-target height on mobile
- Transcript panel height reduces on small screens so it doesn't
  dominate the viewport

Still genuinely untested on a real phone/tablet from this sandbox -
verify on your actual devices, same as the rest of this delivery.

## Known technical choices worth knowing about

- Audio capture uses `ScriptProcessorNode`, which is deprecated in
  favor of `AudioWorklet`. Kept for simplicity/broad compatibility -
  works reliably at speech sample rates, but is a real candidate for
  a future upgrade if you hit browser deprecation warnings.
- Downsampling uses simple linear interpolation, not a high-quality
  resampler - adequate for speech intelligibility (what Deepgram
  needs), not audiophile-grade.
