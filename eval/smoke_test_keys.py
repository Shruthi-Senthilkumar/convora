"""
Phase 0 - Step 3: API key smoke test.
Confirms Deepgram, Groq, and ElevenLabs keys are valid and reachable.

NOTE: Deepgram check updated for deepgram-sdk v7.x API (api_key must be
keyword arg; client.listen.v1.media.transcribe_url replaces the old
listen.rest.v("1") pattern).

Run from the convora project root with the venv activated:
    python eval/smoke_test_keys.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=== API Key Smoke Test ===\n")

# --- Deepgram ---
try:
    from deepgram import DeepgramClient
    dg_key = os.getenv("DEEPGRAM_API_KEY")
    if not dg_key:
        print("[Deepgram] SKIPPED - no key found in .env")
    else:
        client = DeepgramClient(api_key=dg_key)
        # Lightweight real validation: transcribe Deepgram's own small
        # public sample file via URL (cheap, confirms auth + connectivity)
        response = client.listen.v1.media.transcribe_url(
            url="https://dpgr.am/spacewalk.wav",
            model="nova-3",
        )
        transcript = response.results.channels[0].alternatives[0].transcript
        print(f"[Deepgram] OK - key is valid, sample transcription returned "
              f"{len(transcript)} chars")
except Exception as e:
    print(f"[Deepgram] FAILED - {e}")

# --- Groq ---
try:
    from groq import Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[Groq] SKIPPED - no key found in .env")
    else:
        client = Groq(api_key=groq_key)
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"[Groq] OK - key is valid, {len(model_ids)} model(s) available")
        qwen_available = any("qwen3.6-27b" in m.lower() for m in model_ids)
        gpt_oss_available = any("gpt-oss-20b" in m.lower() for m in model_ids)
        print(f"       Qwen3.6-27B available: {qwen_available}")
        print(f"       gpt-oss-20b available: {gpt_oss_available}")
except Exception as e:
    print(f"[Groq] FAILED - {e}")

# --- ElevenLabs ---
try:
    from elevenlabs.client import ElevenLabs
    el_key = os.getenv("ELEVENLABS_API_KEY")
    if not el_key:
        print("[ElevenLabs] SKIPPED - no key found in .env")
    else:
        client = ElevenLabs(api_key=el_key)
        default_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel - default premade voice
        audio_chunks = list(client.text_to_speech.convert(
            voice_id=default_voice_id,
            text="Testing.",
            model_id="eleven_turbo_v2_5",
        ))
        total_bytes = sum(len(chunk) for chunk in audio_chunks)
        print(f"[ElevenLabs] OK - key is valid, TTS call succeeded, {total_bytes} bytes returned")
except Exception as e:
    print(f"[ElevenLabs] FAILED - {e}")

print("\n=== Done ===")
