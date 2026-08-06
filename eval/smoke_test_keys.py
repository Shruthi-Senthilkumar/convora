"""
Phase 0 - Step 3: API key smoke test.
Confirms Deepgram, Groq, and ElevenLabs keys are valid and reachable.
Run from the convora project root with the venv activated:
    python smoke_test_keys.py
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
        client = DeepgramClient(dg_key)
        # Lightweight check: listing projects confirms auth works
        projects = client.manage.v("1").get_projects()
        print(f"[Deepgram] OK - key is valid, {len(projects.projects)} project(s) found")
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
        qwen_available = any("qwen3.6-27b" in m.lower() or "qwen-3.6-27b" in m.lower() for m in model_ids)
        gpt_oss_available = any("gpt-oss-20b" in m.lower() for m in model_ids)
        print(f"       Qwen3.6-27B available: {qwen_available}")
        print(f"       gpt-oss-20b available: {gpt_oss_available}")
        if not qwen_available:
            print("       -> Preview model not visible on this account. "
                  "Check Groq console for waitlist/access status.")
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
        user = client.user.get()
        print(f"[ElevenLabs] OK - key is valid, subscription tier: {user.subscription.tier}")
except Exception as e:
    print(f"[ElevenLabs] FAILED - {e}")

print("\n=== Done ===")
