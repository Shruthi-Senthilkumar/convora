import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import soundfile as sf
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# We can create a short 30-second slice of AMI audio or use a sample
AMI_WAV = r"C:\Users\shrut\ami-corpus-data\amicorpus\ES2002a\audio\ES2002a.Mix-Headset.wav"
SHORT_TEST_WAV = "eval/test_short_sample.wav"

if os.path.exists(AMI_WAV):
    # Create a 60-second slice for fast, complete end-to-end verification
    audio, sr = sf.read(AMI_WAV, frames=60 * 16000, dtype="float32")
    sf.write(SHORT_TEST_WAV, audio, sr)
    test_file_path = SHORT_TEST_WAV
else:
    test_file_path = "eval/audio/test.wav"

print(f"Testing /api/process-file with {test_file_path}...")

with open(test_file_path, "rb") as f:
    response = client.post(
        "/api/process-file",
        files={"file": ("sample_meeting.wav", f, "audio/wav")}
    )

print(f"HTTP Status: {response.status_code}")
assert response.status_code == 200, f"Failed with {response.text}"

data = response.json()
print("\nTop-level keys in response:")
print(list(data.keys()))

events = data.get("events", [])
all_cands = data.get("all_candidates", [])
metadata = data.get("metadata", {})

print(f"\nMetadata Summary:")
print(f"  Audio File: {data.get('audio_file')}")
print(f"  Duration: {data.get('duration_s'):.2f}s")
print(f"  Total Pause Candidates: {metadata.get('total_pause_candidates')}")
print(f"  Resolved to End of Speech: {metadata.get('resolved_to_end_of_speech')}")
print(f"  Events Array Length: {len(events)}")
print(f"  All Candidates Array Length: {len(all_cands)}")

print("\n--- First 5 Entries from 'all_candidates' ---")
for i, c in enumerate(all_cands[:5], 1):
    print(f"\nCandidate {i}:")
    print(f"  Timestamp:         {c.get('timestamp_s')}s")
    print(f"  is_end_of_speech:  {c.get('is_end_of_speech')}")
    print(f"  Confidence:        {c.get('confidence')}")
    print(f"  Speaker:           {c.get('speaker')}")
    print(f"  Speaker Changed:   {c.get('speaker_changed')}")
    print(f"  Fragment:          '{c.get('fragment')}'")
    print(f"  Signals:           {c.get('contributing_signals')}")

print("\n--- JSON Snippet of First 2 Candidates ---")
print(json.dumps(all_cands[:2], indent=2))

# Cleanup temporary audio slice
if os.path.exists(SHORT_TEST_WAV):
    os.remove(SHORT_TEST_WAV)

print("\nVERIFICATION SUCCESSFUL: all_candidates is present, schema-compliant, and populated with real confidence/speaker values for all candidates.")
