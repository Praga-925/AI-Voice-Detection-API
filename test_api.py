import base64
import requests

# Test with AI voice (should return AI_GENERATED)
# audio_file = "dataset/ai/ai_00001.wav"

# Or test with Human voice (should return REAL)
audio_file = "dataset/human/1089_134686_000002_000001.wav"

with open(audio_file, "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post(
    "http://127.0.0.1:8000/detect",
    json={"audio_base64": audio_base64}
)

print(f"Testing: {audio_file}")
print(response.json())
