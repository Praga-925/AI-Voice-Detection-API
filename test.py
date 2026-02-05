import base64
import requests
import sys
import os

API_URL = "https://ai-voice-detection-api-9rr7.onrender.com/api/voice-detection"
API_KEY = "sk_test_123456789"  # Replace with your real Render API key


def test_audio(mp3_file):
    if not os.path.exists(mp3_file):
        print(f"File not found: {mp3_file}")
        return

    # Convert MP3 to Base64
    with open(mp3_file, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    print("Sending request to deployed API...\n")

    response = requests.post(API_URL, json=payload, headers=headers)

    print("HTTP Status Code:", response.status_code)

    try:
        print("\n Response JSON:")
        print(response.json())
    except Exception:
        print("\n Raw Response:")
        print(response.text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python test_render_api.py <audio.mp3>")
        sys.exit(1)

    test_audio(sys.argv[1])
