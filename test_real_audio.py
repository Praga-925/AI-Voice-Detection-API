import base64
import requests
import sys
import os

# ================= CONFIG =================
API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_test_123456789"
# =========================================


def test_audio(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return

    print(f"\n🎵 Testing Audio File: {file_path}")
    print("-" * 60)

    try:
        # Read and encode audio
        with open(file_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Send request to API
        response = requests.post(
            API_URL,
            headers={
                "x-api-key": API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            }
        )

        if response.status_code == 200:
            result = response.json()

            classification = result["classification"]
            confidence = result["confidenceScore"]
            explanation = result["explanation"]

            if classification == "AI_GENERATED":
                print(f"🤖 Classification : {classification}")
            else:
                print(f"👤 Classification : {classification}")

            print(f"📊 Confidence     : {confidence}")
            print(f"🧠 Explanation   : {explanation}")

        else:
            print("❌ API returned an error")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        print("👉 Make sure the server is running:")
        print("   uvicorn app_enhanced:app --reload")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_real_audio.py <audio_file>\n")
        print("Example:")
        print("  python test_real_audio.py Recording.wav")
        return

    test_audio(sys.argv[1])


if __name__ == "__main__":
    main()
