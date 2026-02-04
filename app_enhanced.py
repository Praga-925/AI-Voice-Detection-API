import base64
import io
import os
import numpy as np
import librosa
import joblib
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# ================= CONFIG =================
API_KEY = os.getenv("API_KEY", "sk_test_123456789")
CONFIDENCE_THRESHOLD = 0.60
# =========================================

# Load model
if os.path.exists("model_enhanced.pkl") and os.path.exists("scaler.pkl"):
    print("Loading enhanced model...")
    model = joblib.load("model_enhanced.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    raise RuntimeError("Enhanced model or scaler not found!")

app = FastAPI(title="AI Voice Detection API")

# ---------- Request Schema ----------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ---------- Feature Extraction ----------
def extract_features(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)

    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        mfcc_delta_mean,
        [spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate]
    ])

    return scaler.transform(features.reshape(1, -1))

# ---------- API Endpoint ----------
@app.post("/api/voice-detection")
def voice_detection(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API Key check
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )

    # 🔍 Input validation
    if data.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Only MP3 format supported"}
        )

    try:
        audio_bytes = base64.b64decode(data.audioBase64)
        features = extract_features(audio_bytes)

        probs = model.predict_proba(features)[0]
        confidence = float(max(probs))
        prediction = int(np.argmax(probs))

        # 🧠 Conservative decision
        if confidence < CONFIDENCE_THRESHOLD:
            classification = "HUMAN"
            explanation = "Speech characteristics are ambiguous with mixed human and synthetic patterns"
        else:
            classification = "AI_GENERATED" if prediction == 1 else "HUMAN"
            explanation = (
                "Unnatural pitch consistency and robotic speech patterns detected"
                if classification == "AI_GENERATED"
                else "Natural pitch variation and human-like speech patterns detected"
            )

        return {
            "status": "success",
            "language": data.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Malformed request or invalid audio"}
        )
