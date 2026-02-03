import base64
import io
import numpy as np
import librosa
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load trained model
model = joblib.load("model.pkl")

app = FastAPI(title="AI Voice Detection API")

class AudioRequest(BaseModel):
    audio_base64: str

def extract_mfcc_from_bytes(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Must match training (13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

@app.post("/detect")
def detect_voice(data: AudioRequest):
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        features = extract_mfcc_from_bytes(audio_bytes)

        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])

        return {
            "classification": "AI_GENERATED" if prediction == 1 else "REAL",
            "confidence": f"{confidence * 100:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
