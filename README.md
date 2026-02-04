# 🎤 AI Voice Detection API

A REST API that detects whether a voice sample is **AI-generated** or **human**. Built for the GUVI HCL Hackathon.

---

## 📌 Project Overview

### Problem Statement
With the rise of AI-generated voices (deepfakes, TTS systems), there's a growing need to identify synthetic audio for security, authentication, and content verification purposes.

### Purpose
This API provides a simple, fast, and reliable way to classify voice samples as either AI-generated or human.

### Solution Approach
- Extract audio features (MFCC, spectral features) from voice samples
- Use a trained machine learning model for classification
- Return confidence-based predictions via a REST API

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 AI vs Human Detection | Classifies voice as AI-generated or human |
| 🌐 Multi-language Support | Supports Tamil, English, Hindi, Malayalam, Telugu |
| 🔊 Base64 Audio Input | Accepts Base64-encoded MP3 audio |
| 📡 REST API | Simple JSON request/response format |
| 🔐 API Key Authentication | Secured endpoints with API key |
| ⚡ Fast Inference | Lightweight model for quick predictions |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **FastAPI** | REST API framework |
| **Scikit-learn** | Machine learning model |
| **Librosa** | Audio feature extraction |
| **NumPy** | Numerical operations |
| **Uvicorn** | ASGI server |

---

## 📡 API Specification

### Endpoint

```
POST /api/voice-detection
```

### Headers

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `x-api-key` | `<API_KEY>` |

### Request Body

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | Language of audio (e.g., "english", "tamil") |
| `audioFormat` | string | Format of audio (`mp3`) |
| `audioBase64` | string | Base64-encoded audio data |

### Response Body

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Request status (`success` or `error`) |
| `language` | string | Detected/provided language |
| `classification` | string | `AI_GENERATED` or `HUMAN` |
| `confidenceScore` | number | Confidence percentage (0-100) |
| `explanation` | string | Brief explanation of the result |

---

## 📝 Sample Request & Response

### Request

```json
{
  "language": "english",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2..."
}
```

### Response

```json
{
  "status": "success",
  "language": "english",
  "classification": "AI_GENERATED",
  "confidenceScore": 87.5,
  "explanation": "Synthetic spectral patterns and uniform pitch detected."
}
```

---

## ⚙️ How It Works

```
┌─────────────────┐
│  Base64 Audio   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Decode MP3     │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Extract Features│  ← MFCC, Spectral Centroid, etc.
└────────┬────────┘
         ▼
┌─────────────────┐
│   ML Model      │  ← Random Forest Classifier
└────────┬────────┘
         ▼
┌─────────────────┐
│  Classification │  → AI_GENERATED or HUMAN
└─────────────────┘
```

1. **Audio Decoding**: Base64 MP3 is decoded to raw audio
2. **Feature Extraction**: MFCC and spectral features are extracted using Librosa
3. **Classification**: Trained ML model predicts the class
4. **Response**: Confidence score and explanation are returned

---

## 🚀 Running Locally

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "AI Voice Detection API"
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (if not already trained)

```bash
python train_model.py
```

### 5. Start the Server

```bash
uvicorn app:app --reload
```

### 6. Access the API

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000` | API root |
| `http://127.0.0.1:8000/docs` | Swagger UI (Interactive API docs) |
| `http://127.0.0.1:8000/redoc` | ReDoc documentation |

---

## ☁️ Deployment

- The API can be deployed as a **public REST endpoint**
- Compatible with cloud platforms:
  - **Render**
  - **Railway**
  - **AWS Lambda**
  - **Google Cloud Run**
  - **Heroku**

### Deployed Endpoint (for evaluation)

```
https://<your-deployed-url>/api/voice-detection
```

---

## 📋 Notes

| Note | Description |
|------|-------------|
| **Language-Agnostic** | The model detects AI voices regardless of language |
| **Language Field** | Provided by client for reference, not used in detection |
| **Audio Integrity** | Audio is not modified during inference |
| **Supported Format** | Currently supports MP3 audio |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~91% |
| **Algorithm** | Random Forest Classifier |
| **Features** | 64 (MFCC + Spectral) |
| **Training Samples** | 4,682 audio files |

---

## 📁 Project Structure

```
AI Voice Detection API/
├── app.py                 # FastAPI application
├── train_model.py         # Model training script
├── model.pkl              # Trained model
├── test_api.py            # API testing script
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── dataset/
    ├── ai/                # AI voice samples
    └── human/             # Human voice samples
```

---

## 👨‍💻 Author

Built for **GUVI HCL Hackathon**

---

## 📄 License

This project is for educational and hackathon purposes.
