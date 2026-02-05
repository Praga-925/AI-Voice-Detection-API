# AI Voice Detection API

A production-ready REST API that detects whether a voice sample is **AI-generated** or **human**. Built with Machine Learning for the GUVI HCL Hackathon.

---

## Project Overview

### Problem Statement
With the rise of AI-generated voices (deepfakes, TTS systems), there's a growing need to identify synthetic audio for security, authentication, and content verification purposes.

### Solution
This project uses advanced audio feature extraction combined with a trained Random Forest classifier to detect AI-generated vs human voices with high accuracy. The API is production-ready and deployed on Render.

### Key Capabilities
- Classifies audio as AI-generated or human with confidence scores
- Extracts 64 audio features (MFCC, spectral analysis, zero-crossing rate)
- Uses StandardScaler for feature normalization
- Supports MP3 audio format with Base64 encoding
- Production-deployed with API key authentication
- Fast inference with cold-start optimization

---

## Features

- **AI vs Human Detection**: Classifies voice as AI-generated or human
- **Multi-language Support**: Works with Tamil, English, Hindi, Malayalam, Telugu
- **Base64 Audio Input**: Accepts Base64-encoded MP3 audio
- **REST API**: Simple JSON request/response format
- **API Key Authentication**: Secured endpoints with API key validation
- **Confidence Scoring**: Returns confidence percentage for each prediction
- **Explanations**: Provides reasoning for classifications
- **Production Ready**: Deployed on Render with lazy-loading optimization

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.14.2** | Core programming language |
| **FastAPI** | REST API framework with automatic validation |
| **Scikit-learn** | Random Forest Classifier (200 trees) |
| **Librosa** | Audio feature extraction and processing |
| **NumPy** | Numerical operations for feature engineering |
| **Uvicorn** | ASGI server for production deployment |
| **Joblib** | Model and scaler serialization |

---

## Project Architecture

### Core Files

#### 1. **app_enhanced.py** (Production API)
- Main FastAPI application with enhanced model
- Endpoint: `POST /api/voice-detection`
- Features: 64 extracted audio features
- Authentication: API key header validation
- Lazy loading for Render cold-start optimization
- Confidence threshold: 0.60 for conservative predictions
- Status codes: 200 (success), 400 (malformed request), 401 (invalid API key)

#### 2. **train_model_enhanced.py** (Model Training)
- Extracts 64 audio features from training dataset:
  - 20 MFCC coefficients (mean)
  - 20 MFCC coefficients (standard deviation)
  - 20 Delta MFCC features (how sound changes over time)
  - 4 Spectral features (centroid, rolloff, bandwidth, zero-crossing rate)
- Normalizes audio before feature extraction using librosa.util.normalize()
- Uses StandardScaler for feature normalization across all samples
- Trains Random Forest with 200 trees using all CPU cores
- Train/Test split: 80/20
- Output: `model_enhanced.pkl` (91.89% accuracy) and `scaler.pkl`
- Dataset: 2,173 AI voices + 2,509 human voices = 4,682 total samples

#### 3. **test.py** (Deployed API Testing)
- Tests the production API on Render
- Converts audio to Base64 and sends to deployed endpoint
- URL: `https://ai-voice-detection-api-9rr7.onrender.com/api/voice-detection`
- Requires: Valid API key from Render deployment
- Usage: `python test.py "Audio Testcase  Files\Recording.mp3"`

#### 4. **test_real_audio.py** (Local Testing)
- Tests local FastAPI instance
- Reads audio files and sends to `http://127.0.0.1:8000/api/voice-detection`
- Used for development/debugging
- Displays classification and confidence with explanation
- Usage: `python test_real_audio.py "Audio Testcase  Files\Recording.mp3"`

#### 5. **test_api.py** (Basic Local Testing)
- Minimal test for local API with dataset samples
- Tests both AI and Human voice samples from dataset folder
- Simple endpoint test: `http://127.0.0.1:8000/detect`

#### 6. **analyze_audio.py** (Audio Analysis Tool)
- Analyzes audio file properties (duration, amplitude, sample rate)
- Extracts and displays MFCC features
- Detects audio quality issues (too quiet, clipping)
- Useful for debugging audio files
- Usage: `python analyze_audio.py <audio_file>`

### Data Files

#### Models & Scalers
- **model_enhanced.pkl**: Trained Random Forest classifier (200 trees, 91.89% accuracy)
- **scaler.pkl**: StandardScaler for feature normalization (trained on dataset)

#### Dataset
- **dataset/ai/**: ~2,173 AI-generated voice samples
- **dataset/human/**: ~2,509 human voice samples
- **Audio Testcase Files/**: Test audio samples in multiple languages (Tamil, English, Malayalam, etc.)

### Configuration

#### API Configuration (app_enhanced.py)
```python
API_KEY = os.getenv("API_KEY", "sk_test_123456789")  # From environment variable
CONFIDENCE_THRESHOLD = 0.60  # Conservative threshold for ambiguous cases
SAMPLE_RATE = 16000  # Audio processing rate (Hz)
N_MFCC = 20  # Number of MFCC coefficients
```

#### Feature Engineering Details
- **Audio Loading**: Librosa loads at 16kHz sample rate
- **Normalization**: Audio is normalized to [-1, 1] range
- **MFCC**: Extracts 20 Mel-frequency cepstral coefficients
- **Delta MFCC**: Captures dynamic changes in MFCC over time
- **Spectral Features**:
  - Spectral Centroid: Center of mass of spectrum
  - Spectral Rolloff: Frequency below which 85% of power lies
  - Spectral Bandwidth: Width of spectrum
  - Zero Crossing Rate: How often signal changes sign

---

## API Specification

### Endpoint

```
POST /api/voice-detection
```

### Headers

| Header | Value | Required |
|--------|-------|----------|
| `Content-Type` | `application/json` | Yes |
| `x-api-key` | `<YOUR_API_KEY>` | Yes |

### Request Body

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `language` | string | Yes | Language of audio | `"english"`, `"tamil"` |
| `audioFormat` | string | Yes | Format of audio (must be `mp3`) | `"mp3"` |
| `audioBase64` | string | Yes | Base64-encoded MP3 audio data | `"//NExAAjGZ..."` |

### Response Body (Success - 200)

```json
{
  "status": "success",
  "language": "english",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Response Body (Invalid API Key - 401)

```json
{
  "detail": {
    "status": "error",
    "message": "Invalid API key"
  }
}
```

### Response Body (Malformed Request - 400)

```json
{
  "detail": {
    "status": "error",
    "message": "Malformed request or invalid audio"
  }
}
```

---

## Sample Requests & Responses

### Using cURL

#### AI Voice Detection
```bash
curl -X POST "https://ai-voice-detection-api-9rr7.onrender.com/api/voice-detection" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "english",
    "audioFormat": "mp3",
    "audioBase64": "//NExAAjGZLYAEARQFnqapk5QL..."
  }'
```

#### Response
```json
{
  "status": "success",
  "language": "english",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Using Python

```python
import requests
import base64

# Read audio file
with open("Recording.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# Send request
response = requests.post(
    "https://ai-voice-detection-api-9rr7.onrender.com/api/voice-detection",
    headers={
        "x-api-key": "YOUR_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "language": "english",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)

print(response.json())
```

---

## How It Works (Architecture)

```
┌─────────────────┐
│  MP3 Audio      │
│  (Base64)       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Decode Base64  │
│  & Load Audio   │
└────────┬────────┘
         ▼
┌──────────────────────────┐
│ Feature Extraction (64)  │
│ - 20 MFCC (mean)         │
│ - 20 MFCC (std)          │
│ - 20 Delta MFCC          │
│ - 4 Spectral features    │
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│ Feature Normalization    │
│ (StandardScaler)         │
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│ Random Forest Classifier │
│ (200 trees)              │
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│ Prediction & Confidence  │
│ - Classification         │
│ - Confidence Score       │
│ - Explanation            │
└──────────────────────────┘
```

### Processing Steps

1. **Audio Decoding**: Base64-encoded MP3 is decoded to raw audio waveform
2. **Audio Loading**: Librosa loads audio at 16kHz sample rate and normalizes
3. **Feature Extraction**: 64 audio features are extracted using:
   - MFCC analysis (mean, std, delta)
   - Spectral analysis (centroid, rolloff, bandwidth)
   - Zero-crossing rate
4. **Feature Normalization**: Features are scaled using pre-trained StandardScaler
5. **Classification**: Random Forest model predicts class (0=Human, 1=AI)
6. **Confidence & Explanation**: Model probability and reasoning are returned

---

## Installation & Setup

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

### 4. Train the Model (Optional - Already Trained)

```bash
python train_model_enhanced.py
```

This generates:
- `model_enhanced.pkl` (Random Forest classifier)
- `scaler.pkl` (StandardScaler for features)

### 5. Start the Local Server

```bash
uvicorn app_enhanced:app --reload --host 127.0.0.1 --port 8000
```

The server will start at: `http://127.0.0.1:8000`

### 6. Access the API

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000` | API root |
| `http://127.0.0.1:8000/docs` | Swagger UI (Interactive API docs) |
| `http://127.0.0.1:8000/redoc` | ReDoc documentation |

### 7. Test the API

```bash
# Test with local server
python test_real_audio.py "Audio Testcase  Files\Recording.mp3"

# Test with deployed server (requires valid API key)
python test.py "Audio Testcase  Files\Recording.mp3"

# Analyze audio properties
python analyze_audio.py "Audio Testcase  Files\Recording.mp3"
```

---

## Deployment

### Deployed Endpoint (Production)

```
https://ai-voice-detection-api-9rr7.onrender.com/api/voice-detection
```

### Deployment Configuration

The API is deployed on **Render** with the following settings:
- **Framework**: FastAPI + Uvicorn
- **Environment Variable**: `API_KEY` (set in Render dashboard)
- **Lazy Loading**: Models are loaded on first request to optimize cold-start
- **Python Version**: 3.14.2

### Deployment on Other Platforms

Compatible with:
- **Render** (current)
- **Railway**
- **AWS Lambda** (with serverless framework)
- **Google Cloud Run**
- **Heroku**

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.89% |
| **Algorithm** | Random Forest Classifier |
| **Number of Trees** | 200 |
| **Features** | 64 (MFCC + Spectral) |
| **Training Samples** | 4,682 audio files |
| **AI Samples** | 2,173 |
| **Human Samples** | 2,509 |
| **Confidence Threshold** | 0.60 |

### Classification Logic

- **If confidence >= 0.60**: Returns prediction (AI_GENERATED or HUMAN)
- **If confidence < 0.60**: Conservative prediction (HUMAN) - marked as ambiguous

---

## Important Notes

| Note | Description |
|------|-------------|
| **Language-Agnostic** | The model detects AI voices regardless of language |
| **Language Field** | Provided by client for reference, not used in detection algorithm |
| **Audio Integrity** | Audio is not modified during inference |
| **Supported Format** | Currently supports MP3 audio only |
| **Sample Rate** | Audio is resampled to 16kHz internally |
| **Confidence Score** | Ranges from 0.0 to 1.0 (not percentage) |
| **API Key Required** | Must include valid API key in request headers |

---

## Project Structure

```
AI Voice Detection API/
├── app_enhanced.py              # Production FastAPI application
├── train_model_enhanced.py      # Model training script
├── model_enhanced.pkl           # Trained Random Forest model
├── scaler.pkl                   # StandardScaler for features
├── test.py                      # Test deployed API
├── test_real_audio.py           # Test local API
├── test_api.py                  # Basic local API test
├── analyze_audio.py             # Audio analysis tool
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├── .git/                        # Git repository
├── .venv/                       # Python virtual environment
└── dataset/
    ├── ai/                      # AI voice samples (~2,173)
    └── human/                   # Human voice samples (~2,509)
└── Audio Testcase Files/        # Test audio samples
    ├── Recording.mp3
    ├── Recording.wav
    ├── tamil_audio.mp3
    ├── malayalam_audio.mp3
    └── ... (other test files)
```

---

## Troubleshooting

### API Key Invalid Error
- **Problem**: 401 "Invalid API key"
- **Solution**: Ensure you're using the correct API key from Render deployment settings

### Audio Format Not Supported
- **Problem**: 400 "Only MP3 format supported"
- **Solution**: Convert your audio to MP3 format before uploading

### Connection Refused
- **Problem**: Cannot connect to local API
- **Solution**: Ensure server is running with `uvicorn app_enhanced:app --reload`

### Low Confidence Predictions
- **Problem**: Getting "ambiguous" classifications with confidence < 0.60
- **Solution**: This is by design - use higher quality audio samples

---

## API Usage Examples

### Example 1: Detect AI Voice in English

```python
import requests
import base64

with open("ai_sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "https://ai-voice-detection-api-9rr7.onrender.com/api/voice-detection",
    headers={"x-api-key": "YOUR_KEY"},
    json={
        "language": "english",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)

print(response.json())
# Output: AI_GENERATED with 0.92 confidence
```

### Example 2: Detect Human Voice in Tamil

```python
import requests
import base64

with open("human_tamil.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "https://ai-voice-detection-api-9rr7.onrender.com/api/voice-detection",
    headers={"x-api-key": "YOUR_KEY"},
    json={
        "language": "tamil",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)

print(response.json())
# Output: HUMAN with 0.85 confidence
```

---

## Recent Changes & Improvements

1. **Enhanced Model**: Increased from 13 to 64 audio features for better accuracy
2. **Feature Normalization**: Added StandardScaler for consistent feature scaling
3. **Audio Normalization**: Normalizes audio before feature extraction
4. **Conservative Threshold**: Confidence threshold of 0.60 for ambiguous cases
5. **Lazy Loading**: Models load on first request for Render cold-start optimization
6. **API Key Authentication**: Secured endpoints with API key validation
7. **Detailed Explanations**: Returns reasoning for classifications
8. **Multi-language Support**: Works across different languages

---

## Team & Credits

Built for **GUVI HCL Hackathon**

---

## License

This project is for educational and hackathon purposes.

---

## Contact & Support

For issues, questions, or deployment help, contact the development team.

---

**Last Updated**: February 5, 2026  
**API Version**: 1.0  
**Status**: Production Ready
