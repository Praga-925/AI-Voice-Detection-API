import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = r"D:\GUVI HCL Hackathon\AI Voice Detection API\dataset"
AI_PATH = os.path.join(DATASET_PATH, "ai")
HUMAN_PATH = os.path.join(DATASET_PATH, "human")

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

print("Loading AI-generated audio...")
for file in tqdm(os.listdir(AI_PATH)):
    if file.endswith(".wav"):
        mfcc = extract_mfcc(os.path.join(AI_PATH, file))
        X.append(mfcc)
        y.append(1)  # AI

print("Loading Human audio...")
for file in tqdm(os.listdir(HUMAN_PATH)):
    if file.endswith(".wav"):
        mfcc = extract_mfcc(os.path.join(HUMAN_PATH, file))
        X.append(mfcc)
        y.append(0)  # Human

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {acc * 100:.2f}%")

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
