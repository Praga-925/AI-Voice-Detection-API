import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

DATASET_PATH = r"D:\GUVI HCL Hackathon\AI Voice Detection API\dataset"
AI_PATH = os.path.join(DATASET_PATH, "ai")
HUMAN_PATH = os.path.join(DATASET_PATH, "human")

def extract_features(file_path, n_mfcc=20):
    """Extract enhanced audio features with normalization"""
    try:
        # Load audio at 16kHz
        y, sr = librosa.load(file_path, sr=16000)
        
        # Normalize audio (important for consistency!)
        y = librosa.util.normalize(y)
        
        # Extract MFCC (20 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        
        # Extract delta MFCC (how sound changes over time)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
        
        # Extract spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,           # 20 features
            mfcc_std,            # 20 features  
            mfcc_delta_mean,     # 20 features
            [spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate]  # 4 features
        ])
        
        return features  # Total: 64 features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Collect data
X = []
y = []

print("=" * 60)
print("   Enhanced AI Voice Detection Model Training")
print("=" * 60)

print("\n📂 Loading AI-generated audio...")
ai_files = [f for f in os.listdir(AI_PATH) if f.endswith(".wav")]
for file in tqdm(ai_files):
    features = extract_features(os.path.join(AI_PATH, file))
    if features is not None:
        X.append(features)
        y.append(1)  # AI

print("\n📂 Loading Human audio...")
human_files = [f for f in os.listdir(HUMAN_PATH) if f.endswith(".wav")]
for file in tqdm(human_files):
    features = extract_features(os.path.join(HUMAN_PATH, file))
    if features is not None:
        X.append(features)
        y.append(0)  # Human

X = np.array(X)
y = np.array(y)

print(f"\n📊 Dataset Summary:")
print(f"   Total samples: {len(X)}")
print(f"   AI samples: {sum(y == 1)}")
print(f"   Human samples: {sum(y == 0)}")
print(f"   Features per sample: {X.shape[1]}")

# Normalize features using StandardScaler
print("\n⚙️  Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Training set: {len(X_train)} samples")
print(f"📈 Test set: {len(X_test)} samples")

# Train model with better parameters
print("\n🚀 Training Enhanced Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=300,      # More trees
    max_depth=20,          # Limit depth to prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 60}")
print(f"📊 Model Performance")
print(f"{'=' * 60}")
print(f"✅ Accuracy: {acc * 100:.2f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

# Save model and scaler
joblib.dump(model, "model_enhanced.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"\n💾 Model saved as 'model_enhanced.pkl'")
print(f"💾 Scaler saved as 'scaler.pkl'")

# Feature importance
print(f"\n🎯 Top 10 Most Important Features:")
feature_names = (
    [f'mfcc_mean_{i}' for i in range(20)] +
    [f'mfcc_std_{i}' for i in range(20)] +
    [f'mfcc_delta_{i}' for i in range(20)] +
    ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate']
)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

print(f"\n{'=' * 60}")
print("✅ Training Complete!")
print(f"{'=' * 60}")
