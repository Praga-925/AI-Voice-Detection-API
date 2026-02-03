import librosa
import numpy as np
import sys
import os

def analyze_audio(file_path):
    """Analyze audio file properties"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n📊 Audio Analysis: {file_path}")
    print("=" * 50)
    
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)  # Keep original sample rate
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Get properties
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"🎵 Sample Rate: {sr} Hz")
        print(f"📈 Samples: {len(y)}")
        print(f"🔊 Max Amplitude: {np.max(np.abs(y)):.4f}")
        print(f"🔇 Min Amplitude: {np.min(np.abs(y)):.6f}")
        print(f"📊 Mean Amplitude: {np.mean(np.abs(y)):.4f}")
        
        # Check if audio is too quiet or too loud
        if np.max(np.abs(y)) < 0.1:
            print("\n⚠️  Warning: Audio seems very quiet!")
        elif np.max(np.abs(y)) > 0.99:
            print("\n⚠️  Warning: Audio may be clipping (too loud)!")
        
        # Recommended sample rate
        if sr != 16000:
            print(f"\n💡 Note: Model was trained on 16kHz audio.")
            print(f"   Your audio is {sr}Hz - will be resampled during prediction.")
        
        # Extract MFCC (same as model uses)
        y_16k, _ = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y_16k, sr=16000, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        print(f"\n🎯 MFCC Features (what model sees):")
        print(f"   {mfcc_mean.round(2)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_audio(sys.argv[1])
    else:
        print("Usage: python analyze_audio.py <audio_file>")
        file_path = input("\nEnter audio file path: ").strip().strip('"')
        if file_path:
            analyze_audio(file_path)
