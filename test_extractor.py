# test_extractor.py
from pathlib import Path
import sys
sys.path.append('.')

from src.feature_extraction.audio_features import AudioFeatureExtractor

print("="*50)
print("TESTING AUDIO FEATURE EXTRACTOR")
print("="*50)

# Initialize extractor
extractor = AudioFeatureExtractor(sr=4000, n_mfcc=13)

# Check if ICBHI folder exists
icbhi_path = Path("sound_data/icbhi")
if not icbhi_path.exists():
    print(f"ICBHI folder not found at: {icbhi_path.absolute()}")
    sys.exit(1)

# Get audio files
audio_files = list(icbhi_path.glob("*.wav"))
print(f"\nFound {len(audio_files)} audio files in ICBHI folder")

if len(audio_files) == 0:
    print("No .wav files found")
    sys.exit(1)

# Test on first file
print(f"\n Testing on first file: {audio_files[0].name}")
features = extractor.extract_features_simple(audio_files[0])

if features:
    print("Success! Extracted features:")
    for key, value in list(features.items())[:5]:
        print(f"   {key}: {value}")
    
    print("\n Total features extracted:", len(features))
else:
    print("Failed to extract features")

print("\n" + "="*50)
print("Test complete!")