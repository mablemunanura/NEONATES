# Audio Feature Extraction Module for NeoScreen Project
# Extracts features needed for neonatal respiratory distress detection

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    # Extract audio features for respiratory sound analysis
    
    def __init__(self, sr=4000, n_mfcc=13):
        # Initialize feature extractor
        # Parameters:
        # sr : int (Target sample rate)
        # n_mfcc : int (Number of MFCC coefficients to extract)

        self.sr = sr
        self.n_mfcc = n_mfcc
        print(f"AudioFeatureExtractor initialized with sr={sr}, n_mfcc={n_mfcc}")
    
    def extract_features_simple(self, audio_path):
        # Simplified feature extraction - robust for all files
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            features = {}
            
            # 1. MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            for i in range(self.n_mfcc):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
            # 2. Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 3. RMS Energy
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # 4. Duration
            features['duration'] = float(len(y) / sr)
            
            # 5. Simple harmonic ratio
            try:
                harmonic, _ = librosa.effects.hpss(y)
                if np.mean(np.abs(y)) > 0:
                    features['harmonic_ratio'] = float(np.mean(np.abs(harmonic)) / np.mean(np.abs(y)))
                else:
                    features['harmonic_ratio'] = 0.0
            except:
                features['harmonic_ratio'] = 0.0
            
            # 6. Spectral Centroid
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            except:
                features['spectral_centroid_mean'] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_batch(self, file_list, output_csv=None):
        # Extract features from multiple audio files
        
        all_features = []
        success_count = 0
        fail_count = 0
        
        print(f"\n Processing {len(file_list)} files...")
        
        for i, file_path in enumerate(file_list):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(file_list)} (Success: {success_count}, Failed: {fail_count})")
            
            features = self.extract_features_simple(file_path)
                
            if features:
                features['filename'] = Path(file_path).name
                features['file_path'] = str(file_path)
                all_features.append(features)
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n Completed: {success_count} successful, {fail_count} failed")
        
        df = pd.DataFrame(all_features)
        
        if output_csv and len(df) > 0:
            # Create directory if it doesn't exist
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Saved features to {output_csv}")
        
        return df