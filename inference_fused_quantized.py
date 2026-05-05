"""
Fused Quantized Model Inference
================================

Purpose: Load and run inference with fused INT8/FP16 quantized models
Usage: For production deployment of TotoScreen with optimized models

Features:
- Load INT8 audio model (75% smaller)
- Load FP16 clinical model (50% smaller)
- Compute fused predictions with weighted averaging
- Minimal memory footprint for edge deployment
"""

import torch
import torch.nn.functional as F
import numpy as np
import joblib
import json
import os
from pathlib import Path

class FusedQuantizedModel:
    """Inference wrapper for fused quantized model (INT8 audio + FP16 clinical)"""
    
    def __init__(self, config_path, device='cpu'):
        """
        Initialize fused quantized model
        
        Args:
            config_path: Path to fusion_model_quantized_config.json
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.config = self._load_config(config_path)
        
        # Get directory for model paths
        config_dir = os.path.dirname(os.path.abspath(config_path))
        
        # Extract just the filename from config paths (they may contain full paths)
        audio_filename = os.path.basename(self.config['audio_model']['path'])
        clinical_filename = os.path.basename(self.config['clinical_model']['path'])
        
        # Construct full paths
        audio_path = os.path.join(config_dir, audio_filename)
        clinical_path = os.path.join(config_dir, clinical_filename)
        
        print(f"[LOAD] Loading fused quantized model")
        print(f"   Audio (INT8):   {audio_path}")
        print(f"   Clinical (FP16): {clinical_path}")
        
        self.audio_model = self._load_audio_model(audio_path)
        self.clinical_model = self._load_clinical_model(clinical_path)
        
        self.audio_weight = self.config['audio_weight']
        self.clinical_weight = self.config['clinical_weight']
        
        print(f"[OK] Models loaded successfully")
        print(f"   Audio weight:    {self.audio_weight*100:.0f}%")
        print(f"   Clinical weight: {self.clinical_weight*100:.0f}%")
    
    def _load_config(self, config_path):
        """Load model configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_audio_model(self, model_path):
        """Load INT8 quantized audio model"""
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model
    
    def _load_clinical_model(self, model_path):
        """Load FP16 quantized clinical model"""
        return joblib.load(model_path)
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def predict_audio(self, mel_spectrogram):
        """
        Get audio prediction from INT8 model
        
        Args:
            mel_spectrogram: (1, 128, 500) mel-spectrogram
            
        Returns:
            float: Probability of respiratory distress (0.0-1.0)
        """
        if isinstance(mel_spectrogram, np.ndarray):
            mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
        
        with torch.no_grad():
            logits = self.audio_model(mel_spectrogram.to(self.device))
            probs = F.softmax(logits, dim=1)
            # Return probability of class 1 (distress)
            return float(probs[0, 1].cpu().numpy())
    
    def predict_clinical(self, clinical_features):
        """
        Get clinical prediction from FP16 model
        
        Args:
            clinical_features: (1, 10) clinical features array
            
        Returns:
            float: Probability of respiratory distress (0.0-1.0)
        """
        if isinstance(clinical_features, list):
            clinical_features = np.array([clinical_features], dtype=np.float32)
        elif isinstance(clinical_features, np.ndarray):
            if clinical_features.ndim == 1:
                clinical_features = clinical_features.reshape(1, -1)
            clinical_features = clinical_features.astype(np.float32)
        
        try:
            # Extract ELM components
            w = self.clinical_model['w']
            beta = self.clinical_model['beta']
            b = self.clinical_model['b'].flatten()
            scaler = self.clinical_model['scaler']
            
            # Scale input
            X_scaled = scaler.transform(clinical_features)
            
            # Hidden layer
            h = self._sigmoid(np.dot(X_scaled, w) + b)
            
            # Output prediction
            y_pred = np.dot(h, beta).flatten()[0]
            y_prob = self._sigmoid(y_pred)
            
            return float(y_prob)
        except Exception as e:
            print(f"[ERROR] Clinical prediction failed: {e}")
            return 0.5
    
    def predict_fusion(self, mel_spectrogram, clinical_features):
        """
        Get fused prediction combining audio and clinical models
        
        Args:
            mel_spectrogram: (1, 128, 500) or (128, 500) mel-spectrogram
            clinical_features: (10,) or (1, 10) clinical features array
            
        Returns:
            dict: {
                'audio_prob': float,
                'clinical_prob': float,
                'fusion_prob': float,
                'risk_level': str ('LOW', 'MEDIUM', 'HIGH')
            }
        """
        # Ensure mel_spec has correct shape
        if isinstance(mel_spectrogram, np.ndarray):
            if mel_spectrogram.ndim == 2:
                mel_spectrogram = np.expand_dims(mel_spectrogram, (0, 1))  # Add batch and channel
            elif mel_spectrogram.ndim == 3:
                mel_spectrogram = np.expand_dims(mel_spectrogram, 0)  # Add batch
        
        # Get individual predictions
        audio_prob = self.predict_audio(mel_spectrogram)
        clinical_prob = self.predict_clinical(clinical_features)
        
        # Fuse predictions
        fusion_prob = (self.audio_weight * audio_prob) + (self.clinical_weight * clinical_prob)
        
        # Determine risk level
        if fusion_prob < 0.33:
            risk_level = 'LOW'
        elif fusion_prob < 0.67:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        return {
            'audio_prob': float(audio_prob),
            'clinical_prob': float(clinical_prob),
            'fusion_prob': float(fusion_prob),
            'risk_level': risk_level,
            'risk_threshold': 0.5
        }
    
    def get_model_info(self):
        """Get model configuration info"""
        return {
            'audio_weight': self.audio_weight,
            'clinical_weight': self.clinical_weight,
            'audio_quantization': self.config['audio_model']['quantization'],
            'clinical_quantization': self.config['clinical_model']['quantization'],
            'expected_audio_shape': '(1, 1, 128, 500)',
            'expected_clinical_shape': '(1, 10)',
            'output_range': '(0.0, 1.0)',
        }

# ==================== EXAMPLE USAGE ====================
if __name__ == '__main__':
    # Example: Load fused quantized model
    print("="*70)
    print("FUSED QUANTIZED MODEL INFERENCE EXAMPLE")
    print("="*70 + "\n")
    
    config_path = 'notebooks/MODELS/fusion_model_quantized_config.json'
    
    if os.path.exists(config_path):
        # Initialize model
        model = FusedQuantizedModel(config_path, device='cpu')
        
        print("\n[INFO] Model Configuration:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Generate dummy inputs
        print("\n[DEMO] Running inference with dummy data...")
        
        mel_spec = np.random.randn(1, 1, 128, 500).astype(np.float32)
        clinical_features = np.random.randn(1, 10).astype(np.float32)
        
        # Get predictions
        result = model.predict_fusion(mel_spec, clinical_features)
        
        print(f"\n[RESULTS] Fusion Model Output:")
        print(f"   Audio Probability:     {result['audio_prob']:.4f}")
        print(f"   Clinical Probability:  {result['clinical_prob']:.4f}")
        print(f"   Fusion Probability:    {result['fusion_prob']:.4f}")
        print(f"   Risk Level:            {result['risk_level']}")
        print(f"   Decision Threshold:    0.50")
        
        if result['fusion_prob'] >= 0.5:
            print(f"\n   [ALERT] HIGH RISK - Recommend clinical review")
        else:
            print(f"\n   [OK] LOW RISK - Continue routine monitoring")
        
        print("\n" + "="*70)
    else:
        print(f"[ERROR] Config file not found: {config_path}")
        print("Make sure to run quantize_fusion_models.py first")
