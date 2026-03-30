"""
Fusion Model Inference Module
Combines audio and clinical predictions for neonatal assessment
"""

import numpy as np
import torch
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple


class FusionModelInference:
    """
    Loads and manages the fusion model for inference.
    Combines CNN-LSTM audio model with ELM clinical model.
    """
    
    def __init__(self, config_path: str = None, model_dir: str = None):
        """
        Initialize the fusion model.
        
        Args:
            config_path: Path to fusion_model_config.json
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir or ".")
        self.config_path = Path(config_path or self.model_dir / "fusion_model_config.json")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load models
        audio_model_path = self.model_dir / self.config['audio_model_path']
        clinical_model_path = self.model_dir / self.config['clinical_model_path']
        
        self.audio_model = torch.jit.load(str(audio_model_path))
        self.audio_model.eval()
        
        self.clinical_model = joblib.load(str(clinical_model_path))
        
        # Fusion weights
        self.w_audio = self.config.get('audio_weight', 0.7)
        self.w_clinical = self.config.get('clinical_weight', 0.3)
    
    def elm_predict_proba(self, X: np.ndarray) -> float:
        """
        Predict probability using ELM model.
        
        Args:
            X: Input features (1, 10) - clinical features
            
        Returns:
            Probability prediction (0-1)
        """
        model_dict = self.clinical_model
        
        # Extract model components
        w = model_dict['w']
        beta = model_dict['beta']
        b = model_dict['b']
        scaler = model_dict['scaler']
        
        # Scale input
        X_scaled = scaler.transform(X)
        
        # Sigmoid activation
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        # Hidden layer activation
        h = sigmoid(np.dot(X_scaled, w) + b)
        
        # Output prediction (probability)
        y_pred_prob = np.dot(h, beta)
        return float(y_pred_prob.flatten()[0])
    
    def predict(
        self,
        audio_input: np.ndarray,
        clinical_input: np.ndarray
    ) -> Dict:
        """
        Make fusion model prediction.
        
        Args:
            audio_input: Audio features (1, 1, 128, 500) - mel spectrogram
            clinical_input: Clinical features (1, 10)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Ensure correct shapes
        if audio_input.shape != (1, 1, 128, 500):
            raise ValueError(f"Expected audio shape (1, 1, 128, 500), got {audio_input.shape}")
        if clinical_input.shape != (1, 10):
            raise ValueError(f"Expected clinical shape (1, 10), got {clinical_input.shape}")
        
        # Audio model prediction
        with torch.no_grad():
            audio_tensor = torch.tensor(audio_input, dtype=torch.float32)
            audio_logits = self.audio_model(audio_tensor)
            audio_probs = torch.softmax(audio_logits, dim=1)
            p_audio = float(audio_probs[0, 1].item())  # Probability of class 1
        
        # Clinical model prediction
        p_clinical = self.elm_predict_proba(clinical_input)
        
        # Weighted fusion
        p_fused = (self.w_audio * p_audio) + (self.w_clinical * p_clinical)
        
        # Final prediction (threshold 0.5)
        prediction = 1 if p_fused >= 0.5 else 0
        
        return {
            "audio_probability": round(p_audio, 4),
            "clinical_probability": round(p_clinical, 4),
            "fused_probability": round(p_fused, 4),
            "prediction": prediction,
            "prediction_label": "Positive" if prediction == 1 else "Negative"
        }
    
    def predict_batch(
        self,
        audio_inputs: np.ndarray,
        clinical_inputs: np.ndarray
    ) -> list:
        """
        Make batch predictions.
        
        Args:
            audio_inputs: Audio features (batch_size, 1, 128, 500)
            clinical_inputs: Clinical features (batch_size, 10)
            
        Returns:
            List of prediction dictionaries
        """
        batch_size = audio_inputs.shape[0]
        results = []
        
        for i in range(batch_size):
            result = self.predict(
                audio_inputs[np.newaxis, i],
                clinical_inputs[np.newaxis, i]
            )
            results.append(result)
        
        return results
