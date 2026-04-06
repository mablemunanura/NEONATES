"""
Gradio app for Fusion Model - Neonatal Assessment
Pure Gradio implementation for working audio input on HuggingFace Spaces
"""

import gradio as gr
import numpy as np
import torch
import joblib
import json
from pathlib import Path
from datetime import datetime
import librosa
import io
import traceback
from sklearn.preprocessing import normalize

# ==================== Load Models ====================
def load_models():
    """Load all models"""
    try:
        model_dir = Path(".")
        
        # Audio model
        audio_model = torch.jit.load(str(model_dir / "cnn_lstm_audio_model_scripted.pt"))
        audio_model.eval()
        
        # Clinical model (may be dict or sklearn model)
        elm_data = joblib.load(str(model_dir / "elm_model.pkl"))
        if isinstance(elm_data, dict):
            elm_model = elm_data.get('model', elm_data)  # Extract model if nested in dict
        else:
            elm_model = elm_data
        
        # Config
        with open(model_dir / "fusion_model_config.json") as f:
            config = json.load(f)
        
        return audio_model, elm_model, config
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

try:
    audio_model, elm_model, config = load_models()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Failed to load models: {e}")
    raise

# ==================== Audio Processing ====================
def process_audio_to_mel(audio_input):
    """
    Convert audio to mel-spectrogram
    Gradio audio input: (sample_rate, numpy_array) tuple
    """
    try:
        if audio_input is None:
            return None, "❌ No audio provided"
        
        # Gradio returns (sample_rate, audio_array)
        sr, audio_array = audio_input
        
        # Convert to float32 (Gradio may return int16)
        if audio_array.dtype != np.float32:
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            else:
                audio_array = audio_array.astype(np.float32)
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Resample if needed
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        
        # Check duration
        duration = len(audio_array) / 16000
        if duration < 0.5:
            return None, f"❌ Audio too short ({duration:.2f}s). Need at least 0.5s"
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=16000, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        if mel_db.std() > 0.01:
            mel_db = (mel_db - mel_db.mean()) / mel_db.std()
        else:
            mel_db = mel_db / 80.0
        
        # Pad/crop to 500 frames
        if mel_db.shape[1] < 500:
            mel_db = np.pad(mel_db, ((0, 0), (0, 500 - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :500]
        
        # Final shape: (1, 1, 128, 500)
        audio_tensor = mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return audio_tensor, f"✓ Audio processed ({duration:.1f}s)"
        
    except Exception as e:
        return None, f"❌ Error: {type(e).__name__}: {str(e)}"


def load_audio(file_path):
    """Load and process uploaded audio file"""
    if file_path is None:
        return None
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        return sr, y.astype(np.float32)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


def normalize_clinical_features(ga, bw, hc, dm, apgar1, apgar5, temp, hr, rr, spo2):
    """Normalize clinical features to 0-1 range based on neonatal reference values"""
    # Define min/max for each feature based on neonatal ranges
    feature_ranges = {
        'ga': (28, 42),           # Gestational age: 28-42 weeks
        'bw': (1500, 4500),       # Birth weight: 1.5-4.5 kg
        'hc': (28, 38),           # Head circumference: 28-38 cm
        'dm': (0, 1),             # Delivery mode: 0=vaginal, 1=cesarean
        'apgar1': (0, 10),        # Apgar 1min: 0-10
        'apgar5': (0, 10),        # Apgar 5min: 0-10
        'temp': (35.5, 38.5),     # Temperature: 35.5-38.5°C
        'hr': (100, 180),         # Heart rate: 100-180 bpm
        'rr': (30, 70),           # Respiratory rate: 30-70 bpm
        'spo2': (90, 100)         # SpO2: 90-100%
    }
    
    # Normalize each feature
    normalized = []
    values = [ga, bw, hc, dm, apgar1, apgar5, temp, hr, rr, spo2]
    keys = ['ga', 'bw', 'hc', 'dm', 'apgar1', 'apgar5', 'temp', 'hr', 'rr', 'spo2']
    
    for val, key in zip(values, keys):
        min_val, max_val = feature_ranges[key]
        # Clip to range and normalize to [0, 1]
        normalized_val = (val - min_val) / (max_val - min_val)
        normalized_val = max(0, min(1, normalized_val))  # Clip to [0, 1]
        normalized.append(normalized_val)
    
    return np.array([normalized], dtype=np.float32)


def get_risk_factors(ga, bw, hc, dm, apgar1, apgar5, temp, hr, rr, spo2):
    """Identify which clinical features are outside normal ranges (risk factors)"""
    normal_ranges = {
        'Gestational Age': (37, 40, 'weeks', 'older is better'),
        'Birth Weight': (2500, 4000, 'g', 'heavier is better'),
        'Head Circumference': (32, 36, 'cm', 'normal range'),
        'Apgar 1min': (7, 10, 'score', 'higher is better'),
        'Apgar 5min': (8, 10, 'score', 'higher is better'),
        'Temperature': (36.5, 37.5, '°C', 'normal range'),
        'Heart Rate': (120, 160, 'bpm', 'normal range'),
        'Respiratory Rate': (40, 60, 'bpm', 'normal range'),
        'SpO2': (95, 100, '%', 'higher is better'),
    }
    
    values = [ga, bw, hc, apgar1, apgar5, temp, hr, rr, spo2]
    labels = list(normal_ranges.keys())
    
    risk_factors = []
    protective_factors = []
    
    for label, val in zip(labels, values):
        if label == 'Delivery Mode':
            continue
            
        min_norm, max_norm, unit, direction = normal_ranges[label]
        
        if val < min_norm:
            severity = 'HIGH' if val < min_norm * 0.85 else 'MEDIUM'
            risk_factors.append({
                'feature': label,
                'value': val,
                'unit': unit,
                'status': f'LOW ({severity})',
                'ideal': f'{min_norm}-{max_norm} {unit}'
            })
        elif val > max_norm:
            severity = 'HIGH' if val > max_norm * 1.15 else 'MEDIUM'
            risk_factors.append({
                'feature': label,
                'value': val,
                'unit': unit,
                'status': f'HIGH ({severity})',
                'ideal': f'{min_norm}-{max_norm} {unit}'
            })
        else:
            protective_factors.append({
                'feature': label,
                'value': val,
                'unit': unit,
                'status': '✓ Normal',
                'ideal': f'{min_norm}-{max_norm} {unit}'
            })
    
    return risk_factors, protective_factors


def generate_explanation(audio_pred, clinical_pred, fusion_pred, audio_status, risk_factors, protective_factors):
    """Generate AI-friendly explanation of prediction"""
    explanations = []
    
    # Audio explanation
    if audio_status == "OK":
        audio_risk = "abnormalities" if audio_pred > 0.6 else "normal features"
        explanations.append(f"🎵 <b>Audio:</b> Model detected {audio_risk} in cry pattern (confidence: {abs(audio_pred - 0.5) * 2 * 100:.0f}%)")
    else:
        explanations.append(f"🎵 <b>Audio:</b> {audio_status}")
    
    # Clinical explanation
    if risk_factors:
        risk_text = ", ".join([f"{rf['feature']} ({rf['value']} {rf['unit']})" for rf in risk_factors[:3]])
        explanations.append(f"💊 <b>Clinical Risk Factors:</b> {risk_text}")
    
    if protective_factors:
        prot_text = ", ".join([f"{pf['feature']}" for pf in protective_factors[:2]])
        explanations.append(f"✓ <b>Protective Factors:</b> {prot_text}")
    
    # Fusion explanation
    if fusion_pred >= 0.6:
        explanations.append(f"⚠️ <b>Combined Assessment:</b> Multiple concerning factors detected")
    elif fusion_pred >= 0.5:
        explanations.append(f"⚠️ <b>Combined Assessment:</b> Moderate concerns warrant monitoring")
    else:
        explanations.append(f"✓ <b>Combined Assessment:</b> No major concerns, normal parameters")
    
    return explanations


# ==================== Prediction Function ====================
def predict(audio_input, ga, bw, hc, dm, apgar1, apgar5, temp, hr, rr, spo2):
    """
    Make prediction using fusion model
    audio_input: (sample_rate, audio_array) from Gradio audio component
    """
    try:
        # Process audio
        audio_tensor, audio_msg = process_audio_to_mel(audio_input)
        
        if audio_tensor is None:
            # Fallback to random audio if processing fails
            audio_tensor = np.random.randn(1, 1, 128, 500).astype(np.float32) * 0.1
            audio_pred = 0.5
            audio_info = f"⚠️ Using fallback audio (reason: {audio_msg})"
            audio_status = "FALLBACK"
        else:
            # Get audio prediction
            with torch.no_grad():
                audio_tensor_torch = torch.from_numpy(audio_tensor)
                audio_out = audio_model(audio_tensor_torch)
                # Handle multi-element tensor output
                audio_out = audio_out.flatten()[0]  # Flatten and take first element
                audio_pred = torch.sigmoid(audio_out).item()
            audio_info = audio_msg
            audio_status = "OK"
        
        print(f"[AUDIO] Status: {audio_status}, Pred: {audio_pred:.4f}")
        
        # Prepare and normalize clinical features
        clinical_features = normalize_clinical_features(ga, bw, hc, dm, apgar1, apgar5, temp, hr, rr, spo2)
        
        print(f"[CLINICAL] Normalized features: {clinical_features[0]}")
        print(f"[CLINICAL] Feature values: GA={ga}, BW={bw}, HC={hc}, DM={dm}, APGAR1={apgar1}, APGAR5={apgar5}, TEMP={temp}, HR={hr}, RR={rr}, SPO2={spo2}")
        
        # Get clinical prediction
        clinical_status = "✓ Using Clinical Model"
        clinical_pred = 0.5  # Default value
        try:
            # Handle both sklearn models and dicts
            if hasattr(elm_model, 'predict'):
                clinical_pred = elm_model.predict(clinical_features)[0]
            elif isinstance(elm_model, dict) and 'predict' in elm_model:
                clinical_pred = elm_model['predict'](clinical_features)[0]
            
            clinical_pred = float(clinical_pred)  # Ensure it's a Python float
            # Ensure it's in [0, 1] range
            clinical_pred = max(0, min(1, clinical_pred))
        except Exception as e:
            clinical_pred = 0.5
            clinical_status = f"❌ Error: {type(e).__name__}"
            print(f"[CLINICAL] Exception: {e}")
            print(f"[CLINICAL] elm_model type: {type(elm_model)}")
        
        print(f"[CLINICAL] Status: {clinical_status}, Pred: {clinical_pred:.4f}")
        
        # Fusion: 70% audio + 30% clinical
        audio_weight = config.get("audio_weight", 0.7)
        clinical_weight = config.get("clinical_weight", 0.3)
        
        fusion_pred = (audio_weight * audio_pred) + (clinical_weight * clinical_pred)
        
        print(f"[FUSION] Audio ({audio_weight:.1%} * {audio_pred:.4f}) + Clinical ({clinical_weight:.1%} * {clinical_pred:.4f}) = {fusion_pred:.4f}")
        
        # Generate explainability
        risk_factors, protective_factors = get_risk_factors(ga, bw, hc, dm, apgar1, apgar5, temp, hr, rr, spo2)
        explanations = generate_explanation(audio_pred, clinical_pred, fusion_pred, audio_status, risk_factors, protective_factors)
        
        # Classify
        if fusion_pred >= 0.5:
            diagnosis = "🔴 HIGH RISK"
            risk_class = "results-high-risk"
        else:
            diagnosis = "🟢 LOW RISK"
            risk_class = "results-low-risk"
        
        confidence = abs(fusion_pred - 0.5) * 2 * 100
        
        # Build explanation section
        explanation_html = "<br>".join(explanations)
        
        # Build risk factors list
        risk_factors_html = ""
        if risk_factors:
            risk_factors_html = "<h4 style='color: #d32f2f; margin: 10px 0 5px 0;'>⚠️ Concerning Factors:</h4><ul style='margin: 5px 0; padding-left: 20px;'>"
            for rf in risk_factors:
                risk_factors_html += f"<li><b>{rf['feature']}:</b> {rf['value']} {rf['unit']} ({rf['status']}) - expected: {rf['ideal']}</li>"
            risk_factors_html += "</ul>"
        
        protective_factors_html = ""
        if protective_factors:
            protective_factors_html = "<h4 style='color: #388e3c; margin: 10px 0 5px 0;'>✓ Normal Factors:</h4><ul style='margin: 5px 0; padding-left: 20px;'>"
            for pf in protective_factors[:5]:  # Show top 5
                protective_factors_html += f"<li><b>{pf['feature']}:</b> {pf['value']} {pf['unit']}</li>"
            protective_factors_html += "</ul>"
        
        # Format results with new styling
        results = f"""
        <div class="{risk_class}">
            <h2>{diagnosis}</h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 15px 0;">
                <div class="metric-box">
                    <div class="metric-label">🎵 Audio Prediction</div>
                    <div class="metric-value">{audio_pred:.1%}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">💊 Clinical Prediction</div>
                    <div class="metric-value">{clinical_pred:.1%}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">📊 Weights</div>
                    <div class="metric-value">Audio {audio_weight:.0%} + Clinical {clinical_weight:.0%}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">🔀 Fusion Score</div>
                    <div class="metric-value">{fusion_pred:.1%}</div>
                </div>
            </div>
            
            <hr style="border: none; border-top: 1px solid rgba(0,0,0,0.1); margin: 15px 0;">
            
            <h3 style="margin-top: 15px; margin-bottom: 10px;">🤖 AI Explanation:</h3>
            <div style="background: rgba(0,0,0,0.02); padding: 12px; border-left: 4px solid #1976d2; border-radius: 4px; font-size: 0.95em; line-height: 1.6;">
                {explanation_html}
            </div>
            
            {risk_factors_html}
            {protective_factors_html}
            
            <hr style="border: none; border-top: 1px solid rgba(0,0,0,0.1); margin: 15px 0;">
            
            <h3 style="margin-top: 15px; margin-bottom: 10px;">🔍 Technical Details:</h3>
            <div style="background: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 0.9em; font-family: monospace;">
                <div style="margin-top: 5px;"><b>Fusion Calculation:</b></div>
                <div style="margin-left: 10px;">= ({audio_weight:.0%} × {audio_pred:.4f}) + ({clinical_weight:.0%} × {clinical_pred:.4f})</div>
                <div style="margin-left: 10px;">= {audio_weight * audio_pred:.4f} + {clinical_weight * clinical_pred:.4f}</div>
                <div style="margin-left: 10px;"><b>= {fusion_pred:.4f}</b></div>
            </div>
            
            <hr style="border: none; border-top: 1px solid rgba(0,0,0,0.1); margin: 15px 0;">
            
            <h3 style="margin-top: 15px; margin-bottom: 10px;">📋 Assessment Time:</h3>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li><b>Time:</b> {datetime.now().strftime('%H:%M:%S')}</li>
            </ul>
        </div>
        """
        
        return results
        
    except Exception as e:
        error_msg = f"""
        <div style="background: #ffcccc; padding: 20px; border-radius: 10px;">
            <h3 style="color: #cc0000;">❌ Prediction Error</h3>
            <p><b>{type(e).__name__}:</b> {str(e)}</p>
            <pre>{traceback.format_exc()}</pre>
        </div>
        """
        return error_msg

# ==================== Gradio Interface ====================
with gr.Blocks(
    title="NeoScreen - Neonatal Assessment",
    theme=gr.themes.Soft(),
    css="""
    .header-container { 
        background: linear-gradient(135deg, #1e40af 0%, #0ea5e9 100%);
        padding: 40px 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-container h1 { margin: 0; font-size: 2.5em; }
    .header-container p { margin: 10px 0 0 0; opacity: 0.95; }
    
    .input-section {
        background: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .section-title {
        color: #1e40af;
        font-weight: 700;
        font-size: 1.3em;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 3px solid #0ea5e9;
    }
    
    .results-high-risk {
        background: #fee2e2;
        border-left: 5px solid #dc2626;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .results-high-risk h2 {
        color: #991b1b;
        margin-top: 0;
        font-size: 1.8em;
    }
    
    .results-low-risk {
        background: #dcfce7;
        border-left: 5px solid #16a34a;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .results-low-risk h2 {
        color: #14532d;
        margin-top: 0;
        font-size: 1.8em;
    }
    
    .metric-box {
        background: white;
        padding: 12px 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 8px 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #0f172a;
        font-size: 1.4em;
        font-weight: 600;
        margin-top: 4px;
    }
    
    .action-button {
        background: linear-gradient(135deg, #1e40af 0%, #0ea5e9 100%) !important;
        color: white !important;
        border: none !important;
        height: 50px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        border-radius: 8px !important;
    }
    
    .action-button:hover {
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.3) !important;
    }
    """
) as demo:
    
    gr.HTML("""
    <div class="header-container">
        <h1>🏥 NeoScreen</h1>
        <p><strong>AI-Powered Neonatal Clinical Assessment System</strong></p>
        <p style="font-size: 0.95em; margin-top: 15px;">
            Fusion model combining audio and clinical data for accurate risk assessment
        </p>
    </div>
    """)
    
    # Main layout - two columns  
    with gr.Row():
        # LEFT COLUMN: Audio Input
        with gr.Column(scale=1):
            gr.Markdown("### 🎵 Audio Input (Gradio)")
            gr.Markdown("*Mel-spectrogram (128×500)*")
            gr.Markdown("**Choose method:**")
            
            # Main audio input (receives from all methods)
            audio_input = gr.Audio(label="Audio Data", type="numpy", interactive=False)
            
            with gr.Row():
                tab_gen = gr.Button("🎲 Generate", size="sm")
                tab_up = gr.Button("📁 Upload", size="sm")
                tab_rec = gr.Button("🎤 Record", size="sm")
            
            # Generate tab
            with gr.Group(visible=True) as gen_group:
                gr.Info("📊 Generating sample mel-spectrogram...")
                gen_btn = gr.Button("✓ Generate Sample", size="sm")
            
            # Upload tab
            with gr.Group(visible=False) as up_group:
                gr.Markdown("Upload audio file")
                audio_file = gr.File(label="Choose file", file_types=["audio"])
                audio_file.change(load_audio, inputs=audio_file, outputs=audio_input)
            
            # Record tab
            with gr.Group(visible=False) as rec_group:
                gr.Markdown("Record from microphone")
                rec_audio = gr.Audio(label="Microphone", type="numpy", sources=["microphone"])
                rec_audio.change(lambda x: x, inputs=rec_audio, outputs=audio_input)
        
        # RIGHT COLUMN: Clinical Features
        with gr.Column(scale=1):
            gr.Markdown("### 💊 Clinical Features")
            gr.Markdown("Neonatal vital signs & measurements")
            
            ga = gr.Number(label="Gestational Age (weeks)", value=38, precision=1)
            bw = gr.Number(label="Birth Weight (g)", value=3000, precision=0)
            hc = gr.Number(label="Head Circumference (cm)", value=33, precision=2)
            dm = gr.Dropdown([0, 1], value=0, label="Delivery Mode (0=vaginal, 1=cesarean)")
            apgar1 = gr.Number(label="Apgar Score 1min (0-10)", value=8, precision=0)
            apgar5 = gr.Number(label="Apgar Score 5min (0-10)", value=9, precision=0)
            temp = gr.Number(label="Temperature (°C)", value=37.0, precision=2)
            hr = gr.Number(label="Heart Rate (bpm)", value=140, precision=0)
            rr = gr.Number(label="Respiratory Rate (breaths/min)", value=50, precision=0)
            spo2 = gr.Number(label="SpO2 (%)", value=97, precision=0)
    
    # Tab switching logic
    def toggle_gen(*args):
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    def toggle_up(*args):
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    def toggle_rec(*args):
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
    tab_gen.click(toggle_gen, outputs=[gen_group, up_group, rec_group])
    tab_up.click(toggle_up, outputs=[gen_group, up_group, rec_group])
    tab_rec.click(toggle_rec, outputs=[gen_group, up_group, rec_group])
    
    # Generate random sample
    def gen_sample():
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        return 16000, audio
    
    gen_btn.click(gen_sample, outputs=audio_input)
    
    # Prediction button
    with gr.Row():
        predict_btn = gr.Button("🔮 Generate Prediction", size="lg", variant="primary")
    
    with gr.Row():
        output = gr.HTML(value="Results will appear here...")
    
    
    # Connect prediction
    predict_btn.click(
        predict,
        inputs=[audio_input, ga, bw, hc, dm, apgar1, apgar5, temp, hr, rr, spo2],
        outputs=output
    )
    
    # Info section
    with gr.Accordion("ℹ️ About NeoScreen", open=False):
        with gr.Column():
            gr.Markdown("""
            ### 🏗️ System Architecture
            
            **Audio Analysis (70% weight)**
            - CNN-LSTM neural network
            - Processes mel-spectrogram features
            - Detects acoustic biomarkers from patient audio
            
            **Clinical Analysis (30% weight)**
            - Extreme Learning Machine (ELM)
            - Analyzes 10 vital signs & measurements
            - Captures neonatal health indicators
            
            **Fusion Strategy**
            - Weighted averaging combines both modalities
            - Leverages audio + clinical complementarity
            
            ---
            
            ### 📖 How to Use
            1. **Record or Upload** - Capture patient audio (cries, respiratory sounds, vocalizations)
            2. **Enter Clinical Data** - Input the 10 vital signs from patient assessment
            3. **Generate Assessment** - Click the blue button to analyze
            4. **Review Results** - Check risk classification and confidence scores
            
            ---
            
            ### ⚠️ Important Limitations
            - Audio: Minimum 0.5 seconds required
            - Clinical: Values must be within realistic ranges
            - Results: Recommendations only, not definitive diagnoses
            - Use: Intended for clinical decision support only
            
            ---
            
            **License:** CreativeML OpenRAIL-M  
            **Version:** 1.0  
            **Last Updated:** April 6, 2026
            """)

if __name__ == "__main__":
    demo.launch(share=False)
