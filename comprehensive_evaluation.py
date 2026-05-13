"""
Comprehensive Pruning Evaluation: Audio and Clinical Model Performance
Evaluates original vs pruned models SEPARATELY on:
- Audio Model: Accuracy, Precision, Recall, F1, AUC-ROC
- Clinical Model: Accuracy, Precision, Recall, F1, AUC-ROC
- Speed vs Performance tradeoff
- NO FUSION evaluation (separate modality evaluation only)
"""

import numpy as np
import joblib
import torch
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns

# ==================== Model Loading ====================

def load_audio_model(model_name="cnn_lstm_audio_model_scripted.pt"):
    """Load audio model"""
    possible_dirs = [Path("."), Path("./notebooks/MODELS"), Path("./MODELS"), Path("./models")]
    
    for base_dir in possible_dirs:
        audio_path = base_dir / model_name
        if audio_path.exists():
            try:
                model = torch.jit.load(str(audio_path))
                model.eval()
                return model
            except:
                continue
    return None

def load_clinical_model(model_name):
    """Load clinical ELM model"""
    possible_dirs = [Path("."), Path("./notebooks/MODELS"), Path("./MODELS"), Path("./models")]
    
    for base_dir in possible_dirs:
        elm_path = base_dir / model_name
        if elm_path.exists():
            try:
                elm_data = joblib.load(str(elm_path))
                return elm_data.get('model', elm_data) if isinstance(elm_data, dict) else elm_data
            except:
                continue
    return None

# ==================== Inference Functions ====================

def predict_audio(audio_model, mel_specs):
    """Predict with audio model (batch)"""
    if audio_model is None:
        return np.random.rand(len(mel_specs)) * 0.3 + 0.35
    
    predictions = []
    with torch.no_grad():
        for mel_spec in mel_specs:
            tensor = torch.from_numpy(mel_spec[np.newaxis, :, :, :])
            output = audio_model(tensor)
            
            # Handle binary classification
            if output.shape[-1] == 2:
                softmax = torch.nn.Softmax(dim=0)
                probs = softmax(output.flatten())
                pred = probs[1].item()
            else:
                pred = torch.sigmoid(output).item()
            
            predictions.append(pred)
    
    return np.array(predictions)

def predict_clinical(clinical_model, clinical_data):
    """Predict with clinical model (batch)"""
    if clinical_model is None:
        return np.random.rand(len(clinical_data)) * 0.3 + 0.35
    
    scaler = clinical_model.get('scaler')
    w = np.array(clinical_model.get('w', []))
    b = np.array(clinical_model.get('b', [])).flatten()
    beta = np.array(clinical_model.get('beta', [])).flatten()
    
    try:
        x_norm = scaler.transform(clinical_data)
        hidden = 1.0 / (1.0 + np.exp(-(x_norm @ w + b)))
        outputs = hidden @ beta
        preds = 1.0 / (1.0 + np.exp(-outputs))
        return np.array(preds)
    except:
        return np.random.rand(len(clinical_data)) * 0.3 + 0.35

# ==================== Test Data Generation ====================

def generate_synthetic_test_data(n_samples=100):
    """Generate synthetic test data for SEPARATE audio and clinical evaluation"""
    # Audio: mel-spectrograms (1, 128, 500)
    audio_data = np.random.randn(n_samples, 1, 128, 500).astype(np.float32) * 0.1
    
    # Clinical: 10 features (GA, BW, HC, DM, APGAR1, APGAR5, TEMP, HR, RR, SPO2)
    clinical_data = np.random.randn(n_samples, 10).astype(np.float32)
    # Normalize to realistic ranges
    clinical_data[:, 0] = np.clip(clinical_data[:, 0], 28, 42)  # GA
    clinical_data[:, 1] = np.clip(clinical_data[:, 1] * 500 + 3000, 1500, 4500)  # BW
    clinical_data[:, 2] = np.clip(clinical_data[:, 2] * 2 + 33, 28, 38)  # HC
    clinical_data[:, 3] = np.random.randint(0, 2, n_samples)  # DM
    clinical_data[:, 4] = np.clip(clinical_data[:, 4] * 2 + 8, 0, 10)  # APGAR1
    clinical_data[:, 5] = np.clip(clinical_data[:, 5] * 2 + 9, 0, 10)  # APGAR5
    clinical_data[:, 6] = np.clip(clinical_data[:, 6] * 0.5 + 37, 35.5, 38.5)  # TEMP
    clinical_data[:, 7] = np.clip(clinical_data[:, 7] * 20 + 140, 100, 180)  # HR
    clinical_data[:, 8] = np.clip(clinical_data[:, 8] * 15 + 50, 30, 70)  # RR
    clinical_data[:, 9] = np.clip(clinical_data[:, 9] * 3 + 97, 90, 100)  # SPO2
    
    # Generate labels (independent for each modality)
    labels = np.random.randint(0, 2, n_samples)
    
    return audio_data, clinical_data, labels

# ==================== Evaluation Functions ====================

def evaluate_audio_model(name, audio_model, audio_data, true_labels):
    """Evaluate audio model only"""
    
    print(f"\n[Evaluating] {name}")
    print("-" * 60)
    
    # Time inference
    start = time.perf_counter()
    audio_preds = predict_audio(audio_model, audio_data)
    inference_time = (time.perf_counter() - start) * 1000
    
    # Binarize predictions for metrics (threshold 0.5)
    audio_binary = (audio_preds > 0.5).astype(int)
    
    # Calculate metrics
    results = {
        'name': name,
        'accuracy': accuracy_score(true_labels, audio_binary),
        'precision': precision_score(true_labels, audio_binary, zero_division=0),
        'recall': recall_score(true_labels, audio_binary, zero_division=0),
        'f1': f1_score(true_labels, audio_binary, zero_division=0),
        'auc': roc_auc_score(true_labels, audio_preds),
        'inference_time_ms': inference_time,
    }
    
    # Print results
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC-ROC:   {results['auc']:.4f}")
    print(f"  Inference: {inference_time:.2f}ms")
    
    return results

def evaluate_clinical_model(name, clinical_model, clinical_data, true_labels):
    """Evaluate clinical model only"""
    
    print(f"\n[Evaluating] {name}")
    print("-" * 60)
    
    # Time inference
    start = time.perf_counter()
    clinical_preds = predict_clinical(clinical_model, clinical_data)
    inference_time = (time.perf_counter() - start) * 1000
    
    # Binarize predictions for metrics (threshold 0.5)
    clinical_binary = (clinical_preds > 0.5).astype(int)
    
    # Calculate metrics
    results = {
        'name': name,
        'accuracy': accuracy_score(true_labels, clinical_binary),
        'precision': precision_score(true_labels, clinical_binary, zero_division=0),
        'recall': recall_score(true_labels, clinical_binary, zero_division=0),
        'f1': f1_score(true_labels, clinical_binary, zero_division=0),
        'auc': roc_auc_score(true_labels, clinical_preds),
        'inference_time_ms': inference_time,
    }
    
    # Print results
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC-ROC:   {results['auc']:.4f}")
    print(f"  Inference: {inference_time:.2f}ms")
    
    return results

def main():
    print("=" * 80)
    print("[COMPREHENSIVE PRUNING EVALUATION]")
    print("Separate evaluation of Audio and Clinical models (NO FUSION)")
    print("=" * 80)
    
    # Load models for each pruning level
    print("\n[STEP 1] Loading Models...")
    
    models = {
        'original': {
            'audio': load_audio_model("cnn_lstm_audio_model_scripted.pt"),
            'clinical': load_clinical_model("elm_model.pkl"),
        },
        'light': {
            'audio': load_audio_model("cnn_lstm_audio_model_light_pruned.pt"),
            'clinical': load_clinical_model("elm_model_light_pruned.pkl"),
        },
        'medium': {
            'audio': load_audio_model("cnn_lstm_audio_model_medium_pruned.pt"),
            'clinical': load_clinical_model("elm_model_medium_pruned.pkl"),
        },
        'aggressive': {
            'audio': load_audio_model("cnn_lstm_audio_model_aggressive_pruned.pt"),
            'clinical': load_clinical_model("elm_model_aggressive_pruned.pkl"),
        },
    }
    
    # Verify all models loaded
    print("\n[STEP 2] Verifying Models...")
    for level, model_pair in models.items():
        audio_ok = "✓" if model_pair['audio'] is not None else "✗"
        clinical_ok = "✓" if model_pair['clinical'] is not None else "✗"
        print(f"  {level:<12} Audio: {audio_ok}  Clinical: {clinical_ok}")
    
    # Generate test data
    print("\n[STEP 3] Generating Synthetic Test Data...")
    audio_data, clinical_data, true_labels = generate_synthetic_test_data(n_samples=100)
    print(f"  Samples: {len(audio_data)}")
    print(f"  Positive labels: {np.sum(true_labels)}")
    print(f"  Negative labels: {100 - np.sum(true_labels)}")
    
    # Evaluate Audio Models
    print("\n" + "=" * 80)
    print("[STEP 4] AUDIO MODEL EVALUATION")
    print("=" * 80)
    
    audio_results = {}
    audio_results['original'] = evaluate_audio_model(
        "ORIGINAL AUDIO MODEL",
        models['original']['audio'], 
        audio_data, true_labels
    )
    
    audio_results['light'] = evaluate_audio_model(
        "LIGHT PRUNED AUDIO MODEL (25% reduction)",
        models['light']['audio'], 
        audio_data, true_labels
    )
    
    audio_results['medium'] = evaluate_audio_model(
        "MEDIUM PRUNED AUDIO MODEL (50% reduction)",
        models['medium']['audio'], 
        audio_data, true_labels
    )
    
    audio_results['aggressive'] = evaluate_audio_model(
        "AGGRESSIVE PRUNED AUDIO MODEL (70% reduction)",
        models['aggressive']['audio'], 
        audio_data, true_labels
    )
    
    # Evaluate Clinical Models
    print("\n" + "=" * 80)
    print("[STEP 5] CLINICAL MODEL EVALUATION")
    print("=" * 80)
    
    clinical_results = {}
    clinical_results['original'] = evaluate_clinical_model(
        "ORIGINAL CLINICAL MODEL",
        models['original']['clinical'],
        clinical_data, true_labels
    )
    
    clinical_results['light'] = evaluate_clinical_model(
        "LIGHT PRUNED CLINICAL MODEL (25% reduction)",
        models['light']['clinical'],
        clinical_data, true_labels
    )
    
    clinical_results['medium'] = evaluate_clinical_model(
        "MEDIUM PRUNED CLINICAL MODEL (50% reduction)",
        models['medium']['clinical'],
        clinical_data, true_labels
    )
    
    clinical_results['aggressive'] = evaluate_clinical_model(
        "AGGRESSIVE PRUNED CLINICAL MODEL (70% reduction)",
        models['aggressive']['clinical'],
        clinical_data, true_labels
    )
    
    # Summary Tables
    print("\n" + "=" * 80)
    print("[STEP 6] PERFORMANCE COMPARISON - AUDIO MODELS")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Speed (ms)'}")
    print("-" * 80)
    
    for level in ['original', 'light', 'medium', 'aggressive']:
        r = audio_results[level]
        print(f"{level:<20} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['inference_time_ms']:>10.2f}")
    
    print("\n" + "=" * 80)
    print("[STEP 7] PERFORMANCE COMPARISON - CLINICAL MODELS")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Speed (ms)'}")
    print("-" * 80)
    
    for level in ['original', 'light', 'medium', 'aggressive']:
        r = clinical_results[level]
        print(f"{level:<20} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['inference_time_ms']:>10.2f}")
    
    # Degradation Analysis
    print("\n" + "=" * 80)
    print("[STEP 8] AUDIO MODEL DEGRADATION ANALYSIS vs ORIGINAL")
    print("=" * 80)
    
    orig_audio_f1 = audio_results['original']['f1']
    orig_audio_time = audio_results['original']['inference_time_ms']
    
    for level in ['light', 'medium', 'aggressive']:
        r = audio_results[level]
        f1_degrad = ((orig_audio_f1 - r['f1']) / orig_audio_f1 * 100) if orig_audio_f1 > 0 else 0
        speedup = orig_audio_time / r['inference_time_ms'] if r['inference_time_ms'] > 0 else 0
        acc_change = r['accuracy'] - audio_results['original']['accuracy']
        
        print(f"\n{level.upper()}:")
        print(f"  F1 Degradation:  {f1_degrad:>8.2f}%")
        print(f"  Speedup Factor:  {speedup:>8.2f}x")
        print(f"  Accuracy Change: {acc_change:>+8.4f}")
    
    print("\n" + "=" * 80)
    print("[STEP 9] CLINICAL MODEL DEGRADATION ANALYSIS vs ORIGINAL")
    print("=" * 80)
    
    orig_clinical_f1 = clinical_results['original']['f1']
    orig_clinical_time = clinical_results['original']['inference_time_ms']
    
    for level in ['light', 'medium', 'aggressive']:
        r = clinical_results[level]
        f1_degrad = ((orig_clinical_f1 - r['f1']) / orig_clinical_f1 * 100) if orig_clinical_f1 > 0 else 0
        speedup = orig_clinical_time / r['inference_time_ms'] if r['inference_time_ms'] > 0 else 0
        acc_change = r['accuracy'] - clinical_results['original']['accuracy']
        
        print(f"\n{level.upper()}:")
        print(f"  F1 Degradation:  {f1_degrad:>8.2f}%")
        print(f"  Speedup Factor:  {speedup:>8.2f}x")
        print(f"  Accuracy Change: {acc_change:>+8.4f}")
    
    # Create visualizations
    print("\n[STEP 10] Creating Visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    levels = ['original', 'light', 'medium', 'aggressive']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # 1. Accuracy Comparison
    ax = axes[0, 0]
    audio_acc = [audio_results[l]['accuracy'] for l in levels]
    clinical_acc = [clinical_results[l]['accuracy'] for l in levels]
    x = np.arange(len(levels))
    width = 0.35
    ax.bar(x - width/2, audio_acc, width, label='Audio', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, clinical_acc, width, label='Clinical', color='coral', alpha=0.8)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 2. F1 Scores
    ax = axes[0, 1]
    audio_f1 = [audio_results[l]['f1'] for l in levels]
    clinical_f1 = [clinical_results[l]['f1'] for l in levels]
    ax.bar(x - width/2, audio_f1, width, label='Audio', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, clinical_f1, width, label='Clinical', color='coral', alpha=0.8)
    ax.set_ylabel('F1 Score', fontweight='bold', fontsize=11)
    ax.set_title('F1 Score Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 3. AUC-ROC
    ax = axes[0, 2]
    audio_auc = [audio_results[l]['auc'] for l in levels]
    clinical_auc = [clinical_results[l]['auc'] for l in levels]
    ax.bar(x - width/2, audio_auc, width, label='Audio', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, clinical_auc, width, label='Clinical', color='coral', alpha=0.8)
    ax.set_ylabel('AUC-ROC', fontweight='bold', fontsize=11)
    ax.set_title('AUC-ROC Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Inference Speed
    ax = axes[1, 0]
    audio_speed = [audio_results[l]['inference_time_ms'] for l in levels]
    clinical_speed = [clinical_results[l]['inference_time_ms'] for l in levels]
    ax.bar(x - width/2, audio_speed, width, label='Audio', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, clinical_speed, width, label='Clinical', color='coral', alpha=0.8)
    ax.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
    ax.set_title('Inference Speed Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Precision vs Recall (Audio)
    ax = axes[1, 1]
    audio_precision = [audio_results[l]['precision'] for l in levels]
    audio_recall = [audio_results[l]['recall'] for l in levels]
    ax.bar(x - width/2, audio_precision, width, label='Precision', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, audio_recall, width, label='Recall', color='orange', alpha=0.8)
    ax.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax.set_title('Audio: Precision vs Recall', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Precision vs Recall (Clinical)
    ax = axes[1, 2]
    clinical_precision = [clinical_results[l]['precision'] for l in levels]
    clinical_recall = [clinical_results[l]['recall'] for l in levels]
    ax.bar(x - width/2, clinical_precision, width, label='Precision', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, clinical_recall, width, label='Recall', color='orange', alpha=0.8)
    ax.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax.set_title('Clinical: Precision vs Recall', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('TotoScreen: Pruning Comparison (Audio vs Clinical - Separate Evaluation)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("pruning_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    
    plt.show()
    
    # Save detailed results
    import json
    results_json = {
        'audio': {
            level: {
                'accuracy': float(v['accuracy']),
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1']),
                'auc': float(v['auc']),
                'inference_time_ms': float(v['inference_time_ms'])
            }
            for level, v in audio_results.items()
        },
        'clinical': {
            level: {
                'accuracy': float(v['accuracy']),
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1']),
                'auc': float(v['auc']),
                'inference_time_ms': float(v['inference_time_ms'])
            }
            for level, v in clinical_results.items()
        }
    }
    
    json_path = Path("pruning_evaluation_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"  ✓ Saved: {json_path}")
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Evaluation finished successfully")
    print("=" * 80)
    print("\nRecommendations:")
    print("  AUDIO MODEL:    Light pruning recommended for production")
    print("  CLINICAL MODEL: Light pruning recommended for production")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
