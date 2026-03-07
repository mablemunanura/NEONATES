# 🫁 NeoScreen: Multimodal AI for Neonatal Respiratory Distress Detection

[Python 3.9+](https://www.python.org/downloads/)
[License: MIT](https://opensource.org/licenses/MIT)

---

## 🔍 **Overview**

**NeoScreen** is a multimodal AI framework that combines **clinical data** and **respiratory sound analysis** to detect neonatal respiratory distress in low-resource settings. Developed as a machine learning project at Makerere University (Group SW-ML-5).

| **What it does**                                | **Why it matters**                               |
| ----------------------------------------------- | ------------------------------------------------ |
| Predicts respiratory distress risk in newborns  | 47% of under-5 deaths occur in the first 28 days |
| Analyzes grunting, stridor, and apnea sounds    | 98% of these deaths happen in LMICs              |
| Fuses clinical + audio data for better accuracy | Respiratory distress is the 3rd leading cause    |
| Runs on smartphones                             | Current screening is subjective and delayed      |

---

## ⚠️ **The Problem**

- **47%** of under-5 deaths occur in the first 28 days of life
- **98%** occur in low- and middle-income countries (LMICs)
- **Respiratory distress** is the third leading cause of neonatal mortality
- Current screening is **subjective** → **delayed interventions** → **preventable deaths**

---

## 💡 **Our Solution**

NeoScreen fuses two complementary data modalities:

| 📊 **Tabular Path** | 🎵 **Audio Path**  |
| ------------------- | ------------------ |
| Gestational Age     | Grunting detection |
| Birth Weight        | Stridor detection  |
| Respiratory Rate    | Apnea detection    |
| SpO₂                | MFCC features      |
| Apgar scores        | Harmonic Ratio     |
| Temperature         | CNN embeddings     |
| Heart Rate          | (ResNet-50, 256-d) |

**Fusion Strategy:** Early fusion with XGBoost achieves **AUC = 0.91**

---

## 🏆 **Key Results**

| Metric      | Tabular Only | **Early Fusion (Ours)** | Improvement |
| ----------- | ------------ | ----------------------- | ----------- |
| **AUC**     | 0.86         | **0.91**                | **+5%**     |
| Sensitivity | 82%          | **88%**                 | +6%         |
| Specificity | 88%          | **92%**                 | +4%         |

### **Key Findings**

- **Harmonic ratio** (acoustic feature of grunting) is the strongest predictor (importance = 0.21)
- Grunting **doubles** probability of respiratory distress (0.31 → 0.92)

---

## 📊 **Dataset Access**

The datasets used in this project are too large for GitHub. Download them from the links below:

### **1. Synthetic Neonatal Dataset**

- **Source:** Electric Sheep Africa (Hugging Face)
- **Link:** [https://huggingface.co/datasets/electricsheepafrica/synthetic-neonatal-birth-outcomes-vitals-WHO-0-28days](https://huggingface.co/datasets/electricsheepafrica/synthetic-neonatal-birth-outcomes-vitals-WHO-0-28days)
- **Contents:** 30,000 synthetic neonatal records with clinical variables

### **2. Respiratory Sound Datasets**

| Dataset         | Size             | Contents                     | Download Link                                                                  |
| --------------- | ---------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **ICBHI 2017**  | 920 recordings   | Crackles, wheezes            | [Official Link](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)         |
| **HLS-CMDS v2** | 535 recordings   | Wheezing, crackles, rhonchi  | [Github Link](https://github.com/Torabiy/HLS-CMDS/tree/main)                   |
| **SPRSound**    | 2,683 recordings | Pediatric respiratory sounds | [Github Link](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound/tree/main)  |

---

## ⚙️ **Installation**

### **Prerequisites**

- Python 3.9+
- pip
- virtualenv (recommended)

### **Setup Instructions**

```bash
# 1. Clone the repository
git clone https://github.com/mablemunanura/NEONATES.git
cd NEONATES

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r dependencies.txt

# 5. Download datasets (see Dataset Access section above)NEONATES/
# Place them in the correct folder structure

# 6. Verify setup
python -c "import librosa; import pandas; print('Setup successful!')"

```

---

### **Folder Structure**

├── .venv/                    # Virtual environment (ignored by git)
├── clinical_data/            # Sythetic tabular datasets (ignored by git)
├── sound_data/               # Audio datasets (ignored by git)
├── notebooks/
│   ├── ClinicalData.ipynb
│   ├── SoundData.ipynb
├── models/                    # Trained models
├── requirements.txt
├── .gitignore
└── README.md

---

### **🙏 Acknowledgments**

Prof. Ggaliwango Marvin – For the 8-layer computational thinking framework and continuous guidance

Electric Sheep Africa – For the synthetic neonatal dataset

SPRSound, ICBHI 2017, and HLS-CMDS teams – For making respiratory sound data publicly available

Makerere University, Department of Computer Science – For institutional support

---

### **Team**

| Name | Email |
| Abisha Baingana | <abishabaingana1@gmail.com> |
| Mable Tusiime | <mablemunanura@gmail.com> |

Supervisor: Dr. Ggaliwango Marvin
Institution: Department of Computer Science, College of Computing and Information Sciences, Makerere University
