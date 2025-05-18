# AI System for Automatic Detection of Deceptive Speech

## Project Objective

Develop a machine learning system for the automatic detection of lies from speech (deceptive vs. sincere speech).  
**Course:** BIOSINF I - IAECV

---

## Setup Instructions

### 1. Python Version

- Python 3.9+ required (developed with Python 3.11.9).

### 2. Virtual Environment

It is highly recommended to use a virtual environment:

```bash
python -m venv .venv
```

### 3. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

If you encounter missing packages, try:

```bash
pip install numpy pandas scikit-learn librosa tensorflow tqdm matplotlib jupyterlab
```

**For GPU support with TensorFlow:**  
Ensure you have compatible NVIDIA drivers, CUDA Toolkit, and cuDNN SDK as per the [official TensorFlow GPU guide](https://www.tensorflow.org/install/gpu).

---

## Data

- **Audio files:** `data/extrAudio/` (WAV, 16 kHz, 16-bit PCM)
- **Annotations:** `data/datasetAnnotation/` (CSV, utterance-level, speaker ID, gender)
- **Utterances:** 929 total (463 deceptive, 466 sincere)
- **Speakers:** IDs 1-56 (subjects), 'TM' (interviewers, filtered out)
- **Work unit:** Utterance level

---

## Running the Scripts

Scripts should be run from the `scripts/` directory or project root.

### 1. Data Exploration

**Notebook:** `notebooks/data_exploration.ipynb`  
**Purpose:** Initial analysis, waveform visualization, utterance stats.

```bash
jupyter lab notebooks/data_exploration.ipynb
# or
jupyter notebook notebooks/data_exploration.ipynb
```

---

### 2. Feature Extraction

**Script:** `scripts/feature_extraction.py`  
**Purpose:** Extract MFCCs, F0 stats (normalized per speaker), and spectrograms.

```bash
python scripts/feature_extraction.py
```

**Outputs:**
- `data/algorithmic_features_normalized.npz`
- `data/spectrogram_features.npz`

*Note: Script skips extraction if output files exist. Delete `.npz` files to force re-extraction.*

---

### 3. Preprocess Data

**Script:** `scripts/preprocess_data.py`  
**Purpose:** Create 5-fold speaker-independent cross-validation splits.

```bash
python scripts/preprocess_data.py
```

**Outputs:**
- `data/cv_speaker_splits.npz`

---

### 4. Train Models

**Script:** `scripts/train_models.py`  
**Purpose:** Train/evaluate SVM, Random Forest, FCNN, and CNN models.

**Examples:**

```bash
# Train all models (overwrite results)
python scripts/train_models.py

# Train only CNN models
python scripts/train_models.py --models cnn

# Train SVM and RF models
python scripts/train_models.py --models svm rf

# Train FCNN and append results
python scripts/train_models.py --models fcnn --append_results

# Train all models and append results
python scripts/train_models.py --append_results
```

**Inputs:**
- `data/algorithmic_features_normalized.npz`
- `data/spectrogram_features.npz`
- `data/cv_speaker_splits.npz`

**Outputs:**
- `results/training_results.csv`

*Note: Neural network training can be time-consuming. GPU recommended.*

---

### 5. Evaluate Models

**Script:** `scripts/evaluate_models.py` (to be implemented)  
**Purpose:** Summarize cross-validation results, identify best models.

```bash
python scripts/evaluate_models.py
```

**Inputs:** `results/training_results.csv`  
**Outputs:** Console summary, possible plots/report

---

## Project Requirements Summary

- **Features:**
  - Algorithmic: F0, ≥13 MFCCs (mean & std per utterance), speaker-normalized (z-score)
  - Spectrograms: 25ms frames, 10ms step, Hamming, 512 DFT, [0-8]kHz (257 bins), log amplitude, zero-padding

- **Models:**
  - SVM (≥10 configs)
  - Random Forest (≥10 configs)
  - FCNN (≥100 configs, algorithmic features)
  - CNN (≥20 configs, spectrograms)

- **Validation:** 5-fold cross-validation, speaker-independent, balanced class/gender distribution

- **Evaluation Metric:** Accuracy

- **Neural Network Specifics:** Sigmoid output, binary classification

---