# 📊 Bias and Fairness in Self-Supervised Acoustic Representations for Cognitive Impairment Detection.

This repository contains the code, datasets information, and resources used in our study:

**"Bias and Fairness in Self-Supervised Acoustic Representations for Cognitive Impairment Detection"**  
*Submitted to Speech Communication (2026)*

---

## 📄 Table of Contents
- [Project Description](#project-description)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## 📖 Project Description

This project investigates the use of acoustic representations for the automatic classification of Cognitive 
Impairment (CI) and depression within CI populations, with a specific focus on identifying and quantifying demographic 
and clinical biases in model performance.

We compare traditional acoustic features (MFCCs, eGeMAPS) against transformer-based embeddings (Wav2Vec 2.0) for CI and
depression detection using classical machine learning classifiers. Beyond overall performance, the project conducts bias
analysis across age, gender, and depression status, highlighting fairness concerns in speech-based clinical models.

Key components include:

- Performance benchmarking across imbalanced, CI-balanced, and CI-gender balanced datasets.
- Layer-wise analysis of Wav2Vec 2.0 embeddings.
- Subgroup-wise bias quantification in CI classification outcomes.
- Cross-condition generalization experiments between CI and depression tasks.

---

## 📂 Repository Structure
```
├── data/
│   └── dataset_filenames.xlsx            # Dataset filenames (see Datasets section)
│
├── src/
│   ├── Classifiers/                      # Classifier training and evaluation scripts
│   ├── feature extraction scripts        # MFCC, eGeMAPS, and W2V2 extraction
│   ├── bias analysis scripts             # Bias evaluation and analysis
│   └── supporting utilities              # Helper functions
│
├── requirements.txt                      # Python dependencies
└── README.md                             # Project overview (this file)

```

---

## 📊 Datasets

- The study uses a subset of the publicly available **Pitt Corpus** from the DementiaBank database.
- Lists of filenames for each dataset configuration are available in:
  - `data/dataset_filenames.xlsx`

> ⚠️ **Note:** Raw audio files are not included due to data usage restrictions. Please obtain access via DementiaBank: https://dementia.talkbank.org

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/th-nuernberg/taslp-bias-in-dementia
   cd taslp-bias-in-dementia

2. **Create a virtual environment and install dependencies:**
    
   ```bash
    python3.9 -m venv venv
    source venv/bin/activate  # for macOS/Linux
    venv\Scripts\activate     # for Windows
    pip install -r requirements.txt

---

## 🚀 Usage

1. **Extract Features:**
- Run `src/wav2vec.py` to compute Wav2Vec 2.0 hidden layers embeddings.
- Use `src/acoustic_feature_extraction.py` to generate pickle file of all features i.e., MFCC, eGeMAPS and Wav2Vec 2.0.
2. **Evaluation and Bias Analysis:**
- Run `src/acoustic_classification.py` to evaluate classifiers for  CI vs. NCI and D-CI vs. ND-CI and subgroup performance evaluations.
3. **Customize Experiments:**
- Edit filepaths, configurations, and hyperparameters in scripts.
- Specify alternative dataset, seeds, features, models parameters, or data splits.


---

## 🔄 Reproducibility

To reproduce the reported results:

- Use the dataset filename lists in data/dataset_filenames.xlsx
- Use the same fixed random seeds: 0, 50, 100, 150, 200 for data splitting.
- Refer to the scripts for exact configurations and hyperparameters.






