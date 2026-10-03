# DialectClassification
This repository contains files for the academic paper 'Linguistically Grounded Whisper Adaptation for Tamil Dialects: Balancing Accuracy and Minority Robustness' that is submitted for review to SPELLL 2026
The dataset used is from the DravidianLangTech 2026 shared task on Dialect-based Speech Recognition and Classification in Tamil.

Project Overview:
We present a Mutual Information-guided Whisper adaptation approach using 26 handcrafted acoustic features and dialect-specific classification heads to classify 
four Tamil dialects: Central, Northern, Southern, and Western.
This work demonstrates that a linguistically motivated feature set guided by Mutual Information can achieve competitive performance while maintaining interpretability.
Our approach is grounded in phonological evidence that nasal realisation, retroflex bursts, and vowel centralisation vary across Tamil dialects.

Key Contributions
- 26 handcrafted acoustic features capturing prosodic, MFCC, spectral, nasalisation, and dialect-specific cues
- Mutual Information-guided feature weighting for each dialect head
- Frozen Whisper Small encoder with dialect-specific classification heads
- 5-fold speaker-disjoint cross-validation for robust evaluation
- Interpretable dialect-specific acoustic markers: retroflex bursts for Western Tamil, energy variation for Northern Tamil, and nasalisation for Southern Tamil
- Comparison with baselines: handcrafted-only, Whisper-only, LoRA adaptation, and learned attention variants

Submission:
train_dialect_classifier.py        - Main training script with 5-fold CV and final test prediction

# Linguistically Grounded Whisper Adaptation for Tamil Dialects
Code for the paper *Linguistically Grounded Whisper Adaptation for Tamil Dialects: Balancing Accuracy and Minority Robustness*, submitted to SPELLL 2026. 
Data: DravidianLangTech 2026 shared task on Dialect-based Speech Recognition and Classification in Tamil.

We classify four Tamil dialects (Central, Northern, Southern, Western) with a frozen Whisper Small encoder, 26 handcrafted acoustic features weighted per dialect by Mutual Information, and dialect-specific heads, evaluated with 5-fold speaker-disjoint cross-validation and the official test set.

## Files
- `train_dialect_classifier.py`: main script. Loads `final_heads_with_mi.pt` and writes test predictions, or trains with 5-fold CV if the checkpoint is missing.
- 'final_heads_with_mi.pt' : checkpoint of the trained model.
- `bootstrap_comment.py`: paired bootstrap against the handcrafted-only baseline.

## Usage
Place the data in `data/Train` and `data/Test`, then run from the repository root:

```bash
pip install torch openai-whisper librosa==0.10.1 scikit-learn scipy numpy tqdm
python train_dialect_classifier.py
```

Predictions are saved to `paper/review/final_predictions.txt`. The committed checkpoint is the model behind the proposed model's test results. 
The dataset is not included.
