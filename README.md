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
