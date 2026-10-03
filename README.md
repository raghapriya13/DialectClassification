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
