#!/usr/bin/env python3
"""
Tamil dialect classification with frozen Whisper + MI priors + 5-fold CV.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import whisper
import librosa
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
RANDOM_SEED = 42
DATA_PATH = './data'
WHISPER_MODEL = "small"
BATCH_SIZE = 16
NUM_EPOCHS = 7
LEARNING_RATE = 2e-4
HEAD_HIDDEN_DIM = 128
HEAD_DROPOUT = 0.3
HANDCRAFTED_DIM = 26
WHISPER_N_FRAMES = 3000
N_FOLDS = 5
DURATION = 5
TARGET_SR = 16000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DIALECT_NAMES = ['Central', 'Northern', 'Southern', 'Western']
DIALECT_TO_IDX = {name: i for i, name in enumerate(DIALECT_NAMES)}
IDX_TO_DIALECT = {i: name for i, name in enumerate(DIALECT_NAMES)}

os.makedirs('paper/submission', exist_ok=True)
os.makedirs('paper/saved_final_model', exist_ok=True)
os.makedirs('cv_results', exist_ok=True)

# ==========================================
# FEATURE SET
# ==========================================
ENHANCED_FEATURES = [
    'rms_std', 'rms_mean', 'pitch_std', 'pitch_mean',
    'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean',
    'mfcc1_delta_mean', 'mfcc2_delta_mean', 'mfcc3_delta_mean',
    'mfcc4_delta_mean', 'mfcc5_delta_mean',
    'centroid_mean', 'rolloff_mean', 'f1_mean',
    'nasal_index', 'nasal_std', 'nasal_range',
    'vowel_centralization_north',
    'retroflex_burst_peak', 'alveolar_burst_peak',
    'prosodic_centrality',
    'vowel_centralization',
    'speech_rate',
]
NASAL_BAND = (200, 400)

# ==========================================
# FEATURE EXTRACTOR
# ==========================================
class PhonemeDetector:
    def __init__(self, sr=16000):
        self.sr = sr

    def detect_vowel_segments(self, audio):
        rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
        return rms > np.percentile(rms, 70)

    def detect_nasal_regions(self, audio):
        D = librosa.stft(audio, n_fft=512, hop_length=128)
        mag = np.abs(D)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=512)
        nasal_band = (freqs >= 250) & (freqs <= 500)
        nasal_energy = np.sum(mag[nasal_band, :], axis=0)
        total_energy = np.sum(mag, axis=0)
        return nasal_energy / (total_energy + 1e-10)


class FeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
        self.phoneme_detector = PhonemeDetector(sr)
        self.feature_names = ENHANCED_FEATURES

    def _get(self, d, key, default=0):
        return d.get(key, default)

    def extract(self, path, transcript=None):
        try:
            audio, sr = librosa.load(path, sr=self.sr, duration=DURATION)
            if len(audio) < self.sr:
                return None
            feats = {}
            # STFT
            D = librosa.stft(audio, n_fft=1024, hop_length=256)
            mag = np.abs(D)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            nasal_mask = (freqs >= NASAL_BAND[0]) & (freqs <= NASAL_BAND[1])
            oral_mask = (freqs >= NASAL_BAND[1]) & (freqs <= 3500)
            nasal_energy = np.sum(mag[nasal_mask, :], axis=0)
            oral_energy = np.sum(mag[oral_mask, :], axis=0)
            nasal_per_frame = nasal_energy / (oral_energy + 1e-10)
            feats['nasal_index'] = np.mean(nasal_per_frame)
            feats['nasal_std'] = np.std(nasal_per_frame)
            feats['nasal_range'] = np.max(nasal_per_frame) - np.min(nasal_per_frame)

            pitches, _ = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=300)
            pitches = pitches[pitches > 0]
            feats['pitch_mean'] = np.mean(pitches) if len(pitches) > 0 else 0
            feats['pitch_std'] = np.std(pitches) if len(pitches) > 0 else 0

            rms = librosa.feature.rms(y=audio)[0]
            feats['rms_mean'] = np.mean(rms)
            feats['rms_std'] = np.std(rms)

            cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            feats['centroid_mean'] = np.mean(cent)
            feats['rolloff_mean'] = np.mean(rolloff)

            try:
                lpc = librosa.lpc(audio, order=12)
                roots = np.roots(lpc)
                roots = roots[np.imag(roots) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                formants = sorted(angles * (sr / (2 * np.pi)))
                feats['f1_mean'] = formants[0] if formants else 0
            except:
                feats['f1_mean'] = 0

            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
            for i in range(5):
                feats[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            mfcc_delta = librosa.feature.delta(mfccs)
            for i in range(5):
                feats[f'mfcc{i+1}_delta_mean'] = np.mean(mfcc_delta[i])

            D2 = librosa.stft(audio, n_fft=512, hop_length=128)
            mag2 = np.abs(D2)
            freqs2 = librosa.fft_frequencies(sr=sr, n_fft=512)
            alveolar_band = (freqs2 >= 2500) & (freqs2 <= 4000)
            retroflex_band = (freqs2 >= 1500) & (freqs2 <= 2500)
            alveolar_energy = np.sum(mag2[alveolar_band, :], axis=0)
            retroflex_energy = np.sum(mag2[retroflex_band, :], axis=0)
            feats['alveolar_burst_peak'] = np.mean(alveolar_energy) if len(alveolar_energy) > 0 else 0
            feats['retroflex_burst_peak'] = np.mean(retroflex_energy) if len(retroflex_energy) > 0 else 0

            f1_mean = feats.get('f1_mean', 0)
            ref = 720
            ref2 = 680
            ref_central = (ref + ref2) / 2
            feats['prosodic_centrality'] = abs(f1_mean - ref_central) / (ref_central + 1e-10)

            if transcript:
                feats['speech_rate'] = len(transcript.replace(' ', '')) / DURATION
            else:
                feats['speech_rate'] = 0
            feats['vowel_centralization_north'] = abs(np.mean(mfccs[0]) - (-10)) / 20

            f1_band = (freqs2 >= 300) & (freqs2 <= 1000)
            f1_energy_band = np.sum(mag2[f1_band, :], axis=0)
            if np.sum(f1_energy_band) > 0:
                f1_centroid = np.sum(freqs2[f1_band] * np.mean(mag2[f1_band, :], axis=1)) / (np.mean(f1_energy_band) + 1e-10)
            else:
                f1_centroid = 600
            feats['vowel_centralization'] = abs(f1_centroid - 600) / 600

            return np.array([self._get(feats, name) for name in self.feature_names])
        except:
            return None

# ==========================================
# DATA LOADING
# ==========================================
def load_all_samples(data_path):
    dialect_folders = {'Central_Dialect':0, 'Northern_Dialect':1, 'Southern_Dialect':2, 'Western_Dialect':3}
    all_samples = []
    train_path = os.path.join(data_path, 'Train')
    for folder_name, d_idx in dialect_folders.items():
        folder_path = os.path.join(train_path, folder_name)
        if not os.path.exists(folder_path):
            continue
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item.endswith('_audio'):
                speaker = item.replace('_audio', '')
                transcript_path = os.path.join(folder_path, f"{speaker}_Text.txt")
                if not os.path.exists(transcript_path):
                    continue
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                wavs = sorted([os.path.join(item_path, f) for f in os.listdir(item_path)
                               if f.endswith('.wav') and f.startswith(speaker)])
                for i in range(min(len(wavs), len(lines))):
                    all_samples.append({
                        'audio_path': wavs[i],
                        'transcript': lines[i],
                        'dialect_idx': d_idx,
                        'speaker_id': speaker
                    })
    print(f"Loaded {len(all_samples)} samples.")
    return all_samples

# ==========================================
# WHISPER ENCODER WRAPPER
# ==========================================
class WhisperEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, mel):
        return self.encoder(mel).mean(dim=1)

# ==========================================
# DIALECT-SPECIFIC HEADS
# ==========================================
class DialectHeads(nn.Module):
    def __init__(self, whisper_dim=768, hand_dim=26, hidden=128, num_classes=4, dropout=0.3, mi_weights=None):
        super().__init__()
        self.hand_dim = hand_dim
        if mi_weights is not None:
            self.weights = [torch.tensor(w, dtype=torch.float32) for w in mi_weights]
        else:
            self.weights = [torch.ones(hand_dim) for _ in range(num_classes)]
        self.whisper_proj = nn.Sequential(
            nn.Linear(whisper_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_dim, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.GELU(),
            nn.Dropout(dropout*0.5)
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden + hidden//2, hidden//2),
                nn.BatchNorm1d(hidden//2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden//2, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, whisper_feats, hand_feats):
        whisper_out = self.whisper_proj(whisper_feats)
        logits = []
        for i, head in enumerate(self.heads):
            weight = self.weights[i].to(hand_feats.device)
            weighted_hand = hand_feats * weight
            hand_out = self.hand_proj(weighted_hand)
            combined = torch.cat([whisper_out, hand_out], dim=1)
            logits.append(head(combined))
        return torch.cat(logits, dim=1)

# ==========================================
# EXTRACT FEATURES
# ==========================================
def extract_whisper(samples, encoder, device):
    embs = []
    for s in tqdm(samples, desc="Whisper embeddings"):
        try:
            audio, sr = librosa.load(s['audio_path'], sr=TARGET_SR, duration=DURATION)
            if len(audio) < TARGET_SR*DURATION:
                audio = np.pad(audio, (0, TARGET_SR*DURATION - len(audio)), 'constant')
            else:
                audio = audio[:TARGET_SR*DURATION]
            mel = whisper.log_mel_spectrogram(torch.FloatTensor(audio))
            if mel.shape[1] < WHISPER_N_FRAMES:
                mel = F.pad(mel, (0, WHISPER_N_FRAMES - mel.shape[1]))
            else:
                mel = mel[:, :WHISPER_N_FRAMES]
            mel = mel.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder(mel).cpu().numpy().flatten()
            embs.append(emb)
        except:
            embs.append(np.zeros(768))
    return np.array(embs)

def extract_hand(samples, extractor):
    feats, labels = [], []
    for s in tqdm(samples, desc="Handcrafted features"):
        f = extractor.extract(s['audio_path'], s.get('transcript'))
        feats.append(f if f is not None else np.zeros(HANDCRAFTED_DIM))
        labels.append(s['dialect_idx'])
    return np.array(feats), np.array(labels)

# ==========================================
# TRAIN ONE FOLD
# ==========================================
def train_fold(train_whisper, train_hand, train_y, val_whisper, val_hand, val_y, mi_weights, fold_idx):
    train_whisper = torch.FloatTensor(train_whisper)
    train_hand = torch.FloatTensor(train_hand)
    train_y = torch.LongTensor(train_y)
    val_whisper = torch.FloatTensor(val_whisper)
    val_hand = torch.FloatTensor(val_hand)
    val_y = torch.LongTensor(val_y)

    train_loader = DataLoader(TensorDataset(train_whisper, train_hand, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_whisper, val_hand, val_y), batch_size=BATCH_SIZE, shuffle=False)

    model = DialectHeads(whisper_dim=768, hand_dim=HANDCRAFTED_DIM, hidden=HEAD_HIDDEN_DIM,
                         num_classes=4, dropout=HEAD_DROPOUT, mi_weights=mi_weights).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_f1 = 0.0
    best_state = None
    patience = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for w, h, y in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}"):
            w, h, y = w.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(w, h)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for w, h, y in val_loader:
                w, h, y = w.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
                logits = model(w, h)
                _, pred = torch.max(logits, 1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"  Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
    model.load_state_dict(best_state)
    return best_f1, val_acc

# ==========================================
# MAIN
# ==========================================
def main():
    all_samples = load_all_samples(DATA_PATH)
    if not all_samples:
        return

    speaker_to_dialect = {s['speaker_id']: s['dialect_idx'] for s in all_samples}
    speakers = list(speaker_to_dialect.keys())
    speaker_dialects = [speaker_to_dialect[spk] for spk in speakers]

    # Load frozen Whisper encoder
    whisper_model = whisper.load_model(WHISPER_MODEL).to(DEVICE)
    encoder = WhisperEncoderWrapper(whisper_model.encoder)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    extractor = FeatureExtractor()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    all_mi_weights = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(speakers, speaker_dialects)):
        train_spks = [speakers[i] for i in train_idx]
        val_spks = [speakers[i] for i in val_idx]
        train_samples = [s for s in all_samples if s['speaker_id'] in train_spks]
        val_samples = [s for s in all_samples if s['speaker_id'] in val_spks]

        print(f"\nFold {fold_idx+1}: Train={len(train_samples)}, Val={len(val_samples)}")

        train_hand, train_y = extract_hand(train_samples, extractor)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_hand)

        # Compute MI weights per dialect
        mi_weights = []
        for d in range(4):
            y_bin = (train_y == d).astype(int)
            mi = mutual_info_classif(X_train_scaled, y_bin, random_state=RANDOM_SEED)
            mi_weights.append(mi)
        all_mi_weights.append(mi_weights)

        train_whisper = extract_whisper(train_samples, encoder, DEVICE)
        val_whisper = extract_whisper(val_samples, encoder, DEVICE)
        val_hand, val_y = extract_hand(val_samples, extractor)

        best_f1, best_acc = train_fold(train_whisper, train_hand, train_y,
                                       val_whisper, val_hand, val_y,
                                       mi_weights, fold_idx)
        fold_results.append({'acc': best_acc, 'f1': best_f1})
        print(f"Fold {fold_idx+1} Result: Acc={best_acc:.4f}, F1={best_f1:.4f}")

    # CV summary
    accs = [r['acc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
    print(f"\nCV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    with open('cv_results/cv_results.json', 'w') as f:
        json.dump({'mean_acc': mean_acc, 'std_acc': std_acc,
                   'mean_f1': mean_f1, 'std_f1': std_f1,
                   'fold_results': fold_results}, f, indent=2)

    # Final model on all data
    print("\nTraining final model on all data...")
    all_hand, all_y = extract_hand(all_samples, extractor)
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(all_hand)
    final_mi = []
    for d in range(4):
        y_bin = (all_y == d).astype(int)
        mi = mutual_info_classif(X_all_scaled, y_bin, random_state=RANDOM_SEED)
        final_mi.append(mi)
    all_whisper = extract_whisper(all_samples, encoder, DEVICE)

    # Train final heads (no validation, use full epochs)
    train_whisper = torch.FloatTensor(all_whisper)
    train_hand = torch.FloatTensor(all_hand)
    train_y = torch.LongTensor(all_y)
    train_loader = DataLoader(TensorDataset(train_whisper, train_hand, train_y),
                              batch_size=BATCH_SIZE, shuffle=True)

    model = DialectHeads(whisper_dim=768, hand_dim=HANDCRAFTED_DIM, hidden=HEAD_HIDDEN_DIM,
                         num_classes=4, dropout=HEAD_DROPOUT, mi_weights=final_mi).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for w, h, y in tqdm(train_loader, desc=f"Final Epoch {epoch+1}"):
            w, h, y = w.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(w, h)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1} train acc: {correct/total:.4f}")

    # Predict test set
    test_path = os.path.join(DATA_PATH, 'Test')
    if os.path.exists(test_path):
        test_files = sorted([os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.wav')])
        print(f"Predicting {len(test_files)} test files...")
        test_hand_list, test_whisper_list, test_ids = [], [], []
        for path in tqdm(test_files):
            fid = os.path.splitext(os.path.basename(path))[0]
            test_ids.append(fid)
            f = extractor.extract(path)
            test_hand_list.append(f if f is not None else np.zeros(HANDCRAFTED_DIM))
            try:
                audio, sr = librosa.load(path, sr=TARGET_SR, duration=DURATION)
                if len(audio) < TARGET_SR*DURATION:
                    audio = np.pad(audio, (0, TARGET_SR*DURATION - len(audio)), 'constant')
                else:
                    audio = audio[:TARGET_SR*DURATION]
                mel = whisper.log_mel_spectrogram(torch.FloatTensor(audio))
                if mel.shape[1] < WHISPER_N_FRAMES:
                    mel = F.pad(mel, (0, WHISPER_N_FRAMES - mel.shape[1]))
                else:
                    mel = mel[:, :WHISPER_N_FRAMES]
                mel = mel.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = encoder(mel).cpu().numpy().flatten()
                test_whisper_list.append(emb)
            except:
                test_whisper_list.append(np.zeros(768))
        test_whisper = np.array(test_whisper_list)
        test_hand = np.array(test_hand_list)

        model.eval()
        preds = []
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_whisper), torch.FloatTensor(test_hand)),
                                 batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for w, h in test_loader:
                w, h = w.to(DEVICE), h.to(DEVICE)
                logits = model(w, h)
                _, pred = torch.max(logits, 1)
                preds.extend(pred.cpu().numpy())

        # Save predictions
        with open('paper/submission/final_predictions.txt', 'w') as f:
            for fid, p in zip(test_ids, preds):
                f.write(f"{fid} {IDX_TO_DIALECT[p]}\n")
        print(f"Predictions saved to paper/submission/final_predictions.txt")

        # Save final model
        torch.save({
            'model_state': model.state_dict(),
            'mi_weights': final_mi,
            'feature_names': ENHANCED_FEATURES,
            'dialect_names': DIALECT_NAMES
        }, 'paper/saved_final_model/final_model.pt')
        print("Final model saved to paper/saved_final_model/final_model.pt")

if __name__ == "__main__":
    main()