#!/usr/bin/env python3
"""
GRAFN: Gender-Responsive Adaptive Feature Normalization

Implements:

Step 1: Acoustic Statistical Alignment
    - Local (short-window) mean/variance normalization in the time domain.

Step 2: Adaptive Spectral Rebalancing
    - Learns per-gender spectral filters H_gender(f) that map male/female
      average spectra to a common "neutral" target.

Step 3: Reconstruction and Output
    - Inverse STFT back to waveform for downstream ASR processing.

This is a *model-agnostic* front-end: you can plug the output waveform into
any ASR feature pipeline (MFCCs, log-mel, wav2vec, Whisper, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple

import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _moving_mean_var(x: np.ndarray, win_len: int, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local mean and variance over a sliding window centered at each sample.

    Approximation using convolution; window is symmetric of length win_len.
    """
    if win_len <= 1:
        mu = x
        var = np.zeros_like(x)
        return mu, var

    # Use a simple boxcar window for local stats
    kernel = np.ones(win_len, dtype=np.float64) / float(win_len)

    # Pad to reduce edge effects (reflective padding)
    pad = win_len // 2
    x_pad = np.pad(x, pad, mode="reflect")

    mean = np.convolve(x_pad, kernel, mode="same")[pad:-pad]

    # E[x^2]
    x2 = x_pad ** 2
    mean_x2 = np.convolve(x2, kernel, mode="same")[pad:-pad]

    var = np.maximum(mean_x2 - mean**2, eps)
    return mean, var


@dataclass
class GRAFNNormalizer:
    """
    Gender-Responsive Adaptive Feature Normalization (GRAFN) front-end.

    Usage pattern:
        grafn = GRAFNNormalizer(sr=16000)
        grafn.fit(train_files, train_genders)  # offline training

        y_norm = grafn.transform_waveform(y, gender="female")
        # -> feed y_norm into your ASR feature extraction
    """

    sr: int = 16000
    win_ms: float = 25.0         # window length for local stats (Step 1)
    n_fft: int = 512             # STFT size (Step 2)
    hop_ms: float = 10.0         # STFT hop
    min_gain: float = 0.1        # clamp spectral gains
    max_gain: float = 10.0
    eps: float = 1e-8

    # Learned filters: maps gender label -> magnitude response [n_freqs]
    gender_filters_: Dict[str, np.ndarray] = field(default_factory=dict)
    fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        wav_paths: Iterable[str],
        genders: Iterable[str],
        target_gender_labels: Tuple[str, str] = ("male", "female"),
    ) -> "GRAFNNormalizer":
        """
        Learn gender-specific spectral rebalancing filters H_gender(f).

        Parameters
        ----------
        wav_paths : iterable of str
            Paths to training waveforms.
        genders : iterable of str
            "male" / "female" labels aligned with wav_paths.
        target_gender_labels : tuple
            Which labels to treat as the two genders to equalize.

        Notes
        -----
        For each gender g in {male, female}, we:

            1. Apply Step 1 (local normalization) to each training file.
            2. Compute STFT magnitude |X_g(f, t)|.
            3. Accumulate average magnitude per frequency bin:
                   mean_mag_g(f) = E[|X_g(f, t)|].

        Then we define a neutral target spectrum:

            mean_target(f) = 0.5 * (mean_mag_male(f) + mean_mag_female(f))

        and compute per-gender gains:

            H_g(f) = mean_target(f) / (mean_mag_g(f) + eps)

        with gains clamped into [min_gain, max_gain].
        """
        male_label, female_label = target_gender_labels

        wav_paths = list(wav_paths)
        genders = list(genders)
        assert len(wav_paths) == len(genders), "wav_paths and genders must align"

        # STFT params
        hop_length = int(self.sr * self.hop_ms / 1000.0)

        # Accumulators
        male_sum = None
        female_sum = None
        male_count = 0
        female_count = 0

        for path, g in zip(wav_paths, genders):
            if g not in target_gender_labels:
                # Ignore non-binary labels in this simple baseline
                continue

            y, sr = librosa.load(path, sr=self.sr)
            y = self._ensure_mono(y)

            # Step 1: local normalization
            y_norm = self._local_normalize(y)

            # STFT
            X = librosa.stft(y_norm, n_fft=self.n_fft, hop_length=hop_length, window="hann")
            mag = np.abs(X)  # [n_freqs, n_frames]
            mag_mean = mag.mean(axis=1)  # [n_freqs]

            if g == male_label:
                if male_sum is None:
                    male_sum = mag_mean
                else:
                    male_sum += mag_mean
                male_count += 1
            elif g == female_label:
                if female_sum is None:
                    female_sum = mag_mean
                else:
                    female_sum += mag_mean
                female_count += 1

        if male_count == 0 or female_count == 0:
            raise RuntimeError("Need at least one male and one female example to fit GRAFN.")

        male_avg = male_sum / float(male_count)
        female_avg = female_sum / float(female_count)

        # Neutral target: simple arithmetic mean
        target_mag = 0.5 * (male_avg + female_avg)

        # Compute gains
        H_male = np.clip(target_mag / (male_avg + self.eps), self.min_gain, self.max_gain)
        H_female = np.clip(target_mag / (female_avg + self.eps), self.min_gain, self.max_gain)

        self.gender_filters_ = {
            male_label: H_male,
            female_label: H_female,
        }
        self.fitted_ = True
        return self

    def transform_waveform(self, y: np.ndarray, gender: Optional[str] = None) -> np.ndarray:
        """
        Apply full GRAFN transform to a single waveform.

        Parameters
        ----------
        y : np.ndarray
            Input waveform (mono).
        gender : str or None
            Gender label used to pick H_gender(f). If None or unknown,
            we fall back to no spectral rebalancing (i.e., Step 1 only).

        Returns
        -------
        y_tilde : np.ndarray
            Normalized waveform ˜x(t) ready for ASR feature extraction.
        """
        y = self._ensure_mono(y)

        # Step 1 — Acoustic Statistical Alignment
        y_norm = self._local_normalize(y)

        # Step 2 — Adaptive Spectral Rebalancing (if filter is available)
        hop_length = int(self.sr * self.hop_ms / 1000.0)
        X = librosa.stft(y_norm, n_fft=self.n_fft, hop_length=hop_length, window="hann")

        if self.fitted_ and gender in self.gender_filters_:
            H = self.gender_filters_[gender]  # [n_freqs]
            # Broadcast across time frames
            X_bal = H[:, None] * X
        else:
            # No-op if not fitted or gender unknown
            X_bal = X

        # Step 3 — Reconstruction
        y_tilde = librosa.istft(X_bal, hop_length=hop_length, window="hann", length=len(y))

        # Optional: small global gain normalization to avoid clipping
        max_abs = np.max(np.abs(y_tilde)) + self.eps
        if max_abs > 1.0:
            y_tilde = y_tilde / max_abs

        return y_tilde.astype(np.float32)

    def transform_file(self, wav_path: str, gender: Optional[str] = None) -> np.ndarray:
        """
        Convenience wrapper for: read file -> GRAFN -> waveform.
        """
        y, sr = librosa.load(wav_path, sr=self.sr)
        return self.transform_waveform(y, gender=gender)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_mono(self, y: np.ndarray) -> np.ndarray:
        """Convert multi-channel to mono by averaging."""
        if y.ndim == 1:
            return y
        return np.mean(y, axis=0)

    def _local_normalize(self, y: np.ndarray) -> np.ndarray:
        """
        Step 1: Local mean/variance normalization on waveform.

        Implements:
            μ_x(t)   = E_{τ in W}[x(τ)]
            σ_x^2(t) = E_{τ in W}[(x(τ) - μ_x(t))^2]
            x_norm(t) = (x(t) - μ_x(t)) / σ_x(t)
        """
        win_len = int(self.sr * self.win_ms / 1000.0)
        if win_len < 1:
            win_len = 1

        mu, var = _moving_mean_var(y, win_len, eps=self.eps)
        std = np.sqrt(var + self.eps)
        x_norm = (y - mu) / std
        return x_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Example: end-to-end preprocessing & ASR call
# ---------------------------------------------------------------------------

def example_asr_feature_fn(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Placeholder ASR feature function.

    Replace this with whatever your ASR expects:
        - MFCCs
        - log-mel spectrogram
        - direct waveform (for wav2vec / Whisper / etc.)
    """
    # Example: log-mel spectrogram
    n_fft = 400
    hop_length = 160
    n_mels = 80

    S = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0,
    )
    log_mel = librosa.power_to_db(S**2, ref=np.max)
    return log_mel


def run_grafn_on_corpus(
    grafn: GRAFNNormalizer,
    in_dir: str,
    out_dir: str,
    gender_map: Dict[str, str],
    asr_feature_fn=example_asr_feature_fn,
):
    """
    Example "driver" that:
        1. Walks a directory of wav files.
        2. Applies GRAFN normalization using per-file gender labels.
        3. Extracts features and (optionally) saves normalized audio.

    Parameters
    ----------
    grafn : GRAFNNormalizer
        Fitted GRAFN normalizer.
    in_dir : str
        Directory with input .wav files.
    out_dir : str
        Directory where normalized .wav files will be written.
    gender_map : dict
        Maps filename (or stem) -> "male"/"female".
    asr_feature_fn : callable
        Function that takes (waveform, sr) -> features for ASR.
    """
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(".wav"):
            continue

        path = os.path.join(in_dir, fname)
        stem = os.path.splitext(fname)[0]
        gender = gender_map.get(stem, None)

        # 1) Apply GRAFN
        y_tilde = grafn.transform_file(path, gender=gender)

        # 2) (Optional) Save normalized waveform
        out_wav = os.path.join(out_dir, fname)
        sf.write(out_wav, y_tilde, grafn.sr)

        # 3) Extract features for ASR
        feats = asr_feature_fn(y_tilde, grafn.sr)

        # Here you would call your ASR model, e.g.:
        # asr_logits = asr_model(feats)
        # decoded_text = decode(asr_logits)
        #
        # For now we just print a shape to show it worked.
        print(f"{fname} -> feats shape: {feats.shape}, gender={gender}")


if __name__ == "__main__":
    """
    Minimal example of how you'd wire this up:

    1. Fit GRAFN on a labeled subset of your corpus.
    2. Run GRAFN+ASR on the full dataset.
    """
    # Example training subset (replace with your real paths + labels)
    train_wavs = [
        "data/train/male_001.wav",
        "data/train/male_002.wav",
        "data/train/female_001.wav",
        "data/train/female_002.wav",
    ]
    train_genders = ["male", "male", "female", "female"]

    grafn = GRAFNNormalizer(sr=16000)
    # Offline training: learns H_male(f) and H_female(f)
    grafn.fit(train_wavs, train_genders)

    # Example gender map for the run directory
    gender_map = {
        "utt_001": "male",
        "utt_002": "female",
        # ...
    }

    run_grafn_on_corpus(
        grafn=grafn,
        in_dir="data/test_raw_wav/",
        out_dir="data/test_grafn_wav/",
        gender_map=gender_map,
        asr_feature_fn=example_asr_feature_fn,
    )
