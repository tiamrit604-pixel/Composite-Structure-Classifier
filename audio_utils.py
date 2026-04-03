"""
audio_utils.py
--------------
Exact replication of the feature extraction pipeline from:
    HW3_Code_-_newdataset_-_Grid_CV_Final.ipynb

Changes vs previous version:
  - Every meta entry now stores its own `segment` array (was only seg_idx==0).
    Required for the Signal Processing viewer that lets the user pick any
    segment from any file and inspect Time-series / PSD / MFCC.
  - `signal` (full raw waveform) is also stored on every entry for the
    full-file waveform preview in the results panel.
  - Added `get_plot_data()` — returns the three plot-ready arrays for a
    segment without re-running the full feature extraction pipeline.
  - Audio loading still uses librosa (identical numerical result to the
    notebook's audioread → int16 → /32768 route, just simpler).
"""

import numpy as np
import librosa
import tempfile
import os
from scipy.signal import find_peaks, periodogram


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_bytes(file_bytes: bytes, filename: str) -> tuple:
    """
    Load audio from raw bytes (Streamlit UploadedFile.read()).
    Returns (signal float32 [-1,1], sample_rate int).
    Supports .m4a .wav .mp3 .ogg .flac via librosa/ffmpeg.
    """
    suffix = os.path.splitext(filename)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        signal, sr = librosa.load(tmp_path, sr=None, mono=True)
    finally:
        os.unlink(tmp_path)
    return signal.astype(np.float32), int(sr)


def load_audio_path(filepath: str) -> tuple:
    """Load audio from a file path (used during local training from disk)."""
    signal, sr = librosa.load(filepath, sr=None, mono=True)
    return signal.astype(np.float32), int(sr)


# ---------------------------------------------------------------------------
# Peak detection & segmentation  — exact notebook logic (Cell 9 / Cell 15)
# ---------------------------------------------------------------------------

def detect_peaks_and_segment(signal: np.ndarray, sr: int) -> list:
    """
    Detect percussion hits in signal and extract one segment per hit.

    Parameters (exact notebook values):
      height threshold : 0.3 × max(|signal|)
      min distance     : 0.5 s
      pre-peak window  : 20 ms
      post-peak window : 200 ms

    Each segment is normalised to [-1, 1].
    Returns list of float32 arrays (variable length ≈ 220 ms × sr samples).
    """
    peaks, _ = find_peaks(
        np.abs(signal),
        height=0.3 * np.max(np.abs(signal)),
        distance=int(0.5 * sr)
    )

    if len(peaks) == 0:
        peaks = np.array([len(signal) // 2])

    pre_samples  = int(0.02 * sr)
    post_samples = int(0.20 * sr)

    segments = []
    for peak in peaks:
        start   = max(0, peak - pre_samples)
        end     = min(len(signal), peak + post_samples)
        seg     = signal[start:end].copy()
        max_val = np.max(np.abs(seg))
        if max_val > 0:
            seg = seg / max_val
        segments.append(seg.astype(np.float32))

    return segments


# ---------------------------------------------------------------------------
# Feature extraction — exact notebook logic (Cell 7)
# ---------------------------------------------------------------------------

def extract_features(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    264-dimensional feature vector:
      [0:200]   log-PSD   — periodogram, nfft=4096, first 200 bins
      [200:232] MFCC mean — 32 coefficients
      [232:264] MFCC std  — 32 coefficients

    Exactly mirrors the notebook's extract_features() in Cell 7.
    """
    x = segment.astype(np.float64)

    # PSD — fs=1.0 normalised (matches notebook)
    f, pxx = periodogram(
        x,
        fs=1.0,
        window='boxcar',
        nfft=4096,
        detrend=False,
        scaling='density',
        return_onesided=True
    )
    psd_features = np.log(pxx[:200] + 1e-10)

    # MFCC
    mfcc_mat  = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=32)
    mfcc_mean = mfcc_mat.mean(axis=1)
    mfcc_std  = mfcc_mat.std(axis=1)

    return np.concatenate([psd_features, mfcc_mean, mfcc_std])


# ---------------------------------------------------------------------------
# Plot-data helper — returns raw arrays for Time / PSD / MFCC plots
# without re-extracting the feature vector
# ---------------------------------------------------------------------------

def get_plot_data(segment: np.ndarray, sr: int) -> dict:
    """
    Return the three plot-ready data structures for a single segment.

    Returns a dict with:
      "time_axis"  : 1-D array of time values in seconds
      "waveform"   : 1-D float32 amplitude array (same as segment)
      "psd_f"      : 1-D normalised frequency axis (first 200 bins)
      "psd_pxx"    : 1-D power array (linear scale, NOT log)
      "mfcc_matrix": 2-D array shape (32, n_frames) — raw MFCC matrix
      "sr"         : sample rate (int)
    """
    x = segment.astype(np.float64)

    # Time axis
    time_axis = np.arange(len(segment)) / sr

    # PSD (keep linear for plotting; log is only for the feature vector)
    f, pxx = periodogram(
        x,
        fs=1.0,
        window='boxcar',
        nfft=4096,
        detrend=False,
        scaling='density',
        return_onesided=True
    )

    # MFCC matrix (32 × n_frames)
    mfcc_matrix = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=32)

    return {
        "time_axis":   time_axis.astype(np.float32),
        "waveform":    segment,
        "psd_f":       f[:200].astype(np.float32),
        "psd_pxx":     pxx[:200].astype(np.float32),
        "mfcc_matrix": mfcc_matrix.astype(np.float32),
        "sr":          sr,
    }


# ---------------------------------------------------------------------------
# High-level: process a list of Streamlit UploadedFile objects → X, y, meta
# ---------------------------------------------------------------------------

def process_uploaded_files(
    uploaded_files,
    has_labels: bool = True
) -> tuple:
    """
    Process Streamlit UploadedFile objects into feature matrix + metadata.

    Labelling rule (same as notebook Cell 9):
      filename ends with 'g.<ext>'  →  Good (label = 1)
      anything else                 →  Bad  (label = 0)
      has_labels=False              →  label = -1 (unknown)

    Returns
    -------
    X    : np.ndarray  shape (N, 264)
    y    : np.ndarray  shape (N,)
    meta : list of dicts, one per segment, each with:
             filename    : str
             label       : int  (0 / 1 / -1)
             segment_idx : int  (0-based index within this file)
             n_segments  : int  (total segments extracted from this file)
             error       : str or None
             signal      : np.ndarray  full raw waveform of the file
             segment     : np.ndarray  this specific extracted segment
             sr          : int
    """
    X, y, meta = [], [], []

    for uf in uploaded_files:
        filename   = uf.name
        file_bytes = uf.read()

        try:
            signal, sr = load_audio_bytes(file_bytes, filename)
        except Exception as e:
            meta.append({
                "filename":    filename,
                "label":       -1,
                "segment_idx": 0,
                "n_segments":  0,
                "error":       str(e),
                "signal":      None,
                "segment":     None,
                "sr":          None,
            })
            continue

        segments   = detect_peaks_and_segment(signal, sr)
        n_segments = len(segments)

        if has_labels:
            base = filename.strip().lower()
            label = 1 if base.endswith(("g.wav", "g.m4a",
                                        "g.mp3", "g.ogg",
                                        "g.flac")) else 0
        else:
            label = -1

        for seg_idx, seg in enumerate(segments):
            feat = extract_features(seg, sr)
            X.append(feat)
            y.append(label)
            meta.append({
                "filename":    filename,
                "label":       label,
                "segment_idx": seg_idx,
                "n_segments":  n_segments,
                "error":       None,
                # Full waveform stored on every entry (needed for signal viewer)
                "signal":      signal,
                # This specific extracted segment
                "segment":     seg,
                "sr":          sr,
            })

    X = np.array(X) if X else np.empty((0, 264))
    y = np.array(y) if y else np.array([])
    return X, y, meta
