"""
app_web.py — SCP Delamination Detector (Web / Streamlit Cloud)
--------------------------------------------------------------
Inference-only. No training needed here.

Two model sources:
  Option A — Pre-trained models: loaded directly from the models/ folder
             that is committed to the GitHub repo alongside this file.
             The four .pkl files (knn, svm, decision_tree, logistic_regression)
             are read from disk — no download, no secrets, no internet call.
  Option B — Upload your own: upload any .pkl exported from a local training run.

Pages: Predict & Test  |  Signal Analysis
"""

import io
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st

from audio_utils import process_uploaded_files, get_plot_data
from ml_utils import predict_files, predict_segments, CLASSIFIER_CONFIGS

# ── Model paths — same folder structure as the local app ─────────────────────
# When deployed, the repo root contains a models/ folder with the .pkl files.
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE, "models")

# Classifier name → filename (matches ml_utils._model_path logic)
REPO_MODELS = {
    "KNN":                 "knn_model.pkl",
    "SVM":                 "svm_model.pkl",
    "Decision Tree":       "decision_tree_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
}

def _repo_model_path(clf_name: str) -> str:
    return os.path.join(MODEL_DIR, REPO_MODELS[clf_name])

def _repo_model_exists(clf_name: str) -> bool:
    return os.path.exists(_repo_model_path(clf_name))

@st.cache_resource(show_spinner="Loading model from repo…")
def load_repo_model(clf_name: str):
    """Load a pre-trained model from the models/ folder in the repo."""
    return joblib.load(_repo_model_path(clf_name))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SCP Delamination Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1b2d 0%,#1a2e4a 100%);}
[data-testid="stSidebar"] *{color:#e8edf5!important;}
[data-testid="stSidebar"] [data-testid="stRadio"]>div{gap:6px;}
[data-testid="stFileUploader"] * {color: black !important;}
[data-testid="stSelectbox"] * {color: black !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label{
    background:rgba(255,255,255,0.06);border-radius:8px;
    padding:8px 14px!important;border:1px solid rgba(255,255,255,0.08);}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(255,255,255,0.13);}
.metric-row{display:flex;gap:14px;margin:16px 0;flex-wrap:wrap;}
.metric-card{flex:1;min-width:130px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;}
.metric-card.good{border-left:4px solid #16a34a;background:#f0fdf4;}
.metric-card.bad{border-left:4px solid #dc2626;background:#fef2f2;}
.metric-card.info{border-left:4px solid #2563eb;background:#eff6ff;}
.metric-card.warn{border-left:4px solid #d97706;background:#fffbeb;}
.metric-label{font-size:12px;color:#64748b;font-weight:500;margin-bottom:4px;}
.metric-value{font-size:26px;font-weight:700;color:#0f172a;}
.metric-sub{font-size:12px;color:#94a3b8;margin-top:2px;}
.page-banner{border-radius:14px;padding:28px 32px;margin-bottom:24px;}
.page-banner h1{margin:0;font-size:24px;font-weight:700;color:white;}
.page-banner p{margin:6px 0 0;font-size:14px;opacity:.82;color:white;}
.section-label{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;
    color:#94a3b8;margin:28px 0 12px;padding-bottom:6px;border-bottom:1px solid #f1f5f9;}
.tip-box{background:#eff6ff;border-left:3px solid #2563eb;border-radius:0 8px 8px 0;
    padding:10px 16px;font-size:13px;color:#1e40af;margin:12px 0;}
.block-container{padding-top:1.5rem!important;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def section(t):
    st.markdown(f'<div class="section-label">{t}</div>', unsafe_allow_html=True)

def tip(t):
    st.markdown(f'<div class="tip-box">💡 {t}</div>', unsafe_allow_html=True)

def metric_row(items):
    html = "".join(
        f'<div class="metric-card {k}"><div class="metric-label">{l}</div>'
        f'<div class="metric-value">{v}</div><div class="metric-sub">{s}</div></div>'
        for l, v, s, k in items
    )
    st.markdown(f'<div class="metric-row">{html}</div>', unsafe_allow_html=True)

def plot_panel(pd_data, plot_type, label):
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#ffffff")
    if plot_type == "Time Series":
        ax.plot(pd_data["time_axis"], pd_data["waveform"], lw=0.8, color="#2563eb")
        ax.fill_between(pd_data["time_axis"], pd_data["waveform"], alpha=0.08, color="#2563eb")
        ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("Amplitude", fontsize=8)
    elif plot_type == "PSD":
        ax.plot(pd_data["psd_f"], pd_data["psd_pxx"], lw=1.0, color="#7c3aed")
        ax.fill_between(pd_data["psd_f"], pd_data["psd_pxx"], alpha=0.1, color="#7c3aed")
        ax.set_xlabel("Norm. Frequency (Hz)", fontsize=8); ax.set_ylabel("Power / Hz", fontsize=8)
        ax.set_xlim(0, 0.2)
    elif plot_type == "MFCC":
        plt.close(fig); fig, ax = plt.subplots(figsize=(5.2, 2.8))
        fig.patch.set_facecolor("#ffffff")
        img = librosa.display.specshow(
            pd_data["mfcc_matrix"], x_axis='time', sr=pd_data["sr"], ax=ax, cmap="magma"
        )
        fig.colorbar(img, ax=ax, format="%+.0f", fraction=0.04)
        ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("MFCC coeff.", fontsize=8)
    ax.set_title(label, fontsize=9, fontweight='600', pad=6)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
    if plot_type != "MFCC":
        ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px;text-align:center;'>
      <div style='font-size:28px;'>🔬</div>
      <div style='font-size:15px;font-weight:700;margin-top:6px;'>SCP Detector</div>
      <div style='font-size:11px;opacity:0.6;margin-top:2px;'>HW3 · Web App</div>
    </div>""", unsafe_allow_html=True)

    # Navigation
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:1px;"
        "opacity:0.5;padding:0 4px 6px;text-transform:uppercase;'>Navigation</div>",
        unsafe_allow_html=True,
    )
    page = st.radio("nav", ["🔍  Predict & Test", "📊  Signal Analysis"],
                    label_visibility="collapsed")

    st.markdown("---")

    # ── Model source ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:1px;"
        "opacity:0.5;padding:0 4px 6px;text-transform:uppercase;'>Model</div>",
        unsafe_allow_html=True,
    )

    model_source = st.radio(
        "model_src",
        ["📂  Use pre-trained model", "⬆️  Upload my own .pkl"],
        label_visibility="collapsed",
    )

    model        = None
    model_label  = ""

    # ── Option A: pre-trained from repo ──────────────────────────────────────
    if model_source == "📂  Use pre-trained model":

        # Show which models are available on disk
        available = [name for name in REPO_MODELS if _repo_model_exists(name)]
        unavailable = [name for name in REPO_MODELS if not _repo_model_exists(name)]

        if not available:
            st.markdown(
                "<div style='background:rgba(220,38,38,0.15);border-radius:6px;"
                "padding:8px 12px;font-size:12px;color:#fca5a5;margin:4px 0 8px;'>"
                "⚠️ No pre-trained models found in <code>models/</code>.<br>"
                "Commit your .pkl files to GitHub first.</div>",
                unsafe_allow_html=True,
            )
        else:
            chosen_clf = st.selectbox(
                "Select classifier",
                available,
                label_visibility="collapsed",
                key="repo_clf_select",
            )
            try:
                model       = load_repo_model(chosen_clf)
                model_label = chosen_clf
                st.markdown(
                    f"<div style='background:rgba(22,163,74,0.15);border-radius:6px;"
                    f"padding:8px 12px;font-size:12px;color:#bbf7d0;margin:4px 0;'>"
                    f"✅ {chosen_clf} loaded from repo</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Failed to load: {e}")

            if unavailable:
                st.markdown(
                    "<div style='font-size:11px;opacity:0.5;margin-top:6px;'>"
                    "Not in repo: " + ", ".join(unavailable) + "</div>",
                    unsafe_allow_html=True,
                )

    # ── Option B: user uploads their own pkl ─────────────────────────────────
    else:
        st.markdown(
            "<div style='font-size:11px;opacity:0.65;padding:0 4px 6px;line-height:1.6;'>"
            "Upload any .pkl from your local <code>models/</code> folder.</div>",
            unsafe_allow_html=True,
        )
        pkl_file = st.file_uploader(
            "Upload .pkl model",
            type=["pkl"],
            label_visibility="collapsed",
            key="custom_pkl",
        )
        if pkl_file:
            try:
                model       = joblib.load(io.BytesIO(pkl_file.read()))
                model_label = pkl_file.name.replace(".pkl", "").replace("_", " ").title()
                st.markdown(
                    f"<div style='background:rgba(22,163,74,0.15);border-radius:6px;"
                    f"padding:8px 12px;font-size:12px;color:#bbf7d0;margin:4px 0;'>"
                    f"✅ {model_label} loaded</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Could not load model: {e}")

    st.markdown("---")

    # ── Audio upload ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:1px;"
        "opacity:0.5;padding:0 4px 6px;text-transform:uppercase;'>Audio Files</div>",
        unsafe_allow_html=True,
    )

    has_labels = st.checkbox(
        "Auto-label by filename",
        value=True,
        help="S{n}g.* → Good (1)   ·   S{n}b.* → Bad (0)",
    )
    audio_files = st.file_uploader(
        "audio",
        type=["m4a", "wav", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="audio_upload",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;opacity:0.55;padding:0 4px;line-height:1.9;'>
    <b>Naming convention</b><br>
    S{n}<b>g</b>.m4a → ✅ Good<br>
    S{n}<b>b</b>.m4a → ❌ Bad<br><br>
    <b>Supported formats</b><br>
    .m4a · .wav · .mp3 · .ogg · .flac
    </div>""", unsafe_allow_html=True)

# ── Process audio when files change ──────────────────────────────────────────
if audio_files:
    file_key = tuple(f.name for f in audio_files)
    if st.session_state.get("_file_key") != file_key:
        with st.spinner(f"Segmenting & extracting features from {len(audio_files)} file(s)…"):
            X, y, meta = process_uploaded_files(audio_files, has_labels=has_labels)
        st.session_state.update({
            "test_X": X, "test_y": y, "test_meta": meta, "_file_key": file_key
        })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT & TEST
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍  Predict & Test":

    st.markdown("""
    <div class="page-banner" style="background:linear-gradient(135deg,#1a3a2e 0%,#16a34a 100%);">
      <h1>🔍 Predict & Test</h1>
      <p>Select a model and upload audio files to classify each cell as
         Good (Bonded) or Bad (Debonded).</p>
    </div>""", unsafe_allow_html=True)

    # Guard: model
    if model is None:
        st.markdown("""
        <div class="tip-box">
          👈 <b>Select a model in the sidebar to begin.</b><br><br>
          <b>Option A — Pre-trained model:</b> The <code>models/</code> The application is already equiped with pre-trained .pkl files. Just pick a classifier from
          the dropdown — it loads instantly, no upload needed.<br><br>
          <b>Option B — Upload your own:</b> Run <code>app.py</code> locally to train,
          then upload any of the four .pkl files from your local <code>models/</code> folder.
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Guard: audio
    if "test_meta" not in st.session_state:
        tip("Upload audio files in the sidebar to continue.")
        st.stop()

    X    = st.session_state["test_X"]
    meta = st.session_state["test_meta"]

    errors = [m for m in meta if m.get("error")]
    if errors:
        with st.expander(f"⚠️ {len(errors)} file(s) failed to process"):
            for e in errors:
                st.markdown(f"- `{e['filename']}` — {e['error']}")

    if X.shape[0] == 0:
        st.error("No valid segments extracted. Check file formats.")
        st.stop()

    n_files = len(set(m["filename"] for m in meta if not m.get("error")))
    section("EXTRACTION SUMMARY")
    metric_row([
        ("Files uploaded",  n_files,       "audio files",  "info"),
        ("Segments found",  X.shape[0],    "total hits",   ""),
        ("Features / seg",  X.shape[1],    "dimensions",   ""),
        ("Classifier",      model_label,   "active model", "info"),
    ])

    with st.spinner("Classifying segments…"):
        preds, probas = predict_segments(model, X, meta)
        results_df    = predict_files(model, X, meta)

    section("RESULTS")
    good_n = int((results_df["Label"] == 1).sum())
    bad_n  = int((results_df["Label"] == 0).sum())
    items  = [
        ("✅ Good (Bonded)",  good_n,          "cells", "good"),
        ("❌ Bad (Debonded)", bad_n,            "cells", "bad"),
        ("Total",             len(results_df), "files", "info"),
    ]
    if has_labels:
        tmap = {
            m["filename"]: m["label"]
            for m in meta if not m.get("error") and m["label"] != -1
        }
        if tmap:
            results_df["True Label"] = results_df["Filename"].map(tmap)
            acc  = (results_df["Label"] == results_df["True Label"]).sum() / len(results_df) * 100
            kind = "good" if acc >= 90 else ("warn" if acc >= 70 else "bad")
            items.append(("Accuracy", f"{acc:.1f}%", "file-level", kind))
    metric_row(items)

    def _sp(val):
        if "Good" in str(val): return "color:#16a34a;font-weight:600"
        if "Bad"  in str(val): return "color:#dc2626;font-weight:600"
        return ""

    show_cols = ["Filename", "Prediction", "Confidence (%)", "Segments", "Good Votes", "Bad Votes"]
    if "True Label" in results_df.columns:
        show_cols.insert(2, "True Label")

    st.dataframe(
        results_df[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence (%)": st.column_config.ProgressColumn(
                "Confidence (%)", min_value=0, max_value=100, format="%.1f"
            )
        },
    )

    dl, _ = st.columns([1, 3])
    dl.download_button(
        "⬇  Download CSV",
        data=results_df.to_csv(index=False).encode(),
        file_name="scp_predictions.csv",
        mime="text/csv",
    )

    with st.expander("View segment-level predictions"):
        rows = [
            {
                "File":          m["filename"],
                "Seg #":         m["segment_idx"],
                "Prediction":    "Good" if preds[i] == 1 else "Bad",
                "Prob Good (%)": round(float(probas[i]) * 100, 1),
                "True Label":    {1: "Good", 0: "Bad", -1: "—"}.get(m["label"], "—"),
            }
            for i, m in enumerate(meta) if not m.get("error")
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    tip("Switch to <b>Signal Analysis</b> to visually inspect individual segments.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIGNAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Signal Analysis":

    st.markdown("""
    <div class="page-banner" style="background:linear-gradient(135deg,#3b1f6e 0%,#7c3aed 100%);">
      <h1>📊 Signal Analysis</h1>
      <p>Pick any two segments and compare Time-series, Power Spectral Density,
         and MFCC representations side-by-side.</p>
    </div>""", unsafe_allow_html=True)

    if "test_meta" not in st.session_state:
        st.info("Upload audio files in the sidebar first, then come back here.")
        st.stop()

    meta       = st.session_state["test_meta"]
    valid_meta = [m for m in meta if not m.get("error") and m.get("segment") is not None]

    if not valid_meta:
        st.error("No valid segments available.")
        st.stop()

    def seg_label(m):
        lbl = {1: "✅ Good", 0: "❌ Bad", -1: "?"}.get(m["label"], "?")
        return f"{m['filename']}  ·  seg {m['segment_idx']}  [{lbl}]"

    options = [seg_label(m) for m in valid_meta]

    section("SELECT SEGMENTS TO COMPARE")
    pl, pr = st.columns(2)

    with pl:
        st.markdown(
            '<div style="background:#eff6ff;border-radius:8px;padding:10px 14px;'
            'border:1px solid #bfdbfe;margin-bottom:8px;">'
            '<span style="font-weight:700;color:#1d4ed8;font-size:13px;">Segment A</span>'
            '</div>', unsafe_allow_html=True
        )
        ca = st.selectbox("A", options, index=0, key="sa", label_visibility="collapsed")
        ia = options.index(ca)

    with pr:
        st.markdown(
            '<div style="background:#f5f3ff;border-radius:8px;padding:10px 14px;'
            'border:1px solid #ddd6fe;margin-bottom:8px;">'
            '<span style="font-weight:700;color:#6d28d9;font-size:13px;">Segment B</span>'
            '</div>', unsafe_allow_html=True
        )
        cb = st.selectbox(
            "B", options, index=min(1, len(options) - 1), key="sb",
            label_visibility="collapsed"
        )
        ib = options.index(cb)

    ma = valid_meta[ia]
    mb = valid_meta[ib]

    section("CHOOSE PLOTS")
    pc = st.columns(3)
    show_time = pc[0].checkbox("⏱  Time Series",            value=True)
    show_psd  = pc[1].checkbox("📈  Power Spectral Density", value=True)
    show_mfcc = pc[2].checkbox("🎛  MFCC Heatmap",           value=True)

    plots = (
        (["Time Series"] if show_time else []) +
        (["PSD"]         if show_psd  else []) +
        (["MFCC"]        if show_mfcc else [])
    )
    if not plots:
        tip("Select at least one plot type above.")
        st.stop()

    pda = get_plot_data(ma["segment"], ma["sr"])
    pdb = get_plot_data(mb["segment"], mb["sr"])

    section("COMPARISON")
    hl, hr = st.columns(2)

    def seg_hdr(m, letter, bg, border, color):
        lbl  = {1: "✅ Good", 0: "❌ Bad", -1: "?"}.get(m["label"], "?")
        name = f"{m['filename']} · seg {m['segment_idx']}"
        return (
            f'<div style="background:{bg};border-radius:8px;padding:10px;'
            f'border:1px solid {border};text-align:center;">'
            f'<b style="color:{color};">{letter}</b>  {name}  '
            f'<span style="color:#64748b;font-size:12px;">{lbl}</span></div>'
        )

    hl.markdown(seg_hdr(ma, "A", "#eff6ff", "#bfdbfe", "#1d4ed8"), unsafe_allow_html=True)
    hr.markdown(seg_hdr(mb, "B", "#f5f3ff", "#ddd6fe", "#6d28d9"), unsafe_allow_html=True)

    icons = {"Time Series": "⏱", "PSD": "📈", "MFCC": "🎛"}
    for plot_type in plots:
        st.markdown(
            f'<div style="font-size:13px;font-weight:600;color:#475569;'
            f'margin:20px 0 8px;">{icons[plot_type]}  {plot_type}</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            fig = plot_panel(pda, plot_type, f"A · {ma['filename']} seg {ma['segment_idx']}")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with col_b:
            fig = plot_panel(pdb, plot_type, f"B · {mb['filename']} seg {mb['segment_idx']}")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    with st.expander("🔎  Full raw waveforms (before segmentation)"):
        for m, ltr in [(ma, "A"), (mb, "B")]:
            if m.get("signal") is None:
                continue
            t = np.arange(len(m["signal"])) / m["sr"]
            fig, ax = plt.subplots(figsize=(10, 2.2))
            fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#f8fafc")
            ax.plot(t, m["signal"], lw=0.5, color="#334155")
            ax.set_title(
                f"Segment {ltr} — {m['filename']} (full recording)",
                fontsize=9, fontweight="600",
            )
            ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, lw=0.3, alpha=0.4); fig.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    section("NUMERICAL SUMMARY")
    rows = []
    for lbl, pd_data, m in [("Segment A", pda, ma), ("Segment B", pdb, mb)]:
        pk = pd_data["psd_f"][np.argmax(pd_data["psd_pxx"])]
        rows.append({
            "Segment":        lbl,
            "File":           m["filename"],
            "Seg #":          m["segment_idx"],
            "Duration (ms)":  round(len(m["segment"]) / m["sr"] * 1000, 1),
            "Peak PSD freq":  round(float(pk), 5),
            "Mean PSD power": f"{pd_data['psd_pxx'].mean():.2e}",
            "MFCC mean":      round(float(pd_data["mfcc_matrix"].mean()), 2),
            "MFCC std":       round(float(pd_data["mfcc_matrix"].std()), 2),
            "True label":     {1: "Good", 0: "Bad", -1: "?"}.get(m["label"], "?"),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Segment"), use_container_width=True)
