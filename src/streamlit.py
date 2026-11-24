import os
import sys
import io
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

from torchvision.models.video import r3d_18, R3D_18_Weights


# ----------------------------
# Paths (robust no matter where you run from)
# ----------------------------
ROOT = Path(__file__).resolve().parent.parent  # DeepWatch/
OUTPUTS = ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
METRICS_DIR = OUTPUTS / "metrics"
REPORTS_DIR = OUTPUTS / "reports"
DEMO_DIR = ROOT / "demo_videos"


# ----------------------------
# Streamlit config + styling
# ----------------------------
st.set_page_config(
    page_title="DeepWatch",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """ ... """
CUSTOM_CSS = """
<style>
/* -------- Base (works for both) -------- */
.dw-card {
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
  border: 1px solid rgba(255,255,255,0.10);
}
.dw-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
}
.dw-muted { font-size: 13px; opacity: 0.8; }

/* Status banners */
.dw-ok { border-left: 6px solid #22c55e; }
.dw-bad { border-left: 6px solid #ef4444; }

/* -------- Light theme -------- */
html[data-theme="light"] .stApp {
  background: radial-gradient(circle at 10% 20%, rgba(240,244,255,0.9), rgba(255,255,255,1) 35%);
}
html[data-theme="light"] .dw-card {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(0,0,0,0.08);
  color: #111827;
}
html[data-theme="light"] .dw-muted { color: rgba(17,24,39,0.65); }
html[data-theme="light"] .dw-badge {
  background: rgba(255,255,255,0.80);
  border: 1px solid rgba(0,0,0,0.08);
  color: #111827;
}
html[data-theme="light"] .dw-ok { background: rgba(34,197,94,0.10); }
html[data-theme="light"] .dw-bad { background: rgba(239,68,68,0.10); }

/* -------- Dark theme -------- */
html[data-theme="dark"] .stApp {
  background: radial-gradient(circle at 10% 20%, rgba(30,41,59,0.6), rgba(2,6,23,1) 45%);
}
html[data-theme="dark"] .dw-card {
  background: rgba(17,24,39,0.75);
  border: 1px solid rgba(255,255,255,0.10);
  color: #E5E7EB;
}
html[data-theme="dark"] .dw-muted { color: rgba(229,231,235,0.70); }
html[data-theme="dark"] .dw-badge {
  background: rgba(15,23,42,0.80);
  border: 1px solid rgba(255,255,255,0.12);
  color: #E5E7EB;
}
html[data-theme="dark"] .dw-ok { background: rgba(34,197,94,0.12); }
html[data-theme="dark"] .dw-bad { background: rgba(239,68,68,0.12); }

/* Round images nicely */
img { border-radius: 12px !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------
# Core helpers
# ----------------------------
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model(weight_path: Path, device: str):
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)

    # safer load if supported by your torch version
    try:
        state = torch.load(str(weight_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(weight_path), map_location=device)

    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def read_all_frames(video_path: str, size=112, max_seconds: float | None = None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    max_frames = None
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(max_seconds * fps)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    if len(frames) == 0:
        raise RuntimeError("Could not read frames from video (codec/path issue).")

    return np.array(frames), float(fps)


def make_clip(frames: np.ndarray, start: int, clip_len: int):
    clip = frames[start : start + clip_len].astype(np.float32) / 255.0  # (T,H,W,C)
    clip = np.transpose(clip, (3, 0, 1, 2))  # (C,T,H,W)
    x = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)  # (1,C,T,H,W)
    return x


def sliding_window_probs(model, frames, fps, clip_len=16, stride=8, device="cpu"):
    times, probs, starts = [], [], []

    if len(frames) < clip_len:
        return np.array([0.0]), np.array([0.0]), [0]

    for s in range(0, len(frames) - clip_len + 1, stride):
        x = make_clip(frames, s, clip_len).to(device)
        with torch.no_grad():
            logit = model(x).squeeze()
            prob = torch.sigmoid(logit).item()

        times.append(s / fps)
        probs.append(prob)
        starts.append(s)

    return np.array(times), np.array(probs), starts


def get_segments(times, probs, threshold=0.7):
    segs = []
    on = False
    start_t = 0.0

    for t, p in zip(times, probs):
        if p >= threshold and not on:
            on = True
            start_t = float(t)
        if p < threshold and on:
            on = False
            segs.append((start_t, float(t)))

    if on:
        segs.append((start_t, float(times[-1])))

    return segs


def evidence_frames(frames, starts, times, probs, top_k=6):
    if len(probs) == 0:
        return [], []

    idxs = np.argsort(-probs)[:top_k]
    imgs, caps = [], []
    for i in idxs:
        s = starts[i]
        mid = s + 8  # middle frame of 16-frame window
        if mid >= len(frames):
            continue
        imgs.append(frames[mid])
        caps.append(f"t={times[i]:.2f}s | p={probs[i]:.2f}")
    return imgs, caps


def save_report(times, probs, threshold, evidence_imgs):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = REPORTS_DIR / f"report_{ts}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save timeline plot (matplotlib for consistent image export)
    fig = plt.figure()
    plt.plot(times, probs)
    plt.axhline(threshold, linestyle="--")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Violence probability")
    plt.title("DeepWatch — Violence Timeline")
    plt.tight_layout()

    timeline_path = report_dir / "timeline.png"
    fig.savefig(timeline_path, dpi=200)
    plt.close(fig)

    # Save evidence frames
    ev_dir = report_dir / "evidence_frames"
    ev_dir.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(evidence_imgs, start=1):
        outp = ev_dir / f"evidence_{idx}.jpg"
        cv2.imwrite(str(outp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return report_dir, timeline_path, ev_dir


def zip_folder(folder_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in folder_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(folder_path)))
    buf.seek(0)
    return buf.read()


def file_mtime(path: Path) -> str:
    if not path.exists():
        return "Not found"
    dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt.strftime("%d %b %Y, %I:%M %p")


# ----------------------------
# Header (Product feel)
# ----------------------------
device = get_device()
model_path = MODELS_DIR / "deepwatch_best.pt"

st.markdown(
    f"""
<div class="dw-card">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
    <div>
      <h1 style="margin:0;">DeepWatch</h1>
      <div class="dw-muted">Violence Detection • Timeline • Evidence Frames • Incident Report</div>
    </div>
    <div style="text-align:right;">
      <div class="dw-badge">Device: <b>{device.upper()}</b></div><br/>
      <div class="dw-badge">Model: <b>{model_path.name if model_path.exists() else "Missing"}</b></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="dw-card"><b>🎥 Analyzer</b><div class="dw-muted">Upload video → prediction + timeline + segments + evidence</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="dw-card"><b>🧾 Incident Report</b><div class="dw-muted">Export timeline + evidence frames into a report folder</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="dw-card"><b>📊 Metrics</b><div class="dw-muted">Validation metrics + confusion matrix (RWF-2000 val)</div></div>', unsafe_allow_html=True)

st.write("")

# Load model (cached)
if not model_path.exists():
    st.error("Model not found: outputs/models/deepwatch_best.pt. Train the model first (python src/train.py).")
    st.stop()

model = load_model(model_path, device)


# ----------------------------
# Tabs (no sidebar pop-out)
# ----------------------------
tab_analyzer, tab_metrics, tab_about = st.tabs(["🎥 Analyzer", "📊 Metrics", "ℹ️ About"])


# ----------------------------
# Analyzer Tab
# ----------------------------
with tab_analyzer:
    left, right = st.columns([1.05, 1.5], gap="large")

    with left:
        st.markdown('<div class="dw-card">', unsafe_allow_html=True)
        st.subheader("Input video")

        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        demo_files = []
        if DEMO_DIR.exists():
            demo_files = sorted([p for p in DEMO_DIR.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov"]])

        demo_choice = None
        if demo_files:
            demo_choice = st.selectbox(
                "Or pick a demo clip",
                options=["(None)"] + [p.name for p in demo_files],
                index=0,
            )

        with st.expander("⚙️ Advanced settings", expanded=False):
            threshold = st.slider("Alert threshold", 0.10, 0.95, 0.70, 0.05)
            clip_len = st.selectbox("Clip length (frames)", [16, 24, 32], index=0)
            stride = st.selectbox("Stride (frames)", [4, 8, 12, 16], index=1)
            max_seconds = st.number_input("Max seconds to analyze (0 = full video)", min_value=0, max_value=3600, value=0, step=10)
            show_evidence = st.checkbox("Show evidence frames", value=True)

        st.markdown(
            '<div class="dw-muted">Tip: For huge videos, set “Max seconds to analyze” to 30–60 for faster demo.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="dw-card">', unsafe_allow_html=True)
        st.subheader("Results")

        # Decide input source
        video_path = None
        temp_to_delete = None

        if uploaded is not None:
            suffix = "." + uploaded.name.split(".")[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp_path = tmp.name
            tmp.close()
            video_path = tmp_path
            temp_to_delete = tmp_path

        elif demo_choice and demo_choice != "(None)":
            video_path = str(DEMO_DIR / demo_choice)

        if not video_path:
            st.info("Upload a video (or select a demo clip) to run DeepWatch.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            try:
                st.video(video_path)

                with st.spinner("Analyzing video… extracting frames and running inference"):
                    frames, fps = read_all_frames(
                        video_path,
                        size=112,
                        max_seconds=(None if max_seconds == 0 else float(max_seconds)),
                    )
                    times, probs, starts = sliding_window_probs(
                        model,
                        frames,
                        fps,
                        clip_len=int(clip_len),
                        stride=int(stride),
                        device=device,
                    )

                max_p = float(np.max(probs)) if len(probs) else 0.0
                pred = "VIOLENCE" if max_p >= float(threshold) else "NON-VIOLENCE"

                # Status banner
                if pred == "VIOLENCE":
                    st.markdown(
                        f'<div class="dw-card dw-bad"><h3 style="margin:0;">🚨 ALERT: Violence detected</h3>'
                        f'<div class="dw-muted">Recommendation: Security notification / review flagged segments.</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="dw-card dw-ok"><h3 style="margin:0;">✅ SAFE: No violence detected</h3>'
                        f'<div class="dw-muted">No segments crossed the alert threshold.</div></div>',
                        unsafe_allow_html=True,
                    )

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Prediction", pred)
                k2.metric("Max probability", f"{max_p:.3f}")
                k3.metric("FPS", f"{fps:.1f}")
                k4.metric("Frames analyzed", f"{len(frames)}")

                # Probability bar
                st.caption("Violence confidence (max probability)")
                st.progress(min(max_p, 1.0))

                # Timeline chart (Altair)
                df = pd.DataFrame({"time_s": times, "prob": probs})
                line = (
                    alt.Chart(df)
                    .mark_line()
                    .encode(x=alt.X("time_s:Q", title="Time (seconds)"),
                            y=alt.Y("prob:Q", title="Violence probability", scale=alt.Scale(domain=[0, 1])))
                )
                rule = alt.Chart(pd.DataFrame({"y": [float(threshold)]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
                st.altair_chart((line + rule).properties(height=280), use_container_width=True)

                # Segments table
                segs = get_segments(times, probs, float(threshold))
                st.subheader("Flagged segments")
                if segs:
                    rows = []
                    for s, e in segs:
                        rows.append({"Start (s)": round(s, 2), "End (s)": round(e, 2), "Duration (s)": round(e - s, 2)})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("✅ No segments crossed the threshold.")

                # Evidence gallery
                if show_evidence:
                    st.subheader("Evidence frames (top windows)")
                    imgs, caps = evidence_frames(frames, starts, times, probs, top_k=6)
                    if imgs:
                        cols = st.columns(3)
                        for i, (img, cap) in enumerate(zip(imgs, caps)):
                            with cols[i % 3]:
                                st.image(img, caption=cap, use_container_width=True)
                    else:
                        st.write("No evidence frames available.")

                # Export report
                st.subheader("Export incident report")
                if st.button("📦 Generate report (timeline + evidence)"):
                    imgs, caps = evidence_frames(frames, starts, times, probs, top_k=6)
                    report_dir, timeline_path, ev_dir = save_report(times, probs, float(threshold), imgs)

                    st.success(f"Saved report to: {report_dir}")
                    zip_bytes = zip_folder(report_dir)
                    st.download_button(
                        "⬇️ Download report as ZIP",
                        data=zip_bytes,
                        file_name=f"{report_dir.name}.zip",
                        mime="application/zip",
                    )

            except Exception as e:
                st.error(f"Video processing failed: {e}")

            finally:
                if temp_to_delete and os.path.exists(temp_to_delete):
                    try:
                        os.unlink(temp_to_delete)
                    except Exception:
                        pass

            st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Metrics Tab (Option A — Validation-set metrics)
# ----------------------------
with tab_metrics:
    st.markdown('<div class="dw-card">', unsafe_allow_html=True)
    st.subheader("Validation Metrics (RWF-2000 val)")

    metrics_txt = METRICS_DIR / "metrics.txt"
    cm_img = METRICS_DIR / "confusion_matrix.png"

    cA, cB = st.columns([1.2, 1.0], gap="large")
    with cA:
        st.markdown("**What this page shows**")
        st.write(
            "These metrics are computed on the **fixed validation split** (`data/rwf2000/val`). "
            "So they won’t change when you upload different demo videos. They change only if you retrain the model "
            "and re-run evaluation."
        )
        st.markdown(
            f"""
- Model last modified: **{file_mtime(model_path)}**
- Metrics last modified: **{file_mtime(metrics_txt)}**
""".strip()
        )

        st.markdown("**How to generate/update metrics**")
        st.code(
            r"cd C:\Users\locha\DeepWatch\src" "\n"
            r"python eval_metrics.py",
            language="bat",
        )

    with cB:
        if metrics_txt.exists():
            st.markdown("**Latest metrics**")
            st.text(metrics_txt.read_text(encoding="utf-8", errors="ignore"))
        else:
            st.warning("metrics.txt not found. Run: python src/eval_metrics.py")

    st.write("")
    st.markdown("### Confusion Matrix")
    if cm_img.exists():
        st.image(str(cm_img), use_container_width=True)
    else:
        st.warning("confusion_matrix.png not found. Run: python src/eval_metrics.py")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# About Tab
# ----------------------------
with tab_about:
    st.markdown('<div class="dw-card">', unsafe_allow_html=True)
    st.subheader("About DeepWatch")

    st.markdown(
        """
**DeepWatch workflow**
1. Decode video → frames (OpenCV)  
2. Resize frames to 112×112  
3. Create clips of 16 frames  
4. Run 3D CNN (R3D-18) → probability  
5. Sliding-window inference → timeline  
6. Threshold → flagged segments  
7. Top windows → evidence frames  
8. Export report → timeline + evidence  

**Why this is impressive**
- Not just “violence / non-violence” classification  
- Also gives *when it happened* and *visual evidence*  
- Has a clean UI + exportable incident report + validation metrics  
"""
    )

    st.markdown("**Run project (quick)**")
    st.code(
        r"cd C:\Users\locha\DeepWatch" "\n"
        r"streamlit run app\streamlit_app.py",
        language="bat",
    )
    st.markdown("</div>", unsafe_allow_html=True)


