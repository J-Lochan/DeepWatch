<div align="center">
  <h1>🚨 DeepWatch — AI Video Violence & Anomaly Detection</h1>
  <p><strong>Spatiotemporal 3D Convolutional Neural Network for Real-Time Surveillance Video Auditing</strong></p>

  <p>
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/TorchVision-R3D__18-orange?style=for-the-badge" alt="TorchVision" />
    <img src="https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License" />
  </p>
</div>

---

## 📌 Project Overview

**DeepWatch** is an automated video intelligence solution designed to inspect surveillance footage, CCTV clips, and recorded video streams to detect violent altercations and physical anomalies. 

Unlike traditional 2D image models that inspect isolated frames, DeepWatch leverages **3D ResNet-18 (`r3d_18`)**, extracting joint spatial and temporal cues over a sliding window of sequential video frames.

### Key Capabilities
- ⏱️ **Temporal Sliding Window Analysis:** Evaluates 16-frame clips with stride 8, computing continuous violence probability trajectories over time.
- 🎯 **Incident Segmentation & Flagging:** Automatically clusters high-confidence frames exceeding customizable sensitivity thresholds into bounded time segments.
- 📸 **Key Evidence Frame Extraction:** Captures peak-intensity anomaly frames per detected incident to assist security personnel in rapid post-incident audits.
- 📊 **Interactive Executive Dashboard:** Built with **Streamlit** and **Altair**, featuring interactive timeline graphs, dual light/dark themes, and real-time video playback.
- 📦 **Automated Incident Export:** Generates downloadable ZIP incident packages containing evidence frame captures, timeline plots, and structured audit logs.

---

## 🏗️ Architecture & Pipeline

```
 Surveillance Video / CCTV Feed (.mp4, .avi)
                    │
                    ▼
       OpenCV Frame Preprocessing
     (Resize to 112x112, 30 FPS Normalization)
                    │
                    ▼
     16-Frame Temporal Sliding Window (Stride = 8)
                    │
                    ▼
         3D ResNet-18 Backbone (`r3d_18`)
     Spatiotemporal Feature Extraction + Linear Head
                    │
                    ▼
       Sigmoid Probability Score P(Violence)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Time-Series Plot      Incident Segmentation
   (Altair Curve)     (Threshold Clustered Windows)
                                │
                                ▼
                       Evidence Extraction &
                     Automated Incident Report
```

---

## 🚀 Quickstart & Setup

### 1. Clone the repository
```bash
git clone https://github.com/J-Lochan/DeepWatch.git
cd DeepWatch
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run src/streamlit.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
DeepWatch/
├── src/
│   └── streamlit.py          # Full Streamlit web UI & inference pipeline
├── outputs/
│   ├── models/               # Model weights (.pt / .pth)
│   ├── metrics/              # Validation evaluation metrics & curves
│   └── reports/              # Exported incident bundles
├── demo_videos/              # Sample test clips for demonstration
├── app.py                    # Quickstart entry wrapper
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Evaluation & Performance Benchmark
Trained and validated on standard surveillance violence benchmarks (e.g., **RWF-2000** dataset), capturing violent vs. non-violent actions across outdoor, indoor, and dynamic camera movements.

---

## 👤 Author
**J. Lochan**
- GitHub: [@J-Lochan](https://github.com/J-Lochan)
- LinkedIn: [Lochan Narayana Rao](https://www.linkedin.com/in/lochan-narayana-rao-janapareddy-a59248293)
- Email: [lochannarayanarao@gmail.com](mailto:lochannarayanarao@gmail.com)
