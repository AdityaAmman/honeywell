# 🔒 AI Surveillance Dashboard

A real-time security monitoring system built with Streamlit. Supports live webcam feeds, uploaded video files, and a simulation mode for demos.

---

## Project structure

```
honeywell/
├── app.py                  # Entry point — run this
├── requirements.txt
├── packages.txt            # System deps for Streamlit Cloud
├── .streamlit/
│   └── config.toml
├── samples/
│   └── videoplayback.mp4   # Sample video for testing
└── app/
    ├── core/
    │   ├── camera.py       # Camera init / release / read
    │   ├── detection.py    # Frame analysis & event generation
    │   ├── events.py       # Event storage & stats
    │   ├── session.py      # Session-state defaults
    │   └── video.py        # Uploaded-video processing pipeline
    └── ui/
        ├── alerts.py       # Live alert cards & metrics row
        ├── analytics.py    # Plotly charts & debug panel
        ├── dashboard.py    # Main layout orchestrator
        ├── sidebar.py      # Config sidebar
        └── styles.py       # Custom CSS
```

---

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

---

## Deployment

### ✅ Streamlit Cloud (recommended)

To deploy on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and `packages.txt` automatically.

App will be live at `https://<your-username>-<repo>-app-<hash>.streamlit.app`.

### Alternative: Railway / Render / Fly.io

Any platform that runs a Docker container or a Python process works. Point it at:

```
streamlit run app.py --server.port $PORT --server.headless true
```

---

## Features

| Feature | Details |
|---|---|
| **Live webcam** | Heuristic frame analysis; optional YOLO upgrade |
| **Video upload** | MP4, AVI, MOV, MKV |
| **Simulation mode** | No camera needed; good for demos |
| **Event types** | Weapon · Fight · Theft · Unattended object |
| **Analytics** | Timeline, severity pie, source bar, confidence histogram |
| **Export** | CSV download of the event log |
| **Debug panel** | Per-frame motion / threat-score metrics |

---

## Configuration

All detection knobs are in the sidebar at runtime:

- **Confidence threshold** — minimum score to log an event (default 0.75)
- **Detection sensitivity** — multiplier on event probability (default 1.2×)
- **Event cooldown** — minimum seconds between same-type events (default 5 s)

Permanent defaults live in `app/core/session.py`.
