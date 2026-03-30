# Neurimo

AI-powered developmental screening platform for clinicians. Neurimo uses computer vision, audio analysis, and machine learning to analyze video recordings of standardized behavioral tasks and produce ASD risk assessments for children aged 12-24 months.

## Features

- **Multi-task video assessment** — Analyze three standardized behavioral tasks: Imitation, Joint Attention, and Free Social Play
- **Computer vision pipeline** — MediaPipe pose and hand landmark tracking with Whisper-based speech/vocalization detection
- **ML risk prediction** — XGBoost ensemble model producing four-level risk stratification (low, moderate, moderate-high, high)
- **Clinical questionnaire** — Structured intake for developmental history, clinical flags, and family history
- **Assisted review** — Frame-by-frame video playback with real-time pose overlays and detected behavioral events
- **Automated explanations** — Rule-based explanation generation linking detected behaviors to risk factors

## Architecture

| Service      | Stack                                    | Port |
|--------------|------------------------------------------|------|
| **Frontend** | React 19, TypeScript, Vite               | 5173 |
| **Backend**  | FastAPI, SQLAlchemy 2.0, Alembic         | 8000 |
| **Database** | PostgreSQL 15                            | 5432 |
| **Worker**   | Background processor (polls every 30s)   | —    |
| **ML**       | MediaPipe, faster-whisper, XGBoost       | —    |

## Data Flow

```
Clinician creates patient record
        |
        v
Creates visit and uploads 3 task videos
(Imitation, Joint Attention, Free Play)
        |
        v
Fills clinical questionnaire
(developmental history, family history, clinical flags)
        |
        v
Worker detects ready visit and begins processing
        |
        +--> MediaPipe pose/hand tracking on each video
        +--> Whisper audio event detection (speech, vocalizations)
        +--> Task-specific event extraction
        |      - Imitation: demo/response detection (clapping, arm raises)
        |      - Joint Attention: pointing detection, gaze following
        |      - Free Play: repetitive motion, engagement, hand-to-face
        +--> Feature extraction (29-36 features per task)
        |
        v
XGBoost ensemble prediction
(probability + risk bucket + explanations)
        |
        v
Clinician views report and assisted review
(risk level, explanations, frame-by-frame video overlays)
```

## Synthetic Dataset

The model was trained and validated on a synthetic dataset:

| Property | Value |
|----------|-------|
| Total synthetic children | 167 |
| Usable after filtering | 107 |
| Age group | 12–24 months |
| ASD label | 54 children |
| NT/Control label | 53 children |
| Videos per child | 3 (one per task type) |
| Validation | 5-fold stratified cross-validation (seed=42) |

Each child has three labeled videos corresponding to the Imitation, Joint Attention, and Free Play tasks. The dataset is balanced at approximately 50/50 ASD vs. neurotypical.

## Model Performance

### Deployed Ensemble

The production model is a simple average of three XGBoost classifiers:

| Metric | Value |
|--------|-------|
| **AUROC (out-of-fold)** | **0.847** |
| Mean fold AUROC | 0.860 |
| Per-fold AUROC | 0.800, 0.851, 0.868, 0.882, 0.901 |
| Ensemble weights | Equal (1/3 each) |

### Individual Model AUROC

| Model | Features | AUROC | Accuracy | F1 |
|-------|----------|-------|----------|----|
| all_xgb | All tasks combined | 0.774 | 68.9% | 0.743 |
| joint_xgb | Joint Attention | 0.755 | 73.0% | 0.773 |
| free_xgb | Free Play | 0.702 | 65.7% | 0.727 |
| imit_xgb | Imitation | 0.583 | 59.0% | 0.696 |

### Risk Stratification

Predictions map to four risk buckets based on probability:

| Bucket | Probability Range |
|--------|-------------------|
| Low | 0–25% |
| Moderate | 26–50% |
| Moderate-High | 51–75% |
| High | 76–100% |

## ML Pipeline

### 1. Perception

- **Pose/hand tracking** — MediaPipe PoseLandmarker (2 poses) + HandLandmarker (4 hands) with EMA/Kalman smoothing
- **Audio analysis** — faster-whisper (base model, int8 quantization, CPU) for speech and vocalization detection

### 2. Task-Specific Event Detection

- **Imitation** — Detects adult demonstrations (clapping, arm raises) and child responses within a 7.5s response window
- **Joint Attention** — Extracts pointing events with stability metrics, measures gaze-following and orienting to name
- **Free Play** — Identifies repetitive motion patterns (hand flapping, body rocking), hand-to-face contact, and engagement levels

### 3. Feature Extraction

29–36 features per task including:
- Pose tracking quality and motion energy
- Eye contact duration and face presence ratio
- Task-specific behavioral metrics (follow-point rate, imitation score, repetitive motion fraction)
- Stillness ratio, engagement duration, hand bilateral ratio

### 4. Ensemble Prediction

Three XGBoost models (all-features, joint attention, free play) averaged with equal weights. Per-fold threshold optimization using F1 score.

## Project Structure

```
neurimo/
├── backend/
│   ├── app/
│   │   ├── api/            # REST endpoints (children, visits, videos, questionnaires, reports)
│   │   ├── models/         # SQLAlchemy models (Child, Visit, Video, Questionnaire, MLPrediction)
│   │   ├── services/       # Business logic (annotations, report building)
│   │   └── main.py         # FastAPI app entry point
│   ├── alembic/            # Database migrations
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/            # Axios API client modules
│   │   ├── components/     # UI components (guided review, forms, video player)
│   │   ├── context/        # React context (auth)
│   │   └── pages/          # Route pages (children, visits, questionnaire, report, assisted review)
│   └── Dockerfile
├── ml/
│   ├── src/
│   │   ├── perception/     # MediaPipe tracking, pointing, audio, imitation, free play detection
│   │   ├── features/       # Feature extraction (common, joint attention, imitation, free play)
│   │   ├── train/          # Model training scripts (XGBoost, ensembles, meta-models)
│   │   ├── inference/      # EnsemblePredictor for production predictions
│   │   └── dataset/        # Label generation, manifest building, data splits
│   ├── models/             # Saved model artifacts
│   └── notebooks/          # Exploration and analysis notebooks
├── worker/
│   ├── feature_pipeline.py # Orchestrates perception → features → guided review
│   ├── process_visits.py   # Polls DB for ready visits, runs pipeline, saves predictions
│   ├── config.py           # Paths and polling configuration
│   └── Dockerfile
├── data/                   # Local data directory (gitignored)
├── docker-compose.yml
├── requirements.txt        # ML dependencies
└── .env.example
```

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/children` | Create a new patient record |
| GET | `/children` | List all patients |
| POST | `/children/{id}/visits` | Create a visit |
| POST | `/visits/{id}/videos?task_type=...` | Upload a task video |
| POST | `/visits/{id}/questionnaire` | Submit clinical questionnaire |
| GET | `/visits/{id}/report` | Get ML prediction report |
| GET | `/videos/{id}/guided-review` | Get event data for assisted review |
| GET | `/videos/{id}/overlay-data` | Get frame-level pose data for video overlay |

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local ML/backend development)
- ffmpeg (required by the worker for audio extraction)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/michaeltgao/Neurimo.git
   cd neurimo
   ```

2. Copy and configure environment files:
   ```bash
   cp .env.example .env
   cp frontend/.env.example frontend/.env
   ```

3. Start all services:
   ```bash
   docker compose up --build
   ```

4. Open http://localhost:5173 in your browser.

The worker will automatically begin polling for visits that are ready for processing (3 uploaded videos + completed questionnaire).

## Development

**Backend:**
```bash
cd backend
pip install -e .
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**ML pipeline:**
```bash
pip install -r requirements.txt
python -m ml.src.train.train_meta_model  # Train ensemble
```

**Worker (standalone):**
```bash
python -m worker.process_visits
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Frontend | React 19, TypeScript, Vite, React Router, Axios |
| Backend | FastAPI, SQLAlchemy 2.0, Pydantic, Alembic, psycopg2 |
| Database | PostgreSQL 15 |
| ML/CV | MediaPipe, faster-whisper, XGBoost, scikit-learn, NumPy, pandas |
| Video | OpenCV, ffmpeg |
| Infra | Docker, Docker Compose |
