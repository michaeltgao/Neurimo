# Neurimo

AI-powered behavioral assessment platform. Neurimo uses computer vision and machine learning to analyze video recordings for developmental and behavioral insights.

## Architecture

| Service    | Stack                          | Port  |
|------------|--------------------------------|-------|
| **Frontend** | React 19, TypeScript, Vite   | 5173  |
| **Backend**  | Python, FastAPI, PostgreSQL  | 8000  |
| **Worker**   | Python background processor  | —     |
| **ML**       | MediaPipe, scikit-learn, XGBoost | — |

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Node.js (for local frontend development)
- Python 3.11+ (for local ML/backend development)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/neurimo.git
   cd neurimo
   ```

2. Copy environment files and configure:
   ```bash
   cp .env.example .env
   cp frontend/.env.example frontend/.env
   ```

3. Start all services:
   ```bash
   docker compose up --build
   ```

4. Open http://localhost:5173 in your browser.

## Project Structure

```
neurimo/
├── backend/        # FastAPI REST API + Alembic migrations
├── frontend/       # React SPA (Vite + TypeScript)
├── ml/             # Feature extraction, model training, notebooks
├── worker/         # Background video/feature processing pipeline
├── data/           # Local data directory (gitignored)
└── docker-compose.yml
```

## Development

**Backend:**
```bash
cd backend
pip install -e .
uvicorn app.main:app --reload
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
```
