FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg (audio extraction), OpenCV runtime libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        mediapipe \
        opencv-python-headless \
        faster-whisper \
        xgboost \
        scikit-learn \
        numpy \
        pandas \
        joblib \
        sqlalchemy \
        psycopg2-binary \
        python-dotenv

# Copy entire repo (worker needs ml/, backend/app/models/, data/)
COPY . /app

ENV PYTHONPATH=/app

CMD ["python", "-m", "worker.process_visits"]
