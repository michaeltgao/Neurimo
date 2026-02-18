FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2.6" \
    "sqlalchemy>=2.0" "psycopg2-binary>=2.9" "alembic>=1.13" "python-multipart>=0.0.9" "PyJWT>=2.8" \
    "numpy>=1.24"

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
