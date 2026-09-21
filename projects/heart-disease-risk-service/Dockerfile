FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY data ./data
RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501
