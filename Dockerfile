# ─────────────────────────────────────────────────────────────
# USSD KPI Extractor — Docker Image
# Base: Python 3.11 slim + Tesseract OCR system package
# PaddleOCR models are downloaded at first run (cached in volume)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1  \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────
COPY . .

# ── Runtime dirs ─────────────────────────────────────────────
RUN mkdir -p uploads static

# ── Port ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Start ────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
