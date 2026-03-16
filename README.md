# USSD KPI Extractor — Web Application

A self-hosted web app that extracts radio KPIs from USSD / ServiceMode screenshots,
displays results in a live dashboard, stores everything in a CSV file, and
automatically syncs the CSV to a GitHub repository.

---

## Features

| Feature | Details |
|---|---|
| **Upload any screenshot** | iPhone FTM, NetMonster, Android ServiceMode, camera photos |
| **Auto OCR** | PaddleOCR (primary) → Tesseract (fallback), fully offline after setup |
| **KPI extraction** | PLMN, PCI, RSRP, RSRQ, SINR, TAC, Technology (LTE/3G/2G) |
| **Signal coloring** | Green / Amber / Red based on telecom thresholds |
| **CSV persistence** | All results appended to `kpi_results.csv` |
| **GitHub sync** | CSV auto-pushed after every upload, manual sync button |
| **Live dashboard** | Sortable, filterable table with instant search |
| **Download CSV** | One-click download anytime |

---

## Quick Start

### 1. Clone / set up

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd ussd-kpi-extractor
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

**Tesseract binary (required as fallback):**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows  (download installer)
# https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Download PaddleOCR models (one time, needs internet)

```bash
python setup_offline_models.py
# Models saved to ./paddle_models/
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your values:
nano .env
```

Required `.env` entries:

```env
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_REPO=your-username/your-repo
GITHUB_CSV_PATH=data/kpi_results.csv
GITHUB_BRANCH=main
PADDLE_MODEL_DIR=./paddle_models
```

> **GitHub token scopes needed:** `repo` (full repository access)
> Create at: https://github.com/settings/tokens/new

### 5. Run the app

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open your browser: **http://localhost:8000**

---

## How It Works

```
User uploads image
       ↓
FastAPI receives file → saves to temp
       ↓
extractor.py:
  1. Detect screen type (NetMonster / iPhone Dashboard / Android ServiceMode)
  2. Preprocess image (upscale, deskew, CLAHE, threshold)
  3. OCR with PaddleOCR (or Tesseract fallback)
  4. Parse KPIs with type-specific regex parser
  5. Validate values against telecom ranges
       ↓
Append row to kpi_results.csv
       ↓
Push updated CSV to GitHub (if configured)
       ↓
Return result JSON → update UI
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/`               | Web UI |
| `POST` | `/upload`         | Upload image → extract KPIs → append to CSV |
| `GET`  | `/results`        | Get all rows as JSON |
| `GET`  | `/download`       | Download kpi_results.csv |
| `POST` | `/sync`           | Force-push CSV to GitHub |
| `DELETE` | `/results/{name}` | Delete a row by image name |
| `GET`  | `/health`         | OCR engine status, row count, GitHub status |

### POST /upload response

```json
{
  "success": true,
  "record": {
    "image_name": "screenshot.jpg",
    "PLMN":       "602-02",
    "PCI":        "470",
    "RSRP":       "-87",
    "RSRQ":       "-11",
    "SINR":       "3",
    "TAC":        "21250",
    "Technology": "LTE"
  },
  "csv_row_count": 42,
  "github_pushed": true
}
```

---

## Project Structure

```
ussd-kpi-extractor/
├── app.py                   ← FastAPI backend
├── extractor.py             ← OCR + KPI parsing engine
├── setup_offline_models.py  ← One-time model downloader
├── templates/
│   └── index.html           ← Web UI (single file, no build step)
├── requirements.txt
├── .env.example             ← Config template
├── .env                     ← Your config (git-ignored)
├── kpi_results.csv          ← Local CSV (auto-created)
├── paddle_models/           ← Offline OCR models
│   ├── det/
│   ├── rec/
│   └── cls/
└── uploads/                 ← Temp upload dir (auto-cleaned)
```

---

## Signal Quality Reference

| KPI | Good | Acceptable | Poor |
|---|---|---|---|
| **RSRP** | ≥ -85 dBm | -85 to -100 | < -100 dBm |
| **RSRQ** | ≥ -9 dB | -9 to -14 | < -14 dB |
| **SINR** | ≥ 10 dB | 0 to 10 | < 0 dB |

---

## GitHub Actions (optional — auto-run on push)

Create `.github/workflows/validate.yml` to validate the CSV on every push:

```yaml
name: Validate KPI CSV
on: [push]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install pandas
      - run: |
          python -c "
          import pandas as pd, sys
          df = pd.read_csv('data/kpi_results.csv')
          print(f'Rows: {len(df)}')
          print(df['Technology'].value_counts().to_string())
          "
```
