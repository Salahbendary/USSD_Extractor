"""
╔══════════════════════════════════════════════════════════════════╗
║          USSD / ServiceMode Screenshot  —  KPI Extractor        ║
║          OCR Engine : PaddleOCR  (Tesseract fallback)           ║
║          Version    : 3.0  —  Offline-ready                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Screen types supported:                                         ║
║   1. iPhone FTM / Dashboard  (light & dark)                      ║
║   2. NetMonster app          (multi-operator → Vodafone only)    ║
║   3. Android ServiceMode     (*#0011# USSD screens)              ║
║   4. Camera-photographed versions of the above                   ║
║                                                                  ║
║  Radio technologies handled:                                     ║
║   LTE  → PCI, RSRP, RSRQ, SINR, TAC, PLMN                       ║
║   3G   → PSC→PCI col, RSCP→RSRP col, EcNo→RSRQ col, LAC→TAC col ║
║   2G   → BCCH→PCI col, LAC→TAC col                               ║
║                                                                  ║
║  Output: CSV  image_name,PLMN,PCI,RSRP,RSRQ,SINR,TAC,Technology ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python extractor.py --input ./images --output results.csv
  python extractor.py -i ./images -o results.csv --debug

Offline model setup:
  Run  python setup_offline_models.py  ONCE (needs internet).
  After that all processing works fully offline.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
LOG_FORMAT = "%(levelname)-8s │ %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("kpi_extractor")


# ──────────────────────────────────────────────────────────────
# OCR engine initialisation  (PaddleOCR → Tesseract fallback)
# ──────────────────────────────────────────────────────────────

_paddle_ocr = None          # lazy singleton
_use_paddle = False         # set to True once PaddleOCR loads OK

def _init_paddle(model_dir: str | None = None):
    """
    Initialise PaddleOCR once.
    model_dir : path to offline models folder (optional).
                If None, PaddleOCR looks for models in its default
                cache (~/.paddleocr/) — pre-downloaded by setup script.
    """
    global _paddle_ocr, _use_paddle
    if _paddle_ocr is not None:
        return  # already initialised

    try:
        from paddleocr import PaddleOCR

        # Build kwargs depending on whether an explicit model dir was given
        kwargs = dict(
            use_angle_cls=True,   # auto-correct 180° flipped text
            lang="en",
            show_log=False,
            use_gpu=False,        # CPU-only for broadest compatibility
        )

        if model_dir:
            model_dir = Path(model_dir)
            det_path = model_dir / "det"
            rec_path = model_dir / "rec"
            cls_path = model_dir / "cls"
            if det_path.exists():
                kwargs["det_model_dir"] = str(det_path)
            if rec_path.exists():
                kwargs["rec_model_dir"] = str(rec_path)
            if cls_path.exists():
                kwargs["cls_model_dir"] = str(cls_path)

        _paddle_ocr = PaddleOCR(**kwargs)
        _use_paddle = True
        log.info("✅ PaddleOCR loaded successfully.")

    except ImportError:
        log.warning("⚠  PaddleOCR not installed — falling back to Tesseract.")
        _use_paddle = False
    except Exception as exc:
        log.warning(f"⚠  PaddleOCR failed to init ({exc}) — falling back to Tesseract.")
        _use_paddle = False


def ocr_with_paddle(img_bgr: np.ndarray) -> str:
    """Run PaddleOCR on a BGR image; return plain text string."""
    result = _paddle_ocr.ocr(img_bgr, cls=True)
    if not result or result[0] is None:
        return ""
    lines = []
    for block in result:
        if block is None:
            continue
        for item in block:
            # item = [[box_coords], (text, confidence)]
            if item and len(item) >= 2 and item[1]:
                text, conf = item[1]
                if conf > 0.30:          # discard very low-confidence tokens
                    lines.append(text)
    return "\n".join(lines)


def ocr_with_tesseract(img_arr: np.ndarray, config: str) -> str:
    """Run Tesseract on a pre-processed greyscale/binary image."""
    try:
        import pytesseract
        return pytesseract.image_to_string(img_arr, config=config)
    except Exception as exc:
        log.debug(f"Tesseract error: {exc}")
        return ""


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

NETMONSTER_STEMS: set[str] = {
    "1", "4", "5", "6", "12", "15", "17", "18",
    "24", "29", "40", "181"
}

VALID_RANGES: dict[str, tuple[float, float]] = {
    "RSRP": (-140.0, -40.0),
    "RSRQ": (-20.0,  -3.0),
    "SINR": (-10.0,  40.0),
    "PCI":  (0.0,    503.0),
}

TESS_CLEAN  = "--oem 3 --psm 6"
TESS_SPARSE = "--oem 3 --psm 11"

CSV_COLUMNS = [
    "image_name", "PLMN", "PCI",
    "RSRP", "RSRQ", "SINR", "TAC", "Technology"
]


# ══════════════════════════════════════════════════════════════
# 1.  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════

def _upscale(img: np.ndarray, target: int = 1920) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) >= target:
        return img
    scale = target / max(h, w)
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct mild text rotation (≤ 10°) via Hough-line analysis."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray
    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines[:, 0]
        if x2 != x1 and abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 10
    ]
    if not angles:
        return gray
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def build_variants(img_bgr: np.ndarray,
                   is_photo: bool = False) -> list[tuple[np.ndarray, str]]:
    """
    Return list of (preprocessed_image, tess_config) tuples.
    PaddleOCR always receives the colour-upscaled image (best results).
    Tesseract variants cover dark/light/noisy/blurry cases.
    """
    img_bgr = _upscale(img_bgr)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if not is_photo:
        gray = _deskew(gray)

    tess_cfg = TESS_SPARSE if is_photo else TESS_CLEAN
    variants: list[tuple[np.ndarray, str]] = []

    # A – adaptive threshold (light-bg screens)
    blur_a = cv2.GaussianBlur(gray, (3, 3), 0)
    t_a = cv2.adaptiveThreshold(blur_a, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, 11)
    variants.append((t_a, tess_cfg))

    # B – inverted adaptive (dark-bg / NetMonster screens)
    inv_gray = cv2.bitwise_not(gray)
    blur_b   = cv2.GaussianBlur(inv_gray, (3, 3), 0)
    t_b = cv2.adaptiveThreshold(blur_b, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 35, 11)
    variants.append((t_b, tess_cfg))

    # C – CLAHE + Otsu (uneven lighting / photographed)
    clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enh    = clahe.apply(gray)
    _, t_c = cv2.threshold(enh, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append((t_c, tess_cfg))

    # D – sharpened + Otsu (compressed / blurry images)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp  = cv2.filter2D(gray, -1, k)
    _, t_d = cv2.threshold(sharp, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append((t_d, tess_cfg))

    # E – inverted CLAHE (coloured labels on dark bg)
    inv_enh = cv2.bitwise_not(enh)
    _, t_e  = cv2.threshold(inv_enh, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append((t_e, tess_cfg))

    return variants, img_bgr   # also return upscaled colour for PaddleOCR


# ══════════════════════════════════════════════════════════════
# 2.  OCR DISPATCHER
# ══════════════════════════════════════════════════════════════

KEYWORDS_RE = re.compile(
    r"\b(LTE|RSRP|RSRQ|SINR|SNR|PCI|TAC|PLMN|"
    r"ServiceMode|Serving|Dashboard|Vodafone|"
    r"Earfcn|EARFCN|PSC|BCCH|RSCP|eNb|ARFCN)\b",
    re.I
)


def _ocr_score(txt: str) -> int:
    """Score OCR output by keyword hits (weighted) + total character count."""
    kw_hits   = len(KEYWORDS_RE.findall(txt))
    char_count = len(txt.replace("\n", "").replace(" ", ""))
    return kw_hits * 50 + char_count


def _is_dark_image(gray: np.ndarray) -> bool:
    """True when the image has a dark background (NetMonster / USSD dark theme)."""
    h, w = gray.shape
    centre = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return float(centre.mean()) < 100


def _tess_fast(gray: np.ndarray, is_dark: bool) -> str:
    """
    Single-pass Tesseract OCR, choosing the correct polarity
    (normal vs inverted) based on background darkness.
    Returns OCR text.
    """
    src = cv2.bitwise_not(gray) if is_dark else gray
    blur   = cv2.GaussianBlur(src, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, 11)
    return ocr_with_tesseract(thresh, TESS_CLEAN)


def _tess_full(img_bgr: np.ndarray, photo: bool) -> str:
    """
    Full 5-variant Tesseract sweep — used as fallback when the fast
    single-pass misses important keywords.
    """
    _, img_up = build_variants(img_bgr, is_photo=photo)
    variants, _ = build_variants(img_bgr, is_photo=photo)
    best = ""
    for arr, cfg in variants:
        txt = ocr_with_tesseract(arr, cfg)
        if _ocr_score(txt) > _ocr_score(best):
            best = txt
    return best


def run_ocr(img_path: str) -> str:
    """
    Run OCR on `img_path`.

    Strategy (fast-first, full-fallback):
      1. PaddleOCR  — colour + inverted (if available)  [fastest, best]
      2. Tesseract  — smart single pass (dark/light auto-detected)
      3. Tesseract  — full 5-variant sweep if step-2 result has < 3 KW hits
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Upscale once; reuse everywhere
    img_up   = _upscale(img_bgr)
    gray_up  = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    photo    = _is_photo(img_bgr)
    is_dark  = _is_dark_image(gray_up)

    candidates: list[str] = []

    # ── 1. PaddleOCR (primary) ─────────────────────────────
    if _use_paddle:
        for variant_img in (img_up, cv2.bitwise_not(img_up)):
            try:
                txt = ocr_with_paddle(variant_img)
                if txt.strip():
                    candidates.append(txt)
            except Exception as exc:
                log.debug(f"PaddleOCR error: {exc}")

        if candidates:
            best = max(candidates, key=_ocr_score)
            if _ocr_score(best) >= 50:   # at least 1 keyword hit
                return best

    # ── 2. Tesseract fast single-pass ──────────────────────
    txt_fast = _tess_fast(gray_up, is_dark)
    candidates.append(txt_fast)

    # Also try opposite polarity (some images fool the bg detector)
    txt_opp = _tess_fast(gray_up, not is_dark)
    candidates.append(txt_opp)

    best_so_far = max(candidates, key=_ocr_score)

    # If fast pass already found ≥ 3 keyword hits → good enough
    if len(KEYWORDS_RE.findall(best_so_far)) >= 3:
        return best_so_far

    # ── 3. Tesseract full 5-variant fallback ───────────────
    log.debug(f"  Fast pass insufficient ({_ocr_score(best_so_far)}), running full sweep …")
    gray_clean = _deskew(gray_up) if not photo else gray_up
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray_clean)

    extra_variants = [
        (cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], TESS_CLEAN),
        (cv2.threshold(cv2.bitwise_not(enh), 0, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], TESS_CLEAN),
        (cv2.threshold(cv2.filter2D(gray_clean, -1,
                       np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])),
                       0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], TESS_CLEAN),
    ]
    for arr, cfg in extra_variants:
        txt = ocr_with_tesseract(arr, cfg)
        if txt.strip():
            candidates.append(txt)

    return max(candidates, key=_ocr_score) if candidates else ""


# ══════════════════════════════════════════════════════════════
# 3.  PHOTO DETECTION HEURISTIC
# ══════════════════════════════════════════════════════════════

def _is_photo(img_bgr: np.ndarray) -> bool:
    """
    True when image is a camera photo (not a clean screenshot).
    Heuristic: photos have mid-grey borders (phone bezel) + high noise.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    border_mean = float(np.mean([
        gray[:15, :].mean(), gray[-15:, :].mean(),
        gray[:, :15].mean(), gray[:, -15:].mean()
    ]))
    sample  = cv2.resize(gray, (300, 300))
    lap_var = cv2.Laplacian(sample, cv2.CV_64F).var()
    return (25 < border_mean < 215) and (lap_var > 250)


# ══════════════════════════════════════════════════════════════
# 4.  SCREEN-TYPE DETECTION
# ══════════════════════════════════════════════════════════════

def detect_screen_type(text: str, stem: str) -> str:
    """
    Returns: 'netmonster' | 'iphone_dashboard' | 'android_service_mode'
    All regex searches use re.I — never search the uppercased copy with
    lowercase patterns (that was the previous bug).
    """
    t = text.upper()   # for plain `in` membership tests only

    # Hard override from known NetMonster stems
    if stem in NETMONSTER_STEMS:
        return "netmonster"

    # NetMonster: EARFCN + (CI / eNb) columns + nav bar
    nm_content = (
        ("EARFCN" in t or "ARFCN" in t)
        and ("ENB" in t or " CI " in t or "\nCI\n" in t)
        and any(k in t for k in ("GRAPH", "LIVE", "LOG", "MENU"))
    )
    if nm_content:
        return "netmonster"

    # NetMonster neighbour-list rows: "-87/-101/-11"
    if re.search(r"-\d{2,3}\s*/\s*-\d{2,3}\s*/\s*-\d{1,2}", text):
        return "netmonster"

    # Android ServiceMode (*#0011# screens, often photographed)
    if (re.search(r"servicemode",        text, re.I)
            or re.search(r"serving\s+plmn",  text, re.I)
            or re.search(r"earfcn\s*:",      text, re.I)
            or re.search(r"hplmn\s*\(",      text, re.I)
            or re.search(r"basic\s+information", text, re.I)):
        return "android_service_mode"

    # iPhone FTM / Dashboard screens
    if ((re.search(r"\bdashboard\b", text, re.I) or re.search(r"\bftm\b", text, re.I))
            and (re.search(r"carrier\s*:",              text, re.I)
                 or re.search(r"network\s+(capabilities|plmn)", text, re.I))):
        return "iphone_dashboard"

    # Fallback: any screen with "carrier:" + "plmn:" labels
    if re.search(r"carrier\s*:", text, re.I) and re.search(r"plmn\s*:", text, re.I):
        return "iphone_dashboard"

    return "iphone_dashboard"   # safest default


def detect_technology(text: str) -> str:
    """Determine 2G / 3G / LTE from OCR text content."""
    t = text.upper()

    if any(re.search(p, t) for p in
           (r"\bLTE\b", r"\bEARFCN\b", r"\bRSRQ\b",
            r"\bRSRP\b", r"\bSINR\b")):
        return "LTE"

    if any(re.search(p, t) for p in
           (r"\bPSC\b", r"\bEC.?NO\b", r"\bRSCP\b",
            r"\bUMTS\b", r"\bWCDMA\b", r"\b3G\b", r"\bUARFCN\b")):
        return "3G"

    if any(re.search(p, t) for p in
           (r"\bBCCH\b", r"\bGSM\b", r"\b2G\b")):
        return "2G"

    if re.search(r"\bNSA\b", t):
        return "LTE"

    return "LTE"


# ══════════════════════════════════════════════════════════════
# 5.  FIELD VALIDATION
# ══════════════════════════════════════════════════════════════

def validate(key: str, raw) -> str:
    """Validate value against telecom ranges. Returns string or 'N/A'."""
    if raw is None or str(raw).strip() == "":
        return "N/A"
    try:
        v = float(str(raw).strip())
    except ValueError:
        return "N/A"

    if key in VALID_RANGES:
        lo, hi = VALID_RANGES[key]
        if not (lo <= v <= hi):
            log.debug(f"  Validation: {key}={v} out of [{lo},{hi}] → N/A")
            return "N/A"

    return f"{v:.1f}" if v != int(v) else str(int(v))


def normalise_plmn(raw: str) -> str:
    """
    Normalise any PLMN variant to 'MCC-MNC' format.
    Handles: '602 2', '6022', '60202', '602-2', '602-02'
    OCR artefacts like '602- 02' or '6 0 2 2' are also cleaned.
    """
    if not raw:
        return "N/A"
    raw = raw.strip()

    # Strip all spaces and dashes to get pure digits
    digits = re.sub(r"[\s\-]", "", raw)

    # Must be at least 4 digits (MCC=3, MNC=1..3)
    if not re.fullmatch(r"\d{4,6}", digits):
        return raw   # return as-is if not digit-only

    mcc = digits[:3]
    mnc = digits[3:]

    # Pad 1-digit MNC (e.g. '2' → '02')
    if len(mnc) == 1:
        mnc = "0" + mnc

    return f"{mcc}-{mnc}"


# ══════════════════════════════════════════════════════════════
# 6.  TYPE-SPECIFIC PARSERS
# ══════════════════════════════════════════════════════════════

# ─── 6a. NetMonster ───────────────────────────────────────────

def _nm_vodafone_block(text: str) -> str:
    """
    Isolate the FIRST Vodafone LTE block from a NetMonster OCR dump.
    Stops at competitor operator heading or neighbour-cell list.
    """
    lines    = text.splitlines()
    block: list[str] = []
    in_block = False

    for line in lines:
        u = line.upper()

        if "VODAFONE" in u and any(k in u for k in ("LTE", "4G", "IWLAN")):
            if not in_block:
                in_block = True
                block = [line]
                continue
            else:
                break           # second Vodafone heading → end of block

        if in_block:
            if any(op in u for op in ("ETISALAT", "ORANGE", "WE ", "E& ", "E&")):
                break           # competitor heading → stop
            if re.search(r"-\d{2,3}\s*/\s*-\d{2,3}", line):
                break           # neighbour-list row → stop
            if re.search(r"\b(graph|live|log|menu)\b", u) and len(line) < 60:
                break           # nav bar footer → stop
            block.append(line)

    return "\n".join(block) if block else text


def parse_netmonster(text: str) -> dict:
    rec: dict = {"Technology": "LTE", "PLMN": "602-02"}
    blk = _nm_vodafone_block(text)

    # ── PCI  (re.I for OCR variants "Pcl", "PCi", "PC1") ──────────
    m = re.search(r"\bPC[Ii1lL\]]\s+(\d{1,3})\b", blk, re.I)
    if m:
        rec["PCI"] = validate("PCI", m.group(1))

    # ── TAC ───────────────────────────────────────────────────────
    m = re.search(r"\bTAC\s+(\d{3,6})\b", blk, re.I)
    if m:
        rec["TAC"] = m.group(1)

    # ── RSRP ──────────────────────────────────────────────────────
    # Unit suffix is optional — OCR often mangles "dBm" → "dam", "d8m", etc.
    # Pattern: RSRP  <value>  [anything-that-looks-like-dBm]
    m = re.search(
        r"\bRSRP\s*:?\s*(-?\d{2,3}(?:\.\d)?)\s*(?:dBm|d[A-Za-z0-9]{1,2})?",
        blk, re.I
    )
    if m:
        rec["RSRP"] = validate("RSRP", m.group(1))

    # ── RSRQ ──────────────────────────────────────────────────────
    # OCR may render '-11 dB' as '"118', '-11¢B', '-110B', etc.
    # Strategy: grab the numeric token right after "RSRQ", then validate.
    m = re.search(r"\bRSRQ\s*:?\s*([\"'\-]?\d{1,2}(?:\.\d)?)", blk, re.I)
    if m:
        raw_rsrq = m.group(1).lstrip("\"'")  # strip OCR quote artefacts
        if not raw_rsrq.startswith("-"):
            raw_rsrq = "-" + raw_rsrq         # RSRQ is always negative
        rec["RSRQ"] = validate("RSRQ", raw_rsrq)

    # ── SNR / SINR ─────────────────────────────────────────────────
    # OCR may render '3 dB' as '30B', '3dB', '3.0B', etc.
    # Use a looser unit match: value followed by optional [0-9a-zA-Z]{0,3}
    for pat in (
        r"\bSNR\s*:?\s*(-?\d{1,2}(?:\.\d)?)\s*[dD]",
        r"\bSINR\s*:?\s*(-?\d{1,2}(?:\.\d)?)\s*[dD]",
        # fallback: any value clearly labelled SNR/SINR with no unit
        r"\bSNR\s+(-?\d{1,2}(?:\.\d)?)\b",
        r"\bSINR\s+(-?\d{1,2}(?:\.\d)?)\b",
    ):
        m = re.search(pat, blk, re.I)
        if m:
            rec["SINR"] = validate("SINR", m.group(1))
            break

    # ── RTL / Arabic positional fallback ──────────────────────────
    # Some NetMonster layouts (Arabic UI) have values without labels.
    # OCR output: "-105 dBm" on its own line, "22090" as TAC, etc.
    if rec.get("RSRP", "N/A") == "N/A":
        # Look for a standalone negative dBm value in range
        m = re.search(r"(?<!\d)(-1[0-3]\d(?:\.\d)?)\s*(?:dBm|d[A-Za-z]m?)?",
                      blk, re.I)
        if m:
            rec["RSRP"] = validate("RSRP", m.group(1))

    if rec.get("RSRQ", "N/A") == "N/A":
        # Look for a standalone "-N dB" value
        m = re.search(r"(?<!\d)(-\d{1,2}(?:\.\d)?)\s*(?:dB|[¢cC][Bb])",
                      blk, re.I)
        if m:
            rec["RSRQ"] = validate("RSRQ", m.group(1))

    if rec.get("TAC", "N/A") == "N/A":
        # 5-digit number in TAC range (10000–65535)
        for cand in re.findall(r"(?<!\d)(\d{5})(?!\d)", blk):
            if 10000 <= int(cand) <= 65535:
                rec["TAC"] = cand
                break

    if rec.get("PCI", "N/A") == "N/A":
        # 3-digit standalone number in PCI range 0-503
        for cand in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", blk):
            v = validate("PCI", cand)
            if v != "N/A" and int(cand) > 0:
                rec["PCI"] = v
                break

    return rec


# ─── 6b. iPhone Dashboard / FTM ──────────────────────────────

def parse_iphone_dashboard(text: str) -> dict:
    rec: dict = {"Technology": detect_technology(text)}

    # PLMN  "PLMN: 602 2"  /  "Network PLMN: 602 2"
    m = re.search(
        r"(?:Network\s+)?PLMN\s*:?\s*([0-9]{3}[\s\-]+[0-9]{1,2})",
        text, re.I
    )
    if m:
        rec["PLMN"] = normalise_plmn(m.group(1))
    else:
        m2 = re.search(r"(602[\s\-]?0?[12])\b", text)
        if m2:
            rec["PLMN"] = normalise_plmn(m2.group(1))

    # TAC
    m = re.search(r"\bTAC\s*:?\s*(\d{4,6})\b", text, re.I)
    if m:
        rec["TAC"] = m.group(1)

    # PCI — OCR sometimes reads 'I' as 'l', '|', ']', '1'
    m = re.search(r"\bPC[Iil|1\]]\s*:?\s*(\d{1,3})\b", text, re.I)
    if m:
        rec["PCI"] = validate("PCI", m.group(1))

    # RSRP  "-113 dBm"
    m = re.search(r"\bRSRP\s*:?\s*(-?\d{2,3}(?:\.\d)?)\s*dBm?", text, re.I)
    if m:
        rec["RSRP"] = validate("RSRP", m.group(1))

    # RSRQ  "-13 dB"
    m = re.search(r"\bRSRQ\s*:?\s*(-?\d{1,2}(?:\.\d)?)\s*dB", text, re.I)
    if m:
        rec["RSRQ"] = validate("RSRQ", m.group(1))

    # SINR — prefer SINR0/SINRO (primary cell: '0' vs letter 'O' OCR noise)
    # then SINR, then SNR
    for pat in (
        r"\bSINR[0O]\s*:?\s*(-?\d{1,2}(?:\.\d)?)\s*dB",
        r"\bSINR\s*:?\s*(-?\d{1,2}(?:\.\d)?)\s*dB",
        r"\bSNR\s*:?\s*(-?\d{1,2}(?:\.\d)?)\s*dB",
    ):
        m = re.search(pat, text, re.I)
        if m:
            rec["SINR"] = validate("SINR", m.group(1))
            break

    return rec


# ─── 6c. Android ServiceMode / USSD ──────────────────────────

def parse_android_service_mode(text: str) -> dict:
    rec: dict = {"Technology": detect_technology(text)}

    # PLMN  "Serving PLMN(602-02)-LTE"  /  "HPLMN(602-02)"
    m = re.search(
        r"(?:Serving|H)PLMN\s*\(\s*([0-9]{3}[-\s]?[0-9]{1,2})\s*\)",
        text, re.I
    )
    if m:
        rec["PLMN"] = normalise_plmn(m.group(1))
    else:
        m2 = re.search(r"\bPLMN[:\s(]+([0-9]{3}[-\s]?[0-9]{1,2})", text, re.I)
        if m2:
            rec["PLMN"] = normalise_plmn(m2.group(1))

    # TAC  "TAC(11070)"  /  "TAC: 11070"  /  inline in EMM line
    m = re.search(r"\bTAC[\s:(]+(\d{4,6})[)\s,]?", text, re.I)
    if m:
        rec["TAC"] = m.group(1)

    # LAC (2G/3G) → TAC column
    if rec.get("Technology") in ("2G", "3G"):
        m = re.search(r"\bLAC\s*[:(]+\s*(\d{3,6})", text, re.I)
        if m:
            rec["TAC"] = m.group(1)

    # PCI  "Earfcn: 525, PCI: 105"   or  "PCh: 105"  (OCR: h for I)
    m = re.search(r"\bPC[IilhH|1\]]\s*:?\s*(\d{1,3})\b", text, re.I)
    if m:
        rec["PCI"] = validate("PCI", m.group(1))

    # PSC (3G) → PCI column
    if rec.get("Technology") == "3G" and not rec.get("PCI"):
        m = re.search(r"\bPSC\s*:?\s*(\d{1,3})\b", text, re.I)
        if m:
            rec["PCI"] = m.group(1)

    # BCCH (2G) → PCI column
    if rec.get("Technology") == "2G" and not rec.get("PCI"):
        m = re.search(r"\bBCCH\s*:?\s*(\d{1,4})\b", text, re.I)
        if m:
            rec["PCI"] = m.group(1)

    # RSRP — collect all hits (handles "R0 RSRP:", "R0. RSRP:", "RSRP:")
    hits = re.findall(
        r"(?:R0[\.\s]+)?RSRP\s*[;:,\s]\s*(-?\d{2,3}(?:\.\d)?)", text, re.I
    )
    valid_rsrp = [
        float(v) for v in hits
        if VALID_RANGES["RSRP"][0] <= float(v) <= VALID_RANGES["RSRP"][1]
    ]
    if valid_rsrp:
        rec["RSRP"] = validate("RSRP", max(valid_rsrp))

    # RSCP (3G) → RSRP column
    if rec.get("Technology") == "3G" and not rec.get("RSRP"):
        m = re.search(r"\bRSCP\s*:?\s*(-?\d{2,3}(?:\.\d)?)", text, re.I)
        if m:
            rec["RSRP"] = m.group(1)

    # RSRQ  "RSRQ:-5"
    m = re.search(r"\bRSRQ\s*[;:,\s]\s*(-?\d{1,2}(?:\.\d)?)", text, re.I)
    if m:
        rec["RSRQ"] = validate("RSRQ", m.group(1))

    # Ec/No (3G) → RSRQ column
    if rec.get("Technology") == "3G" and not rec.get("RSRQ"):
        m = re.search(r"Ec/?No\s*:?\s*(-?\d{1,2}(?:\.\d)?)", text, re.I)
        if m:
            rec["RSRQ"] = m.group(1)

    # SNR / SINR  "SNR:30"  — also catches OCR artefact "SNP:30"
    for pat in (
        r"\bSINR\s*[;:,\s]\s*(-?\d{1,2}(?:\.\d)?)",
        r"\bSN[RP]\s*[;:,\s]\s*(-?\d{1,2}(?:\.\d)?)",
    ):
        m = re.search(pat, text, re.I)
        if m:
            rec["SINR"] = validate("SINR", m.group(1))
            break

    return rec


# ══════════════════════════════════════════════════════════════
# 7.  SINGLE IMAGE PROCESSOR
# ══════════════════════════════════════════════════════════════

def process_image(img_path: str | Path) -> dict:
    """
    Full pipeline: read → OCR → detect type → parse → validate → return.
    """
    img_path = str(img_path)
    stem     = Path(img_path).stem
    fname    = Path(img_path).name

    record = {col: "N/A" for col in CSV_COLUMNS}
    record["image_name"] = fname

    try:
        ocr_text = run_ocr(img_path)

        if not ocr_text.strip():
            log.warning(f"[{fname}] OCR returned no text.")
            return record

        screen_type = detect_screen_type(ocr_text, stem)

        if screen_type == "netmonster":
            parsed = parse_netmonster(ocr_text)
        elif screen_type == "android_service_mode":
            parsed = parse_android_service_mode(ocr_text)
        else:
            parsed = parse_iphone_dashboard(ocr_text)

        # Merge parsed fields into record
        for key in CSV_COLUMNS[1:]:   # skip image_name
            v = parsed.get(key, "")
            if v and str(v).strip() not in ("", "N/A", "None"):
                record[key] = str(v).strip()

        # ── Debug print ──────────────────────────────────────
        ocr_engine = "PaddleOCR" if _use_paddle else "Tesseract"
        img_bgr    = cv2.imread(img_path)
        photo_flag = _is_photo(img_bgr) if img_bgr is not None else False

        print(f"\n{'─'*65}")
        print(f"  {'📸' if photo_flag else '📱'} {fname}  "
              f"[{screen_type}]  [{ocr_engine}]")
        print(f"  ── OCR text (first 400 chars) ──")
        for ln in ocr_text[:400].splitlines():
            print(f"    {ln}")
        print(f"  ── Extracted ──")
        print(f"    PLMN={record['PLMN']}  Technology={record['Technology']}")
        print(f"    PCI={record['PCI']}  TAC={record['TAC']}")
        print(f"    RSRP={record['RSRP']}  RSRQ={record['RSRQ']}  SINR={record['SINR']}")

    except Exception as exc:
        log.error(f"[{fname}] Unhandled error: {exc}", exc_info=True)

    return record


# ══════════════════════════════════════════════════════════════
# 8.  BATCH RUNNER
# ══════════════════════════════════════════════════════════════

def run_batch(input_dir: str,
              output_csv: str,
              model_dir: str | None = None) -> pd.DataFrame:
    """
    Process every image in input_dir and write results to output_csv.
    model_dir : path to offline PaddleOCR models (optional).
    """
    # Initialise OCR engine
    _init_paddle(model_dir)

    input_path = Path(input_dir)
    if not input_path.exists():
        log.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Gather images, sort numerically
    images: list[Path] = []
    for ext in ("*.jpeg", "*.jpg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(input_path.glob(ext))

    images = sorted(
        set(images),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)
    )

    if not images:
        log.error(f"No images found in {input_dir}")
        sys.exit(1)

    engine_label = "PaddleOCR" if _use_paddle else "Tesseract (fallback)"
    print(f"\n{'═'*65}")
    print(f"  USSD KPI Extractor  |  OCR: {engine_label}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_csv}")
    print(f"  Images : {len(images)}")
    print(f"{'═'*65}")

    records: list[dict] = []
    for i, img_path in enumerate(images, 1):
        log.info(f"[{i:>3}/{len(images)}] {img_path.name}")
        records.append(process_image(img_path))

    df = pd.DataFrame(records, columns=CSV_COLUMNS)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # ── Summary report ──────────────────────────────────────
    total = len(df)
    print(f"\n{'═'*65}")
    print(f"  ✅  Finished!  {total} images  →  {output_csv}")
    print(f"{'─'*65}")
    print(f"  Field fill-rate  (non-N/A values):")
    for col in CSV_COLUMNS[1:]:
        n   = (df[col] != "N/A").sum()
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 5)
        print(f"    {col:<13} {n:>3}/{total}  ({pct:5.1f}%)  {bar}")

    tech_counts = df["Technology"].value_counts()
    print(f"{'─'*65}")
    print(f"  Technology breakdown:")
    for tech, cnt in tech_counts.items():
        print(f"    {tech:<8}  {cnt} images")
    print(f"{'═'*65}\n")

    return df


# ══════════════════════════════════════════════════════════════
# 9.  CLI
# ══════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extractor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input",     "-i", default="./images",
                   help="Folder containing JPEG/PNG images")
    p.add_argument("--output",    "-o", default="kpi_output.csv",
                   help="Output CSV file path")
    p.add_argument("--model-dir", "-m", default=None,
                   help="Path to offline PaddleOCR models folder "
                        "(contains det/ rec/ cls/ sub-folders)")
    p.add_argument("--debug",     "-d", action="store_true",
                   help="Enable verbose debug output")
    return p


def main():
    args = _build_parser().parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)
    run_batch(args.input, args.output, args.model_dir)


if __name__ == "__main__":
    main()
