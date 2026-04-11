from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="NEXORA Turbo Mobile Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
CARD_DIRS = [
    BASE_DIR,  # ✅ รองรับรูปอยู่ root repo
    BASE_DIR / "cards",
    BASE_DIR / "public" / "cards",
    BASE_DIR / "assets" / "cards",
]

CANON_W = 480
CANON_H = 672
TOP_K = 3
SHORTLIST_K = 18

REFERENCE_DB: list[dict[str, Any]] = []


class PredictRequest(BaseModel):
    image: str


# =========================
# HELPERS
# =========================
def load_image_from_base64(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    image_bytes = base64.b64decode(encoded)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def safe_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if image is None or image.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(image, (w, h))


def corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    a_std = float(a_flat.std())
    b_std = float(b_flat.std())

    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0

    score = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return 0.0 if np.isnan(score) else score


# =========================
# MOBILE PREPROCESS
# =========================
def preprocess_card(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    image = cv2.resize(image, (CANON_W, CANON_H))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return image


# =========================
# ROI
# =========================
def crop_right_number_strip(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    strip = image[
        int(h * 0.08):int(h * 0.88),
        int(w * 0.77):int(w * 0.998),
    ]
    strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return safe_resize(strip, (280, 96))


def crop_top_title(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return safe_resize(
        image[int(h * 0.01):int(h * 0.16), int(w * 0.14):int(w * 0.86)],
        (380, 108),
    )


def crop_bottom_stats(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return safe_resize(
        image[int(h * 0.74):int(h * 0.98), int(w * 0.14):int(w * 0.88)],
        (380, 132),
    )


def crop_center_art(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return safe_resize(
        image[int(h * 0.18):int(h * 0.72), int(w * 0.12):int(w * 0.86)],
        (180, 220),
    )


def global_thumb(image: np.ndarray) -> np.ndarray:
    return safe_resize(image, (96, 128))


# =========================
# FEATURE
# =========================
def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray.astype(np.float16) / 255.0


def normalize_binary_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    _, th = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    th = cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)
    return th.astype(np.float16) / 255.0


def blended_text_score(g1, g2, b1, b2) -> float:
    return corr_score(g1, g2) * 0.35 + corr_score(b1, b2) * 0.65


def extract_card_no_from_filename(path: Path) -> str | None:
    m = re.search(r"(\d{3})", path.stem)
    return m.group(1) if m else None


# =========================
# REFERENCE DB
# =========================
def build_reference_db() -> list[dict[str, Any]]:
    refs = []

    card_dir = None
    for d in CARD_DIRS:
        if d.exists():
            jpgs = list(d.glob("*.jpg"))
            if jpgs:
                card_dir = d
                break

    if card_dir is None:
        print("❌ no card directory")
        return refs

    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files.extend(card_dir.glob(ext))

    for path in sorted(files):
        card_no = extract_card_no_from_filename(path)
        if not card_no:
            continue

        ref = cv2.imread(str(path))
        if ref is None:
            continue

        ref = preprocess_card(ref)

        refs.append(
            {
                "cardNo": card_no,
                "strip_g": normalize_gray(crop_right_number_strip(ref)),
                "strip_b": normalize_binary_text(crop_right_number_strip(ref)),
                "title_g": normalize_gray(crop_top_title(ref)),
                "title_b": normalize_binary_text(crop_top_title(ref)),
                "stats_g": normalize_gray(crop_bottom_stats(ref)),
                "stats_b": normalize_binary_text(crop_bottom_stats(ref)),
                "art_g": normalize_gray(crop_center_art(ref)),
                "global_g": normalize_gray(global_thumb(ref)),
            }
        )

    print(f"✅ Loaded reference cards: {len(refs)}")
    return refs


def ensure_reference_db():
    global REFERENCE_DB
    if not REFERENCE_DB:
        REFERENCE_DB = build_reference_db()


# =========================
# PREDICT
# =========================
def build_candidate_features(img):
    return {
        "strip_g": normalize_gray(crop_right_number_strip(img)),
        "strip_b": normalize_binary_text(crop_right_number_strip(img)),
        "title_g": normalize_gray(crop_top_title(img)),
        "title_b": normalize_binary_text(crop_top_title(img)),
        "stats_g": normalize_gray(crop_bottom_stats(img)),
        "stats_b": normalize_binary_text(crop_bottom_stats(img)),
        "art_g": normalize_gray(crop_center_art(img)),
        "global_g": normalize_gray(global_thumb(img)),
    }


def score_full(cand, ref):
    s_strip = blended_text_score(cand["strip_g"], ref["strip_g"], cand["strip_b"], ref["strip_b"])
    s_title = blended_text_score(cand["title_g"], ref["title_g"], cand["title_b"], ref["title_b"])
    s_stats = blended_text_score(cand["stats_g"], ref["stats_g"], cand["stats_b"], ref["stats_b"])
    s_art = corr_score(cand["art_g"], ref["art_g"])
    s_global = corr_score(cand["global_g"], ref["global_g"])

    return (
        s_strip * 0.28
        + s_title * 0.30
        + s_stats * 0.18
        + s_art * 0.14
        + s_global * 0.10
    )


def predict_card(candidate_raw: np.ndarray):
    ensure_reference_db()

    if not REFERENCE_DB:
        return {"cardNo": None, "confidence": 0.0, "top3": []}

    candidate = preprocess_card(candidate_raw)
    cand = build_candidate_features(candidate)

    # ⚡ Stage 1 quick shortlist
    stage1 = []
    for ref in REFERENCE_DB:
        fast = (
            blended_text_score(cand["title_g"], ref["title_g"], cand["title_b"], ref["title_b"]) * 0.5
            + corr_score(cand["art_g"], ref["art_g"]) * 0.3
            + corr_score(cand["global_g"], ref["global_g"]) * 0.2
        )
        stage1.append((ref, fast))

    shortlist = sorted(stage1, key=lambda x: x[1], reverse=True)[:SHORTLIST_K]

    ranked = []
    for ref, _ in shortlist:
        ranked.append((ref["cardNo"], score_full(cand, ref)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top3 = ranked[:TOP_K]

    if not top3:
        return {"cardNo": None, "confidence": 0.0, "top3": []}

    best, best_score = top3[0]
    second = top3[1][1] if len(top3) > 1 else 0
    margin = max(0.0, best_score - second)

    confidence = min(0.99, max(0.65, 0.82 + margin * 1.6))

    return {
        "cardNo": best,
        "confidence": round(confidence, 4),
        "top3": [{"cardNo": c, "score": round(s, 4)} for c, s in top3],
    }


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    ensure_reference_db()
    return {
        "ok": True,
        "message": "NEXORA Turbo Mobile Matcher running",
        "references": len(REFERENCE_DB),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if req.image == "warmup":
            return {"ok": True}

        image = load_image_from_base64(req.image)
        return predict_card(image)

    except Exception as e:
        return {
            "cardNo": None,
            "confidence": 0.0,
            "top3": [],
            "error": str(e),
        }