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

app = FastAPI(title="NEXORA Scan AI")

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
    BASE_DIR / "public" / "cards",
    BASE_DIR / "cards",
]

CANON_W = 480
CANON_H = 672
TOP_K = 3


# =========================
# REQUEST
# =========================
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


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_card(image: np.ndarray) -> np.ndarray:
    original = image.copy()
    h, w = image.shape[:2]

    scale = 1000.0 / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best = None
    image_area = image.shape[0] * image.shape[1]

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.15:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            best = approx.reshape(4, 2).astype("float32")
            break

    if best is not None:
        best = best / scale
        rect = order_points(best)

        dst = np.array(
            [
                [0, 0],
                [CANON_W - 1, 0],
                [CANON_W - 1, CANON_H - 1],
                [0, CANON_H - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(original, matrix, (CANON_W, CANON_H))

    return cv2.resize(original, (CANON_W, CANON_H))


def upright_card(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(image, (CANON_W, CANON_H))


def preprocess_card(image: np.ndarray) -> np.ndarray:
    return upright_card(warp_card(image))


def crop_right_number_strip(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = int(w * 0.82)
    x2 = int(w * 0.99)
    y1 = int(h * 0.12)
    y2 = int(h * 0.82)

    strip = image[y1:y2, x1:x2]
    strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.resize(strip, (220, 70))


def crop_top_title(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = int(w * 0.22)
    x2 = int(w * 0.78)
    y1 = int(h * 0.02)
    y2 = int(h * 0.14)

    title = image[y1:y2, x1:x2]
    return cv2.resize(title, (320, 90))


def global_thumb(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (96, 128))


def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) / 255.0


def corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    a_mean = a_flat.mean()
    b_mean = b_flat.mean()

    a_std = a_flat.std()
    b_std = b_flat.std()

    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0

    return float(
        np.mean(((a_flat - a_mean) / a_std) * ((b_flat - b_mean) / b_std))
    )


def hist_score(a: np.ndarray, b: np.ndarray) -> float:
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [30, 32], [0, 180, 0, 256])

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def extract_card_no_from_filename(path: Path) -> str | None:
    m = re.search(r"(\d{3})", path.stem)
    return m.group(1) if m else None


# =========================
# REFERENCE DB
# =========================
REFERENCE_DB: list[dict[str, Any]] = []


def build_reference_db() -> list[dict[str, Any]]:
    refs = []

    card_dir = None
    for d in CARD_DIRS:
        if d.exists():
            card_dir = d
            break

    if card_dir is None:
        print("❌ ไม่เจอโฟลเดอร์การ์ด")
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
                "global": normalize_gray(global_thumb(ref)),
                "strip": normalize_gray(crop_right_number_strip(ref)),
                "title": normalize_gray(crop_top_title(ref)),
                "color": ref,
            }
        )

    print(f"✅ Loaded reference cards: {len(refs)}")
    return refs


@app.on_event("startup")
def startup_event():
    global REFERENCE_DB
    REFERENCE_DB = build_reference_db()


# =========================
# PREDICT
# =========================
def predict_card(candidate_raw: np.ndarray) -> dict[str, Any]:
    if not REFERENCE_DB:
        return {
            "cardNo": None,
            "confidence": 0.0,
            "top3": [],
            "error": "No reference cards loaded",
        }

    candidate = preprocess_card(candidate_raw)

    variants = [
        candidate,
        cv2.rotate(candidate, cv2.ROTATE_180),
    ]

    variant_features = []
    for var in variants:
        variant_features.append(
            {
                "img": var,
                "global": normalize_gray(global_thumb(var)),
                "strip": normalize_gray(crop_right_number_strip(var)),
                "title": normalize_gray(crop_top_title(var)),
            }
        )

    # 🚀 STAGE 1 PREFILTER
    coarse_scores = []

    for ref in REFERENCE_DB:
        best = -999.0

        for vf in variant_features:
            score = (
                corr_score(vf["strip"], ref["strip"]) * 0.60
                + corr_score(vf["title"], ref["title"]) * 0.28
                + corr_score(vf["global"], ref["global"]) * 0.12
            )
            best = max(best, score)

        coarse_scores.append((ref, best))

    top_refs = sorted(
        coarse_scores,
        key=lambda x: x[1],
        reverse=True
    )[:35]

    # 🎯 STAGE 2 FULL MATCH
    best_scores = {}

    for vf in variant_features:
        for ref, _ in top_refs:
            score = (
                corr_score(vf["strip"], ref["strip"]) * 0.50
                + corr_score(vf["title"], ref["title"]) * 0.26
                + corr_score(vf["global"], ref["global"]) * 0.16
                + hist_score(vf["img"], ref["color"]) * 0.08
            )

            card_no = ref["cardNo"]

            if card_no not in best_scores or score > best_scores[card_no]:
                best_scores[card_no] = score

    ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    top3 = ranked[:TOP_K]

    if not top3:
        return {
            "cardNo": None,
            "confidence": 0.0,
            "top3": [],
        }

    best_card_no, best_score = top3[0]
    second_score = top3[1][1] if len(top3) > 1 else -1.0

    margin = max(0.0, best_score - second_score)
    confidence = min(0.99, max(0.45, 0.82 + margin))

    return {
        "cardNo": best_card_no,
        "confidence": round(confidence, 4),
        "top3": [
            {"cardNo": card_no, "score": round(score, 4)}
            for card_no, score in top3
        ],
    }


@app.get("/")
def root():
    return {
        "ok": True,
        "message": "NEXORA Card Matcher is running",
        "references": len(REFERENCE_DB),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        image = load_image_from_base64(req.image)
        return predict_card(image)
    except Exception as e:
        return {
            "cardNo": None,
            "confidence": 0.0,
            "top3": [],
            "error": str(e),
        }