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

app = FastAPI(title="NEXORA Fast Hybrid Mobile Matcher")

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
REFERENCE_DB: list[dict[str, Any]] = []


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


def enhance_mobile_capture(image: np.ndarray) -> np.ndarray:
    # เร็ว + เก่งในที่มืด
    image = cv2.GaussianBlur(image, (3, 3), 0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5.0, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    image = cv2.filter2D(image, -1, kernel)
    return image


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
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
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
    image = enhance_mobile_capture(image)
    image = warp_card(image)
    image = upright_card(image)
    return image


def crop_right_number_strip(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    strip = image[int(h * 0.10):int(h * 0.84), int(w * 0.82):int(w * 0.99)]
    strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.resize(strip, (240, 76))


def crop_top_title(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    title = image[int(h * 0.015):int(h * 0.145), int(w * 0.18):int(w * 0.82)]
    return cv2.resize(title, (360, 96))


def crop_bottom_stats(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    stats = image[int(h * 0.76):int(h * 0.97), int(w * 0.18):int(w * 0.86)]
    return cv2.resize(stats, (360, 120))


def crop_bottom_left_stats(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    roi = image[int(h * 0.78):int(h * 0.98), int(w * 0.04):int(w * 0.38)]
    return cv2.resize(roi, (180, 100))


def crop_bottom_right_stats(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    roi = image[int(h * 0.78):int(h * 0.98), int(w * 0.62):int(w * 0.96)]
    return cv2.resize(roi, (180, 100))


def global_thumb(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (96, 128))


def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) / 255.0


def corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)

    a_std = a.std()
    b_std = b.std()

    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def extract_card_no_from_filename(path: Path) -> str | None:
    m = re.search(r"(\d{3})", path.stem)
    return m.group(1) if m else None


# =========================
# REFERENCE DB
# =========================
def build_reference_db() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []

    card_dir = None
    for d in CARD_DIRS:
        if d.exists():
            card_dir = d
            break

    if card_dir is None:
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
                "stats": normalize_gray(crop_bottom_stats(ref)),
                "stats_l": normalize_gray(crop_bottom_left_stats(ref)),
                "stats_r": normalize_gray(crop_bottom_right_stats(ref)),
            }
        )

    print(f"✅ Loaded reference cards: {len(refs)}")
    return refs


def build_variant_features(img: np.ndarray) -> dict[str, Any]:
    return {
        "global": normalize_gray(global_thumb(img)),
        "strip": normalize_gray(crop_right_number_strip(img)),
        "title": normalize_gray(crop_top_title(img)),
        "stats": normalize_gray(crop_bottom_stats(img)),
        "stats_l": normalize_gray(crop_bottom_left_stats(img)),
        "stats_r": normalize_gray(crop_bottom_right_stats(img)),
    }


def score_against_reference(vf: dict[str, Any], ref: dict[str, Any]) -> float:
    s_strip = corr_score(vf["strip"], ref["strip"])
    s_title = corr_score(vf["title"], ref["title"])
    s_stats = corr_score(vf["stats"], ref["stats"])
    s_stats_l = corr_score(vf["stats_l"], ref["stats_l"])
    s_stats_r = corr_score(vf["stats_r"], ref["stats_r"])
    s_global = corr_score(vf["global"], ref["global"])

    return float(
        (s_strip * 0.36)
        + (s_title * 0.24)
        + (s_stats * 0.16)
        + (s_stats_l * 0.04)
        + (s_stats_r * 0.04)
        + (s_global * 0.16)
    )


def predict_card(candidate_raw: np.ndarray) -> dict[str, Any]:
    global REFERENCE_DB

    if not REFERENCE_DB:
        REFERENCE_DB = build_reference_db()

    if not REFERENCE_DB:
        return {"cardNo": None, "confidence": 0.0, "top3": []}

    candidate = preprocess_card(candidate_raw)

    variants = [
        candidate,
        cv2.rotate(candidate, cv2.ROTATE_180),
    ]

    variant_features = [build_variant_features(v) for v in variants]

    coarse_scores: list[tuple[dict[str, Any], float]] = []

    for ref in REFERENCE_DB:
        best = -999.0
        for vf in variant_features:
            score = score_against_reference(vf, ref)
            if score > best:
                best = score
        coarse_scores.append((ref, best))

    top_refs = sorted(coarse_scores, key=lambda x: x[1], reverse=True)[:18]

    best_scores: dict[str, float] = {}

    for vf in variant_features:
        for ref, _ in top_refs:
            score = score_against_reference(vf, ref)
            card_no = ref["cardNo"]

            if card_no not in best_scores or score > best_scores[card_no]:
                best_scores[card_no] = score

    ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    top3 = ranked[:TOP_K]

    if not top3:
        return {"cardNo": None, "confidence": 0.0, "top3": []}

    best_card_no, best_score = top3[0]
    second_score = top3[1][1] if len(top3) > 1 else -1.0

    margin = max(0.0, best_score - second_score)
    confidence = min(0.99, max(0.55, 0.82 + margin))

    return {
        "cardNo": best_card_no,
        "confidence": round(confidence, 4),
        "top3": [
            {"cardNo": card_no, "score": round(score, 4)}
            for card_no, score in top3
        ],
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "message": "NEXORA Fast Hybrid Mobile Matcher is running",
        "references": len(REFERENCE_DB),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
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