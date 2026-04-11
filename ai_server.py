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

app = FastAPI(title="NEXORA Fast Number Lock")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
CARD_DIRS = [
    BASE_DIR / "public" / "cards",
    BASE_DIR / "cards",
]

CANON_W = 480
CANON_H = 672
REFERENCE_DB: list[dict[str, Any]] = []


class PredictRequest(BaseModel):
    image: str


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


def enhance_mobile_capture(image: np.ndarray) -> np.ndarray:
    image = cv2.GaussianBlur(image, (3, 3), 0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    kernel = np.array(
        [[0, -1, 0], [-1, 4.8, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    return cv2.filter2D(image, -1, kernel)


def preprocess_card(image: np.ndarray) -> np.ndarray:
    image = enhance_mobile_capture(image)
    return cv2.resize(image, (CANON_W, CANON_H))


def crop_right_number_strip(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    strip = image[int(h * 0.10):int(h * 0.84), int(w * 0.82):int(w * 0.99)]
    strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return safe_resize(strip, (240, 76))


def crop_top_title(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    title = image[int(h * 0.015):int(h * 0.145), int(w * 0.18):int(w * 0.82)]
    return safe_resize(title, (360, 96))


def crop_bottom_stats(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    stats = image[int(h * 0.76):int(h * 0.97), int(w * 0.18):int(w * 0.86)]
    return safe_resize(stats, (360, 120))


def global_thumb(image: np.ndarray) -> np.ndarray:
    return safe_resize(image, (96, 128))


def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) / 255.0


def normalize_binary_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    _, th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return th.astype(np.float32) / 255.0


def corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    a_std = float(a_flat.std())
    b_std = float(b_flat.std())

    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0

    score = float(np.corrcoef(a_flat, b_flat)[0, 1])
    if np.isnan(score):
        return 0.0
    return score


def blended_score(gray_a: np.ndarray, gray_b: np.ndarray, bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    s_gray = corr_score(gray_a, gray_b)
    s_bin = corr_score(bin_a, bin_b)
    return (s_gray * 0.35) + (s_bin * 0.65)


def extract_card_no_from_filename(path: Path) -> str | None:
    m = re.search(r"(\d{3})", path.stem)
    return m.group(1) if m else None


def build_reference_db() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []

    card_dir = None
    for d in CARD_DIRS:
        if d.exists():
            card_dir = d
            break

    if card_dir is None:
        print("❌ ไม่เจอโฟลเดอร์การ์ด")
        return refs

    files: list[Path] = []
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

        strip = crop_right_number_strip(ref)
        title = crop_top_title(ref)
        stats = crop_bottom_stats(ref)
        global_img = global_thumb(ref)

        refs.append(
            {
                "cardNo": card_no,
                "strip_g": normalize_gray(strip),
                "strip_b": normalize_binary_text(strip),
                "title_g": normalize_gray(title),
                "stats_g": normalize_gray(stats),
                "global_g": normalize_gray(global_img),
            }
        )

    print(f"✅ Loaded reference cards: {len(refs)}")
    return refs


def ensure_reference_db() -> None:
    global REFERENCE_DB
    if not REFERENCE_DB:
        REFERENCE_DB = build_reference_db()


def read_card_number_from_strip(strip_img: np.ndarray) -> str | None:
    ensure_reference_db()

    cand_gray = normalize_gray(strip_img)
    cand_bin = normalize_binary_text(strip_img)

    best_card = None
    best_score = -999.0

    for ref in REFERENCE_DB:
        score = blended_score(
            cand_gray,
            ref["strip_g"],
            cand_bin,
            ref["strip_b"],
        )

        if score > best_score:
            best_score = score
            best_card = ref["cardNo"]

    if best_score < 0.46:
        return None

    return best_card


def fallback_match(candidate: np.ndarray) -> tuple[str | None, float]:
    title = normalize_gray(crop_top_title(candidate))
    stats = normalize_gray(crop_bottom_stats(candidate))
    global_img = normalize_gray(global_thumb(candidate))

    best_card = None
    best_score = -999.0

    for ref in REFERENCE_DB:
        s_title = corr_score(title, ref["title_g"])
        s_stats = corr_score(stats, ref["stats_g"])
        s_global = corr_score(global_img, ref["global_g"])

        score = (s_title * 0.34) + (s_stats * 0.33) + (s_global * 0.33)

        if score > best_score:
            best_score = score
            best_card = ref["cardNo"]

    return best_card, best_score


def predict_card(candidate_raw: np.ndarray) -> dict[str, Any]:
    ensure_reference_db()

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

    for variant in variants:
        strip = crop_right_number_strip(variant)
        direct_card_no = read_card_number_from_strip(strip)

        if direct_card_no:
            return {
                "cardNo": direct_card_no,
                "confidence": 0.97,
                "top3": [{"cardNo": direct_card_no, "score": 0.97}],
            }

    best_card, best_score = fallback_match(candidate)

    if not best_card:
        return {
            "cardNo": None,
            "confidence": 0.0,
            "top3": [],
        }

    confidence = min(0.92, max(0.60, 0.75 + best_score))

    return {
        "cardNo": best_card,
        "confidence": round(confidence, 4),
        "top3": [{"cardNo": best_card, "score": round(best_score, 4)}],
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "message": "NEXORA Fast Number Lock running",
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