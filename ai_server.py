# ai_server.py
# 🚀 NEXORA Stable AI v3
# Edge Number Boost without vector dimension break

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import json

app = FastAPI()

with open("card-vectors.json", "r", encoding="utf-8") as f:
    CARD_VECTORS = json.load(f)


class ScanRequest(BaseModel):
    image: str


def decode_base64_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_main_signature(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 128))
    return gray.astype(np.float32).flatten() / 255.0


def extract_edge_score(img: np.ndarray):
    """
    🚀 ใช้แค่คะแนนเสริมจากเลขขอบ
    ไม่เปลี่ยนขนาด vector เดิม
    """
    h, w = img.shape[:2]

    top_left = img[
        0 : int(h * 0.18),
        0 : int(w * 0.35),
    ]

    gray = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(
        gray,
        alpha=1.9,
        beta=10,
    )

    # วัดความเด่นของตัวเลขแบบง่าย
    edge_strength = float(np.std(gray) / 255.0)

    return min(edge_strength * 0.03, 0.03)


@app.post("/predict")
def predict(req: ScanRequest):
    try:
        if req.image == "warmup":
            return {"ok": True}

        img = decode_base64_image(req.image)

        if img is None:
            return {
                "cardNo": None,
                "confidence": 0,
            }

        query_sig = build_main_signature(img)
        edge_boost = extract_edge_score(img)

        best_card = None
        best_score = -1

        for item in CARD_VECTORS:
            vec = np.array(
                item["vector"],
                dtype=np.float32,
            )

            score = cosine_similarity(
                query_sig,
                vec,
            )

            # 🚀 boost เลขขอบแบบปลอดภัย
            score += edge_boost

            card_no = str(
                item.get("cardNo", "")
            ).zfill(3)

            if card_no.isdigit():
                n = int(card_no)

                if 1 <= n <= 293:
                    score += 0.01

            if score > best_score:
                best_score = score
                best_card = item

        return {
            "cardNo": (
                best_card.get("cardNo")
                if best_card
                else None
            ),
            "confidence": round(
                float(best_score),
                4,
            ),
        }

    except Exception as e:
        return {
            "cardNo": None,
            "confidence": 0,
            "error": str(e),
        }