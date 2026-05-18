"""Runtime loader for the SIGHT confusion classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from core.config import Config

log = logging.getLogger("SmartTeacher.ConfusionModel")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ConfusionPrediction:
    confused: bool
    prob: float
    threshold: float
    model_name: str
    max_length: int


class ConfusionModel(nn.Module):
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls)).squeeze(-1)


def _resolve_bundle_path(raw_path: str | Path) -> Path:
    bundle_path = Path(raw_path)
    if bundle_path.is_absolute():
        return bundle_path
    return PROJECT_ROOT / bundle_path


class SIGHTConfusionDetector:
    def __init__(self, bundle_path: Path):
        self.bundle_path = bundle_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.threshold = 0.6
        self.max_length = 96
        self.model_name = "xlm-roberta-base"
        self._ready = False
        self._load()

    def _load(self) -> None:
        if not self.bundle_path.exists():
            log.warning("Confusion model bundle not found at %s", self.bundle_path)
            return

        try:
            bundle = torch.load(self.bundle_path, map_location="cpu", weights_only=False)
            self.threshold = float(bundle.get("threshold", self.threshold))
            if Config.SIGHT_THRESHOLD is not None:
                self.threshold = Config.SIGHT_THRESHOLD
            self.max_length = int(bundle.get("max_length", self.max_length))
            self.model_name = str(bundle.get("model_name", self.model_name))

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = ConfusionModel(self.model_name)
            model.load_state_dict(bundle["model_state"])
            model.to(self.device)
            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self._ready = True
            log.info(
                "Loaded SIGHT confusion model from %s (model=%s, threshold=%.3f, max_length=%s, device=%s)",
                self.bundle_path,
                self.model_name,
                self.threshold,
                self.max_length,
                self.device,
            )
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self._ready = False
            log.warning("Failed to load SIGHT confusion model from %s: %s", self.bundle_path, exc)

    @property
    def ready(self) -> bool:
        return self._ready

    @torch.inference_mode()
    def predict(self, text: str) -> Optional[ConfusionPrediction]:
        if not self.ready or self.model is None or self.tokenizer is None:
            return None

        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return ConfusionPrediction(
                confused=False,
                prob=0.0,
                threshold=self.threshold,
                model_name=self.model_name,
                max_length=self.max_length,
            )

        encoded = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        prob = torch.sigmoid(self.model(input_ids, attention_mask)).item()
        return ConfusionPrediction(
            confused=prob >= self.threshold,
            prob=prob,
            threshold=self.threshold,
            model_name=self.model_name,
            max_length=self.max_length,
        )


@lru_cache(maxsize=1)
def get_confusion_detector(model_path: Optional[str] = None) -> Optional[SIGHTConfusionDetector]:
    raw_path = model_path or Config.CONFUSION_MODEL_PATH
    detector = SIGHTConfusionDetector(_resolve_bundle_path(raw_path))
    return detector if detector.ready else None


def predict_confusion(text: str, model_path: Optional[str] = None, return_feedback: bool = False) -> Optional[ConfusionPrediction | dict]:
    detector = get_confusion_detector(model_path)
    if detector is None:
        if return_feedback:
            return {"confused": False, "prob": 0.0, "feedback": "Modèle de confusion indisponible."}
        return None
    pred = detector.predict(text)
    if not return_feedback:
        return pred
    # Feedback négatif inspiré de pruju-ai/LECTOR
    if pred and pred.confused:
        feedback = "Je détecte une possible confusion ou hésitation dans ta réponse. Peux-tu préciser ta question ou demander une clarification ?"
    else:
        feedback = "Pas de signe de confusion détecté."
    return {"confused": bool(pred.confused) if pred else False, "prob": float(pred.prob) if pred else 0.0, "feedback": feedback}