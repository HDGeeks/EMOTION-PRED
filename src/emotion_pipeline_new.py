"""
emotion_pipeline_new.py
Clean, unified, multi-aspect ABSA-emotion pipeline (single + multi-model).
"""

import os
import time
import warnings
from typing import List, Dict, Union, Optional

import pandas as pd
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
warnings.filterwarnings("ignore")

# ---------------------------------------------
# SETTINGS
# ---------------------------------------------
MODEL_NAMES = [
    "j-hartmann/emotion-english-distilroberta-base",
    "j-hartmann/emotion-english-roberta-large",
    "nateraw/bert-base-uncased-emotion",
    "joeddav/distilbert-base-uncased-go-emotions-student",
    "cardiffnlp/twitter-roberta-base-emotion",
    "mrm8488/t5-base-finetuned-emotion",
    "SamLowe/roberta-base-go_emotions",
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PIPELINE_DEVICE = 0 if torch.backends.mps.is_available() else -1


# ---------------------------------------------
# UNIFIED BATCH PREDICTOR
# ---------------------------------------------
def _predict_batch(model_name: str, texts: List[str]) -> List[str]:
    """Runs the correct batch pipeline depending on architecture."""
    
    # ---- T5 MODE ----
    if "t5" in model_name.lower():
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=PIPELINE_DEVICE
        )
        out = pipe(texts, max_length=32)
        return [o["generated_text"].strip().lower() for o in out]

    # ---- CARDIFFNLP MODE ----
    if "cardiffnlp" in model_name.lower():
        labels = ["anger", "joy", "optimism", "sadness"]
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)

        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        idx = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        return [labels[i] for i in idx]

    # ---- GOEMOTIONS MODE ----
    if "go_emotions" in model_name.lower():
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True,
            device=PIPELINE_DEVICE
        )
        outs = pipe(texts)
        out = []
        for scores in outs:
            best = max(scores, key=lambda x: x["score"])
            out.append(best["label"].lower())
        return out

    # ---- DEFAULT MODE ----
    pipe = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=1,
        device=PIPELINE_DEVICE
    )
    outs = pipe(texts)
    def norm(o):
        if isinstance(o, list):
            o = o[0]
        return o["label"].lower()
    return [norm(o) for o in outs]


# ---------------------------------------------
# MAIN USER-FACING FUNCTION
# ---------------------------------------------
def annotate(df: pd.DataFrame, model_name: Optional[str] = None):
    """
    Main entrypoint:
    - annotate(df) → ALL models
    - annotate(df, "model") → single model
    
    Supports:
    - aspect_term = "service"
    - aspect_term = ["service", "staff"]
    """

    if "sentence" not in df or "aspect_term" not in df:
        raise ValueError("df must have columns: 'sentence', 'aspect_term'")

    # MULTI-MODEL MODE
    if model_name is None:
        out = {}
        print("Running all models...\n")
        for m in MODEL_NAMES:
            print(f"=== {m} ===")
            out[m] = annotate(df, m)
        return out

    # SINGLE MODEL MODE
    results = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        aspect = row["aspect_term"]

        # MULTI-ASPECT PER ROW
        if isinstance(aspect, list):
            texts = [f"[ASPECT] {a} [SENTENCE] {sentence}" for a in aspect]
            preds = _predict_batch(model_name, texts)
            results.append(preds)

        # SINGLE ASPECT PER ROW
        else:
            text = f"[ASPECT] {aspect} [SENTENCE] {sentence}"
            preds = _predict_batch(model_name, [text])[0]
            results.append(preds)

    return results