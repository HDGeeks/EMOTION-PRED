"""
emotion_pipeline.py
Single-entry emotion evaluation & annotation pipeline.
"""

import os
import re
import warnings
import pandas as pd
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAMES = [
    "j-hartmann/emotion-english-distilroberta-base",
    "j-hartmann/emotion-english-roberta-large",
    "nateraw/bert-base-uncased-emotion",
    "joeddav/distilbert-base-uncased-go-emotions-student",
    "cardiffnlp/twitter-roberta-base-emotion",
    "mrm8488/t5-base-finetuned-emotion"
]

# ======================================================
#  DEVICE HELPERS
# ======================================================

def get_device():
    """Prefer Apple MPS GPU, otherwise CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_pipeline_device():
    """Transformers pipeline device mapping."""
    return 0 if torch.backends.mps.is_available() else -1


DEVICE = get_device()
PIPELINE_DEVICE = get_pipeline_device()


# ======================================================
#  EMOTION CLASSIFICATION (per model)
# ======================================================

def build_classifier(model_name):
    """
    Unified classifier wrapper for all model architectures.
    Returns: classify(sentence, aspect)
    """
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type
    arch = str(config.architectures)

    print(f"  🔍 Model type: {model_type} | arch={arch}")

    # ---- T5 ----
    if "t5" in model_name.lower():
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=PIPELINE_DEVICE
        )

        def classify(sentence, aspect):
            prompt = f"classify emotion: [ASPECT] {aspect} [SENTENCE] {sentence}"
            return pipe(prompt)[0]["generated_text"].strip().lower()

        return classify

    # ---- GoEmotions ----
    elif "go_emotions" in model_name.lower():
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True,
            device=PIPELINE_DEVICE
        )

        def classify(sentence, aspect):
            text = f"[ASPECT] {aspect} [SENTENCE] {sentence}"
            scores = pipe(text)[0]
            best = max(scores, key=lambda x: x["score"])
            return best["label"].lower()

        return classify

    # ---- CardiffNLP ----
    elif "cardiffnlp" in model_name.lower():
        labels = ["anger", "joy", "optimism", "sadness"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)

        def classify(sentence, aspect):
            text = f"[ASPECT] {aspect} [SENTENCE] {sentence}"
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_id = torch.softmax(logits, dim=1).argmax().item()
            return labels[pred_id]

        return classify

    # ---- Default models ----
    else:
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=PIPELINE_DEVICE,
            top_k=1
        )

        def classify(sentence, aspect):
            text = f"[ASPECT] {aspect} [SENTENCE] {sentence}"
            out = pipe(text)[0]
            if isinstance(out, list):
                out = out[0]
            return out["label"].lower()

        return classify


# ======================================================
#  MODEL EVALUATION
# ======================================================

def evaluate_model(model_name, texts, true_labels, label_names, sample_limit=200):
    classify = build_classifier(model_name)

    preds = []
    for t in texts[:sample_limit]:
        try:
            preds.append(classify(t, ""))
        except Exception:
            preds.append("unknown")

    pred_indices = [label_names.index(p) if p in label_names else -1 for p in preds]
    valid = [i for i, x in enumerate(pred_indices) if x != -1]

    y_true = [true_labels[i] for i in valid]
    y_pred = [pred_indices[i] for i in valid]

    if len(y_true) < 3:
        print("⚠ Not enough valid predictions for evaluation.")
        return pd.DataFrame(), None

    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report).transpose()
    cm = confusion_matrix(y_true, y_pred)

    print("  📊 Macro F1:", df_report.loc["macro avg", "f1-score"])
    return df_report, cm


# ======================================================
#  MAIN ENTRY POINT
# ======================================================

def run_full_emotion_pipeline(
    input_csv,
    model_names=MODEL_NAMES,
    dataset_name="dataset",
    sample_limit=200,
    results_root="/Users/hd/Desktop/EMOTION-PRED/src/results/"
):
    """
    SINGLE ENTRY POINT:
      1) Load dataset
      2) Annotate with all models
      3) Save each annotated CSV
    """

    print("\n🚀 Starting full emotion pipeline\n")

    # Load
    df = pd.read_csv(input_csv, dtype=str)

    if "sentence" not in df or "aspect_term" not in df:
        raise ValueError("CSV must contain: sentence, aspect_term")

    out_dir = os.path.join(results_root, f"emotion_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Saving outputs to: {out_dir}")

    # Loop through models
    for model_name in model_names:
        print(f"\n==============================")
        print(f"🔹 Annotating with: {model_name}")
        print("==============================")

        classify = build_classifier(model_name)

        df_out = df.copy()
        df_out["emotion_auto"] = df_out.apply(
            lambda r: classify(r["sentence"], r["aspect_term"]),
            axis=1
        )

        safe = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
        csv_path = os.path.join(out_dir, f"{safe}_annotated.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"   ✅ Saved → {csv_path}")

    print("\n🎯 Pipeline completed successfully.\n")


__all__ = [
    "run_full_emotion_pipeline",
    "evaluate_model",
    "build_classifier"
]