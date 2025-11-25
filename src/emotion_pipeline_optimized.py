"""
emotion_pipeline_optimized.py
Optimized emotion evaluation & annotation pipeline with batched inference.
"""

import os
import re
import time
import warnings
from typing import List, Tuple, Optional

import pandas as pd
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from sklearn.metrics import classification_report, confusion_matrix
from transformers.utils import logging

# Silence HF warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ---------------------------------------------------------------------
# PATHS (same logic as before, but this file can also be imported)
# ---------------------------------------------------------------------
try:
    # Running as normal Python script inside src/
    this_file = os.path.abspath(__file__)
    src_root = os.path.dirname(this_file)          # EMOTION-PRED/src
    project_root = os.path.dirname(src_root)       # EMOTION-PRED/
except NameError:
    # Running inside Jupyter (likely src/notebooks or src/)
    cwd = os.getcwd()
    if cwd.endswith("notebooks"):
        src_root = os.path.abspath(os.path.join(cwd, ".."))
        project_root = os.path.dirname(src_root)
    else:
        project_root = cwd
        src_root = os.path.join(project_root, "src")

results_root = os.path.join(src_root, "results")
data_root = os.path.join(src_root, "data")

print(f"Project root: {project_root}")
print(f"Source root:  {src_root}")
print(f"Results root: {results_root}")
print(f"Data root:    {data_root}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

MODEL_NAMES = [
    "j-hartmann/emotion-english-distilroberta-base",
    "j-hartmann/emotion-english-roberta-large",
    "nateraw/bert-base-uncased-emotion",
    "joeddav/distilbert-base-uncased-go-emotions-student",
    "cardiffnlp/twitter-roberta-base-emotion",
    "mrm8488/t5-base-finetuned-emotion",
    "SamLowe/roberta-base-go_emotions",
]

# ---------------------------------------------------------------------
# DEVICE HELPERS
# ---------------------------------------------------------------------


def get_device() -> str:
    """Prefer Apple MPS if available, otherwise CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_pipeline_device() -> int:
    """
    Device index for HF pipelines:
      - 0 for MPS (treated like GPU)
      - -1 for CPU
    """
    return 0 if torch.backends.mps.is_available() else -1


DEVICE = get_device()
PIPELINE_DEVICE = get_pipeline_device()


# ---------------------------------------------------------------------
# BUILD CLASSIFIER (for evaluation / single-step use)
# ---------------------------------------------------------------------


def build_classifier(model_name: str):
    """
    Unified classifier wrapper for all model architectures.
    Returns: classify(sentence, aspect) -> str (emotion label)
    (Used mainly for evaluation; annotation uses batched code below.)
    """
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type
    arch = str(config.architectures)

    print(f"  Model type: {model_type} | arch={arch}")

    # ---- T5 ----
    if "t5" in model_name.lower():
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=PIPELINE_DEVICE,
        )

        def classify(sentence: str, aspect: str) -> str:
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
            device=PIPELINE_DEVICE,
        )

        def classify(sentence: str, aspect: str) -> str:
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

        def classify(sentence: str, aspect: str) -> str:
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
            top_k=1,
        )

        def classify(sentence: str, aspect: str) -> str:
            text = f"[ASPECT] {aspect} [SENTENCE] {sentence}"
            out = pipe(text)[0]
            if isinstance(out, list):
                out = out[0]
            return out["label"].lower()

        return classify


# ---------------------------------------------------------------------
# OPTIMIZED BATCH ANNOTATION
# ---------------------------------------------------------------------


def _build_texts(sentences: List[str], aspects: List[str]) -> List[str]:
    """Build joint input: [ASPECT] ... [SENTENCE] ..."""
    return [f"[ASPECT] {a} [SENTENCE] {s}" for s, a in zip(sentences, aspects)]


def annotate_model(model_name: str, df: pd.DataFrame) -> List[str]:
    """
    Fast batch processing for any model.
    Returns list of emotion labels (one per row in df).
    Assumes df has columns: 'sentence', 'aspect_term'.
    """
    sentences = df["sentence"].astype(str).tolist()
    aspects = df["aspect_term"].astype(str).tolist()

    # ---- T5 branch (text2text generation, batched) ----
    if "t5" in model_name.lower():
        print("  [T5] Using batched text2text-generation...")
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=PIPELINE_DEVICE,
        )

        texts = [
            f"classify emotion: [ASPECT] {a} [SENTENCE] {s}"
            for s, a in zip(sentences, aspects)
        ]

        outputs: List[str] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out = pipe(batch, max_length=32)
            outputs.extend([o["generated_text"].strip().lower() for o in out])

        return outputs

    # ---- CardiffNLP branch (manual batching with PT) ----
    elif "cardiffnlp" in model_name.lower():
        print("  [CardiffNLP] Using manual batched forward pass...")
        labels = ["anger", "joy", "optimism", "sadness"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)

        texts = _build_texts(sentences, aspects)
        outputs: List[str] = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(DEVICE)

            with torch.no_grad():
                logits = model(**enc).logits

            preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            outputs.extend([labels[p] for p in preds])

        return outputs

    # ---- GoEmotions branch (batched, return_all_scores=True) ----
    elif "go_emotions" in model_name.lower():
        print("  [GoEmotions] Using batched text-classification with scores...")
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True,
            device=PIPELINE_DEVICE,
        )

        texts = _build_texts(sentences, aspects)
        outputs: List[str] = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out_batch = pipe(batch)
            # out_batch: List[List[{"label": ..., "score": ...}, ...]]
            for scores in out_batch:
                best = max(scores, key=lambda x: x["score"])
                outputs.append(best["label"].lower())

        return outputs

    # ---- Default models (HF pipeline, batched) ----
    else:
        print("  [Default] Using batched text-classification pipeline...")
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=PIPELINE_DEVICE,
            top_k=1,
        )

        texts = _build_texts(sentences, aspects)
        outs = pipe(texts, batch_size=32)

        def norm(o):
            if isinstance(o, list):
                o = o[0]
            return o["label"].lower()

        return [norm(o) for o in outs]


# ---------------------------------------------------------------------
# MODEL EVALUATION (unchanged, small sample only)
# ---------------------------------------------------------------------


def evaluate_model(
    model_name: str,
    texts: List[str],
    true_labels: List[int],
    label_names: List[str],
    sample_limit: int = 200,
) -> Tuple[pd.DataFrame, Optional[torch.Tensor]]:
    """
    Evaluate a single emotion model against a gold validation set.
    Returns a classification report (as DataFrame) and a confusion matrix.
    """

    classify = build_classifier(model_name)
    preds: List[str] = []

    for t in texts[:sample_limit]:
        try:
            preds.append(classify(t, ""))  # no aspect for evaluation
        except Exception:
            preds.append("unknown")

    pred_indices = [
        label_names.index(p) if p in label_names else -1
        for p in preds
    ]

    valid = [i for i, x in enumerate(pred_indices) if x != -1]
    y_true = [true_labels[i] for i in valid]
    y_pred = [pred_indices[i] for i in valid]

    if len(y_true) < 3:
        print("Not enough valid predictions for evaluation.")
        return pd.DataFrame(), None

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    df_report = pd.DataFrame(report).transpose()
    cm = confusion_matrix(y_true, y_pred)

    print("  Macro F1:", df_report.loc["macro avg", "f1-score"])

    return df_report, cm


# ---------------------------------------------------------------------
# MAIN ENTRY POINT (uses optimized annotate_model)
# ---------------------------------------------------------------------


def run_full_emotion_pipeline(
    input_csv: str,
    model_names: List[str] = MODEL_NAMES,
    dataset_name: str = "dataset",
    results_root: str = results_root,
):
    """
    SINGLE ENTRY POINT:
      1) Load dataset
      2) Annotate with all models (batched & optimized)
      3) Save each annotated CSV
      4) Print per-model timing + total timing
    """

    print("\nStarting full optimized emotion pipeline\n")

    # Start global timer
    global_start = time.time()

    # Load dataset
    df = pd.read_csv(input_csv, dtype=str)

    if "sentence" not in df or "aspect_term" not in df:
        raise ValueError("CSV must contain columns: 'sentence', 'aspect_term'")

    out_dir = os.path.join(results_root, f"emotion_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving outputs to: {out_dir}")

    # Collect per-model timings
    model_timings = []

    # Annotate with each model
    for model_name in model_names:
        print("\n==============================")
        print(f"Annotating with: {model_name}")
        print("==============================")

        # Start timer for this model
        model_start = time.time()

        df_out = df.copy()
        preds = annotate_model(model_name, df_out)
        df_out["emotion_auto"] = preds

        # End timer for this model
        model_end = time.time()
        elapsed = model_end - model_start
        model_timings.append((model_name, elapsed))

        print(f"  → {model_name} completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

        # Save annotated CSV
        safe = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
        csv_path = os.path.join(out_dir, f"{safe}_annotated.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"   Saved → {csv_path}")

    # End global timer
    global_end = time.time()
    total_elapsed = global_end - global_start

    # Print summary
    print("\n================ MODEL TIMING SUMMARY ================")
    total = 0
    for name, sec in model_timings:
        print(f"{name:45s} : {sec:.2f} sec   ({sec/60:.2f} min)")
        total += sec

    print("------------------------------------------------------")
    print(f"TOTAL (per model sum)           : {total:.2f} sec  ({total/60:.2f} min)")
    print(f"TOTAL (wall clock actual)       : {total_elapsed:.2f} sec  ({total_elapsed/60:.2f} min)")
    print("======================================================\n")

    print("\nOptimized pipeline completed successfully.\n")