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
import json

# silence warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ---------------------------------------------------------
# PATH RESOLUTION (UNCHANGED)
# ---------------------------------------------------------
try:
    this_file = os.path.abspath(__file__)
    src_root = os.path.dirname(this_file)
    project_root = os.path.dirname(src_root)
except NameError:
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

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
MODEL_NAMES = [
    "j-hartmann/emotion-english-distilroberta-base",
    "j-hartmann/emotion-english-roberta-large",
    "nateraw/bert-base-uncased-emotion",
    "joeddav/distilbert-base-uncased-go-emotions-student",
    "cardiffnlp/twitter-roberta-base-emotion",
    "mrm8488/t5-base-finetuned-emotion",
    "SamLowe/roberta-base-go_emotions",
]

# ---------------------------------------------------------
# DEVICE HELPERS
# ---------------------------------------------------------
def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"

def get_pipeline_device() -> int:
    return 0 if torch.backends.mps.is_available() else -1

DEVICE = get_device()
PIPELINE_DEVICE = get_pipeline_device()

# ---------------------------------------------------------
# JSONL LOADER → DataFrame
# ---------------------------------------------------------
def load_jsonl_dataset(jsonl_path: str) -> pd.DataFrame:
    """Load MAMS-style JSONL and expand into a flat ABSA DataFrame."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sentence = obj["input"]
            for item in obj["output"]:
                records.append({
                    "sentence": sentence,
                    "aspect_term": item.get("aspect", ""),
                    "polarity": item.get("polarity", ""),
                    "emotion": item.get("emotion", None),
                })
    df = pd.DataFrame(records)
    df["row_id"] = df.index
    return df[["row_id", "sentence", "aspect_term", "polarity", "emotion"]]


# ---------------------------------------------------------
# BUILD CLASSIFIER
# ---------------------------------------------------------
def build_classifier(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type
    arch = str(config.architectures)
    print(f"  Model type: {model_type} | arch={arch}")

    # ---- T5 ----
    if "t5" in model_name.lower():
        pipe = pipeline("text2text-generation", model=model_name,
                        tokenizer=model_name, device=PIPELINE_DEVICE)

        def classify(sentence, aspect):
            prompt = f"classify emotion: [ASPECT] {aspect} [SENTENCE] {sentence}"
            return pipe(prompt)[0]["generated_text"].strip().lower()
        return classify

    # ---- GoEmotions ----
    elif "go_emotions" in model_name.lower():
        pipe = pipeline("text-classification", model=model_name,
                        tokenizer=model_name, return_all_scores=True,
                        device=PIPELINE_DEVICE)

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
            enc = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits
            pred = logits.softmax(1).argmax().item()
            return labels[pred]
        return classify

    # ---- Default ----
    else:
        pipe = pipeline("text-classification", model=model_name,
                        tokenizer=model_name, device=PIPELINE_DEVICE, top_k=1)

        def classify(sentence, aspect):
            text = f"[ASPECT] {aspect} [SENTENCE] {sentence}"
            out = pipe(text)[0]
            if isinstance(out, list):
                out = out[0]
            return out["label"].lower()
        return classify


# ---------------------------------------------------------
# BATCH PIPELINE (annotate_model)
# ---------------------------------------------------------
def _build_texts(sentences, aspects):
    return [f"[ASPECT] {a} [SENTENCE] {s}" for s, a in zip(sentences, aspects)]


def annotate_model(df: pd.DataFrame, model_name=None):
    if df is None:
        raise ValueError("DataFrame required.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("First argument must be a DataFrame.")
    if "sentence" not in df or "aspect_term" not in df:
        raise ValueError("Missing required columns.")

    # RUN ALL MODELS
    if model_name is None:
        results = {}
        for m in MODEL_NAMES:
            print(f"=== Running: {m} ===")
            results[m] = annotate_model(df, m)
        return results

    sentences = df["sentence"].astype(str).tolist()
    aspects = df["aspect_term"].astype(str).tolist()

    # ---- T5 ----
    if "t5" in model_name.lower():
        pipe = pipeline("text2text-generation", model=model_name,
                        tokenizer=model_name, device=PIPELINE_DEVICE)
        texts = [f"classify emotion: [ASPECT] {a} [SENTENCE] {s}"
                 for s, a in zip(sentences, aspects)]

        outputs = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            out = pipe(batch, max_length=32)
            outputs.extend([o["generated_text"].strip().lower() for o in out])
        return outputs

    # ---- CardiffNLP ----
    elif "cardiffnlp" in model_name.lower():
        labels = ["anger", "joy", "optimism", "sadness"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)

        texts = _build_texts(sentences, aspects)
        outputs = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits
            preds = logits.softmax(1).argmax(1).cpu().numpy()
            outputs.extend([labels[p] for p in preds])
        return outputs

    # ---- GoEmotions ----
    elif "go_emotions" in model_name.lower():
        pipe = pipeline("text-classification", model=model_name,
                        tokenizer=model_name, return_all_scores=True,
                        device=PIPELINE_DEVICE)
        texts = _build_texts(sentences, aspects)

        outputs = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            out = pipe(batch)
            for scores in out:
                best = max(scores, key=lambda x: x["score"])
                outputs.append(best["label"].lower())
        return outputs

    # ---- Default ----
    else:
        pipe = pipeline("text-classification", model=model_name,
                        tokenizer=model_name, device=PIPELINE_DEVICE, top_k=1)
        texts = _build_texts(sentences, aspects)
        outs = pipe(texts, batch_size=32)

        def norm(o):
            if isinstance(o, list):
                o = o[0]
            return o["label"].lower()

        return [norm(o) for o in outs]


# ---------------------------------------------------------
# FULL PIPELINE: NOW TAKES DATAFRAME, NOT CSV
# ---------------------------------------------------------
def run_full_emotion_pipeline(
    df: pd.DataFrame,
    model_names: List[str] = MODEL_NAMES,
    dataset_name: str = "dataset",
    results_root: str = results_root,
):
    """
    Runs the full optimized pipeline on a DataFrame
    (not on a CSV path anymore).
    """

    print("\nStarting full optimized emotion pipeline\n")
    global_start = time.time()

    if "sentence" not in df or "aspect_term" not in df:
        raise ValueError("DataFrame must have 'sentence' and 'aspect_term'.")

    out_dir = os.path.join(results_root, f"emotion_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    model_timings = []

    for model_name in model_names:
        print("\n===================================")
        print(f"Annotating with: {model_name}")
        print("===================================")

        model_start = time.time()

        df_out = df.copy()
        preds = annotate_model(df_out, model_name)
        df_out["emotion_auto"] = preds

        model_end = time.time()
        elapsed = model_end - model_start
        model_timings.append((model_name, elapsed))

        print(f"→ {model_name} completed in {elapsed:.2f} sec")

        safe = re.sub(r"[^a-zA-Z0-9]", "_", model_name)

        # ------------------------
        # Save annotated JSONL (grouped, original structure)
        # ------------------------
        jsonl_path = os.path.join(out_dir, f"{safe}_annotated.jsonl")
        save_jsonl_grouped(df_out, jsonl_path)
        print(f"   Saved JSONL → {jsonl_path}")

    global_end = time.time()

    print("\nMODEL TIMING SUMMARY")
    for name, sec in model_timings:
        print(f"{name:40s}: {sec:.2f} sec")

    print(f"\nTotal wall time: {global_end - global_start:.2f} sec")

    print("\nPipeline done.\n")

def save_jsonl_grouped(df_out: pd.DataFrame, jsonl_path: str):
    """
    Convert flat DF back into grouped JSONL format:
    {
      "input": "...sentence...",
      "output": [
         {"aspect": "...", "polarity": "...", "emotion": "..."},
         ...
      ]
    }
    """
    grouped = df_out.groupby("sentence")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sentence, group in grouped:
            outputs = []
            for _, row in group.iterrows():
                outputs.append({
                    "aspect": row["aspect_term"],
                    "polarity": row["polarity"],
                    "emotion": row["emotion_auto"],   # predicted emotion
                })

            obj = {
                "input": sentence,
                "output": outputs
            }

            f.write(json.dumps(obj) + "\n")

    print(f"✔ JSONL saved → {jsonl_path}")
# ---------------------------------------------------------
# MAIN: LOAD JSONL → RUN ONE-MODEL PIPELINE
# ---------------------------------------------------------
if __name__ == "__main__":

    jsonl_path = os.path.join(
        src_root,
        "data",
        "MAMS-ACSA",
        "raw",
        "data_jsonl",
        "train.jsonl"
    )

    print("Loading JSONL dataset...")
    df = load_jsonl_dataset(jsonl_path)

    model_to_use = ["SamLowe/roberta-base-go_emotions"]

    run_full_emotion_pipeline(
        df=df,
        model_names=model_to_use,     # ONE MODEL ONLY
        dataset_name="MAMS-ACSA",
        results_root=results_root,
    )