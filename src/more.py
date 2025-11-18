from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd, torch, os, re

import warnings
warnings.filterwarnings("ignore")

INPUT_CSV = "/Users/hd/Desktop/EMOTION-PRED/src/data/mams_for_annotation.csv"   # needs columns: sentence, aspect_term
OUTPUT_DIR = "/Users/hd/Desktop/EMOTION-PRED/src/results/outputs_with_emotion_2"; os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    "j-hartmann/emotion-english-roberta-large",
    "j-hartmann/emotion-english-distilroberta-base",
    "SamLowe/roberta-base-go_emotions",
    "joeddav/distilbert-base-uncased-go-emotions-student",
    "cardiffnlp/twitter-roberta-base-emotion",
    "arpanghoshal/EmoRoBERTa",
    "nateraw/bert-base-uncased-emotion",
    "mrm8488/t5-base-finetuned-emotion"
]

df = pd.read_csv(INPUT_CSV)

def safe_name(name): return re.sub(r"[^a-zA-Z0-9]", "_", name)

for m in MODELS:
    print(f"\n▶ {m}")
    out_csv = os.path.join(OUTPUT_DIR, f"{safe_name(m)}_addedemotion.csv")

    try:
        lower = m.lower()

        # (A) T5 generative
        if "t5" in lower:
            clf = pipeline("text2text-generation", model=m, tokenizer=m ,device=0)
            def classify(s, a):
                text = f"classify emotion: [ASPECT] {a} [SENTENCE] {s}"
                return clf(text, max_new_tokens=4)[0]["generated_text"].strip().lower()

        # (B) GoEmotions multi-label
        elif "go_emotions" in lower or "goemotions" in lower:
            clf = pipeline("text-classification", model=m, tokenizer=m, return_all_scores=True, device=0)
            def classify(s, a):
                text = f"[ASPECT] {a} [SENTENCE] {s}"
                scores = sorted(clf(text)[0], key=lambda x: x["score"], reverse=True)
                # pick top-1; adapt to threshold if you want multi-label output
                return scores[0]["label"].lower()

        # (C) CardiffNLP tweet models (some need explicit labels)
        elif "cardiffnlp/twitter-roberta-base-emotion" in m:
            tok = AutoTokenizer.from_pretrained(m)
            mdl = AutoModelForSequenceClassification.from_pretrained(m)
            # Single-label TweetEval emotions (see model card)
            labels = ['anger','joy','optimism','sadness']
            def classify(s, a):
                text = f"[ASPECT] {a} [SENTENCE] {s}"
                inputs = tok(text, return_tensors="pt")
                with torch.no_grad():
                    probs = torch.softmax(mdl(**inputs).logits, dim=1)[0]
                return labels[int(torch.argmax(probs))]

        # (D) All other plain classifiers
        else:
            clf = pipeline("text-classification", model=m, tokenizer=m, return_all_scores=False ,device=0)
            def classify(s, a):
                text = f"[ASPECT] {a} [SENTENCE] {s}"
                return clf(text)[0]["label"].lower()

        df_out = df.copy()
        df_out["emotion_auto"] = df_out.apply(lambda r: classify(r["sentence"], r["aspect_term"]), axis=1)
        df_out.to_csv(out_csv, index=False)
        print(f"✅ saved → {out_csv}")

    except Exception as e:
        print(f"⚠️ {m} failed: {e}")
