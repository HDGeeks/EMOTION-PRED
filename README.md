

# 🐍 Virtual Environment Setup Guide

This project requires a Python virtual environment.  
Python **3.9 or higher** is recommended.

Follow the steps below depending on your operating system.

---

## 1. Create a Virtual Environment

### macOS / Linux
```bash
python3 -m venv venv

Windows

python -m venv venv


⸻

2. Activate the Virtual Environment

macOS / Linux

source venv/bin/activate

Windows (PowerShell)

.\venv\Scripts\activate

When activated, your terminal prompt will show:

(venv) $

This confirms that the environment is active.

⸻

3. Install Project Dependencies

Make sure the virtual environment is still activated, then run:

pip install -r requirements.txt


⸻

4. Set Up Jupyter Kernel (Optional but Recommended)

If you are using Jupyter Notebook or JupyterLab, register the virtual environment as a kernel:

python -m ipykernel install --user --name=venv --display-name="Python (venv)"

Then inside Jupyter:
	•	Open Kernel → Change Kernel
	•	Select Python (venv)

⸻

🎉 You’re Ready!

Run the notebooks, experiment, break things, fix things —
enjoy the project!

# 🔍 Understanding the Model Types in This Repository

This project evaluates emotion classification across several **state-of-the-art transformer architectures**.  
Although all models output emotions, they belong to very different families.  
This section explains **what each type of model is**, **how it works**, and **why it is included** in the experiment.

---

# 🧠 1. Encoder-Only Transformer Models  
*(BERT, RoBERTa, DistilBERT, DistilRoBERTa)*

Encoder-only models convert text into contextual embeddings and then predict a fixed label such as `joy`, `anger`, or `sadness`.  
They do **not** generate text—they classify.

These architectures power most traditional NLP classification systems.

---

## **BERT (Bidirectional Encoder Representations from Transformers)**  
**Models in this repo:**  
- `nateraw/bert-base-uncased-emotion`

**What it is**  
The original milestone transformer encoder. It reads text in both directions and builds deep contextual understanding.

**Why it matters**  
BERT became the foundation for modern NLP. It remains one of the most stable and reliable baselines for classification tasks.

**Used for**  
- Emotion classification  
- Sentiment analysis  
- Topic tagging  
- QA (with special heads)  
- NLI  

---

## **RoBERTa (Robustly Optimized BERT)**  
**Models in this repo:**  
- `j-hartmann/emotion-english-roberta-large`  
- `cardiffnlp/twitter-roberta-base-emotion`

**What it is**  
A refined and retrained version of BERT with:  
- more data  
- better hyperparameters  
- removal of weak pretraining tasks  

**Why it matters**  
RoBERTa consistently outperforms BERT on classification tasks.

**Special variant**  
The Twitter-RoBERTa model is tuned for **short, informal, social-media style language**, improving emotion detection in noisy text.

---

## **DistilBERT & DistilRoBERTa (Distilled Models)**  
**Models in this repo:**  
- `joeddav/distilbert-base-uncased-go-emotions-student`  
- `j-hartmann/emotion-english-distilroberta-base`

**What they are**  
Lightweight versions of BERT/RoBERTa created through **knowledge distillation**.  
A smaller “student” model learns to mimic a larger “teacher”.

**Why they matter**  
They keep **~95% of the accuracy**, but are:  
- smaller  
- faster  
- cheaper  
Perfect for high-volume or real-time classification.

---

# 🧩 2. Sequence-to-Sequence Models  
*(T5 – Text-to-Text Transfer Transformer)*

**Model in this repo:**  
- `mrm8488/t5-base-finetuned-emotion`

T5 is fundamentally different from all encoder-only models above.

---

## **What T5 is**  
A **text-to-text transformer** with both an encoder and a decoder.  
Every task becomes a *generation* task.

Examples:  
- Summarization → *generate a summary*  
- Translation → *generate a translation*  
- Emotion classification → *generate the emotion label as text*

---

## **How T5 differs from the others**

Unlike BERT/RoBERTa models:

- T5 does **not** use a classification head  
- T5 does **not** have `id2label` or `label2id`  
- T5 does **not** produce logits over fixed classes  

Instead:

- The model **generates** text labels like `"joy"` or `"anger"`  
- Labels are learned as **words**, not as predefined class indices  

This means the T5 model behaves more like a small LLM that answers with a one-word emotion.

---

# 🎯 Summary: Why These Model Families Were Included

| Model Family | Architecture | Behavior | Strength |
|--------------|-------------|----------|----------|
| **BERT** | Encoder-only | Predicts fixed labels | Stable, strong baseline |
| **RoBERTa** | Encoder-only | Predicts fixed labels | High accuracy, robust |
| **DistilBERT / DistilRoBERTa** | Encoder-only (compressed) | Predicts fixed labels | Fast & efficient |
| **T5** | Encoder + Decoder (Seq2Seq) | Generates label words | Flexible, generative |

Using multiple architectures allows the project to measure:

- how different model families respond to emotional language  
- how **encoder vs decoder** architectures react to prompt variations  
- how generative and non-generative models express sensitivity  

This diversity provides a more complete picture of real-world LLM behavior.

---

# 📌 Models Used in This Repository

```python
MODEL_NAMES = [
    "j-hartmann/emotion-english-distilroberta-base",
    "j-hartmann/emotion-english-roberta-large",
    "nateraw/bert-base-uncased-emotion",
    "joeddav/distilbert-base-uncased-go-emotions-student",
    "cardiffnlp/twitter-roberta-base-emotion",
    "mrm8488/t5-base-finetuned-emotion"
]
