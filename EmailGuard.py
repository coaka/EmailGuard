# =========================================================
# EmailGuard: Multi-Agent Phishing Detection Framework
# =========================================================

import os
import re
import numpy as np
import pandas as pd
import torch
import joblib
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.special import softmax

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# =========================================================
# CONFIG
# =========================================================

EMAIL_DATA_PATH = "phishing_email.csv"   # merged 7-dataset corpus
URL_DATA_PATH   = "PhiUSIIL.csv"

OUTPUT_DIR = "./EmailGuard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# DATA LOADING
# =========================================================

df = pd.read_csv(EMAIL_DATA_PATH)

df.columns = [c.lower() for c in df.columns]

text_col = [c for c in df.columns if "text" in c or "body" in c][0]
label_col = [c for c in df.columns if "label" in c or "type" in c][0]

df = df[[text_col, label_col]]
df.columns = ["text", "label"]

df["label"] = df["label"].astype(str).str.lower().replace({
    "phishing email": "phishing",
    "spam": "phishing",
    "legit": "legit",
    "ham": "legit",
    "safe": "legit",
    "1": "phishing",
    "0": "legit"
})

df = df[df["label"].isin(["phishing", "legit"])].dropna()

label_map = {"legit": 0, "phishing": 1}
df["y"] = df["label"].map(label_map)

texts = df["text"].values
labels = df["y"].values

# =========================================================
# URL FEATURE ENGINEERING
# =========================================================

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")

def extract_urls(text):
    return URL_REGEX.findall(text)

def url_features(url):
    return [
        len(url),
        sum(c.isdigit() for c in url),
        sum(c in "-_?=&%./" for c in url),
        url.count("."),
        1 if "https" in url else 0
    ]

def vectorize_urls(urls):
    if len(urls) == 0:
        return np.zeros(5)
    feats = np.array([url_features(u) for u in urls])
    return np.concatenate([feats.mean(axis=0), feats.max(axis=0)])

# =========================================================
# LOAD PHIUSIIL DATA (LEXICAL AGENT)
# =========================================================

url_df = pd.read_csv(URL_DATA_PATH)

url_df.columns = [c.lower() for c in url_df.columns]

url_col = [c for c in url_df.columns if "url" in c][0]
url_label_col = [c for c in url_df.columns if "label" in c][0]

url_df = url_df[[url_col, url_label_col]]
url_df.columns = ["url", "label"]

url_df["y"] = url_df["label"].astype(str).str.lower().map(label_map)

X_url = np.array([vectorize_urls([u]) for u in url_df["url"]])
y_url = url_df["y"].values

lexical_model = LogisticRegression(max_iter=1000)
lexical_model.fit(X_url, y_url)

joblib.dump(lexical_model, os.path.join(OUTPUT_DIR, "lexical_model.joblib"))

# =========================================================
# TF-IDF SURFACE AGENT
# =========================================================

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
surface_model = LogisticRegression(max_iter=1000)

X_surface = tfidf.fit_transform(texts)
surface_model.fit(X_surface, labels)

joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf.joblib"))
joblib.dump(surface_model, os.path.join(OUTPUT_DIR, "surface_model.joblib"))

# =========================================================
# SEMANTIC AGENT (DISTILBERT)
# =========================================================

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

from datasets import Dataset

dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_ds = dataset["train"]
val_ds   = dataset["test"]

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True)

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "y"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "y"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(DEVICE)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()

semantic_model_path = os.path.join(OUTPUT_DIR, "semantic_model")
trainer.save_model(semantic_model_path)
tokenizer.save_pretrained(semantic_model_path)

# =========================================================
# LOAD MODELS FOR INFERENCE
# =========================================================

semantic_model = AutoModelForSequenceClassification.from_pretrained(
    semantic_model_path
).to(DEVICE)

semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_path)

surface_model = joblib.load(os.path.join(OUTPUT_DIR, "surface_model.joblib"))
tfidf = joblib.load(os.path.join(OUTPUT_DIR, "tfidf.joblib"))
lexical_model = joblib.load(os.path.join(OUTPUT_DIR, "lexical_model.joblib"))

# =========================================================
# AGENTS
# =========================================================

def semantic_score(text):
    inputs = semantic_tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = semantic_model(**inputs).logits
    probs = softmax(logits.cpu().numpy()[0])
    return float(probs[1])


def surface_score(text):
    vec = tfidf.transform([text])
    return float(surface_model.predict_proba(vec)[0][1])


def lexical_score(urls):
    feats = vectorize_urls(urls)
    return float(lexical_model.predict_proba([feats])[0][1])

# =========================================================
# ENSEMBLE (STACKING)
# =========================================================

meta_model = LogisticRegression()

def build_meta_features(text):
    urls = extract_urls(text)

    return np.array([
        semantic_score(text),
        surface_score(text),
        lexical_score(urls)
    ])

# TRAIN META MODEL
X_meta = []
y_meta = []

for i in range(len(texts)):
    X_meta.append(build_meta_features(texts[i]))
    y_meta.append(labels[i])

X_meta = np.array(X_meta)
y_meta = np.array(y_meta)

meta_model.fit(X_meta, y_meta)

joblib.dump(meta_model, os.path.join(OUTPUT_DIR, "meta_model.joblib"))

# =========================================================
# SDN ENFORCEMENT LAYER
# =========================================================

def emailguard(text):

    x = build_meta_features(text)
    score = meta_model.predict_proba([x])[0][1]

    if score >= 0.85:
        action = "quarantine"
    elif score >= 0.50:
        action = "mirror"
    else:
        action = "deliver"

    return {
        "risk_score": round(score, 4),
        "action": action
    }

# =========================================================
# TEST EXAMPLE
# =========================================================

if __name__ == "__main__":

    test_email = """
    Urgent: Your bank account is locked.
    Please verify immediately: http://secure-login.com
    """

    print(emailguard(test_email))