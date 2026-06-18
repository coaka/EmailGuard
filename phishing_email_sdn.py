# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Mean confusion matrix

arr=np.array([[2200, 2210, 2195,  2219],[55 , 60 , 62 , 59],[12 , 15 , 14 , 15],[1440 , 1455 , 1438 , 1459]])
# Standard deviation matrix (calculated from the 4 folds)
arr_2=np.array([[38500 , 38620 , 38540 , 38604],[1015 , 1035 , 1028 , 1038],[410 , 415 , 412 , 411],[42450 , 42500 , 42470 , 42696]])
std_list = np.std(arr, axis=1, ddof=1).tolist()
std_list2 = np.std(arr_2, axis=1, ddof=1).tolist()
print(std_list2 )
# Class names
arr1=[]
arr2=[]
c=0
for i in std_list:
  #arr2.append([i])

  if c >2:
    arr1.append(arr2)
    arr2=[]
  arr2.append(i)
  c+=1
print(arr1)


class_names = ['legit', 'phishing']
arr1=[[10.677078252031311 ,2.943920288775949], [1.4142135623730951,10.55146119422961]]
arr_2nd=[[55.952360688952766, 10.23067283548187], [2.160246899469287, 113.21366230863364]]
arr1 =np.array(arr1 )
arr_2nd =np.array(arr_2nd )
      # Create the plot for standard deviation
plt.figure(figsize=(5, 4))
sns.heatmap(
    arr1,#arr_2nd,     #(dataset1 or     dataset2)
    annot=True,
    fmt=".4f",  # Format for decimal values
    cmap="Blues",  # Using Reds colormap for standard deviation
    cbar=True,
    cbar_kws={'label': 'Standard Deviation'},
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray',
    square=True,
    annot_kws={'size': 11}
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Standard Deviation of Confusion Matrix\nAcross 4 Folds")
plt.tight_layout()
plt.show()



#drive/MyDrive/Phishing_Email.csv
# distilbert_finetune_sdn_legacy.py
# Requirements:
# pip install transformers datasets accelerate torch scikit-learn joblib scipy

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import hashlib
import datetime
import tldextract
import whois
from dateutil import parser as dateparser
from sklearn.metrics import roc_curve, auc
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from scipy.special import softmax
import joblib

# ---------------------------
# 1) Load dataset
# ---------------------------
DATA_PATH = "drive/MyDrive/Phishing_Email.csv"  # change to your CSV file
OUTPUT_DIR = "drive/MyDrive/artifacts/distilbert_phish"
BEST_DIR = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Identify columns
cols_lower = {c.lower(): c for c in df.columns}
text_col = None
for key in ["email text", "email_text", "text", "body"]:
    if key in cols_lower:
        text_col = cols_lower[key]
        break
label_col = None
for key in ["email type", "email_type", "label", "target"]:
    if key in cols_lower:
        label_col = cols_lower[key]
        break
if text_col is None or label_col is None:
    raise ValueError(f"Could not find expected text/label columns. Found: {list(df.columns)}")

# Normalize labels
df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
df["label"] = df["label"].astype(str).str.strip().str.lower()
label_map = {
    "phishing email": "phishing",
    "phishing": "phishing",
    "spam": "phishing",
    "1": "phishing",
    "safe email": "legit",
    "safe": "legit",
    "legit": "legit",
    "ham": "legit",
    "0": "legit",
}
df["label"] = df["label"].map(lambda x: label_map.get(x, x))
df = df[df["label"].isin(["phishing", "legit"])].dropna(subset=["text"]).reset_index(drop=True)

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist(),
)

# Encode labels
le = LabelEncoder()
train_label_ids = le.fit_transform(train_labels)  # e.g., legit=0, phishing=1
val_label_ids = le.transform(val_labels)
id2label = {i: lab for i, lab in enumerate(le.classes_)}  # {0:'legit', 1:'phishing'}
label2id = {lab: i for i, lab in id2label.items()}

# HF datasets
train_ds = Dataset.from_dict({"text": train_texts, "labels": train_label_ids})
val_ds = Dataset.from_dict({"text": val_texts, "labels": val_label_ids})
raw_datasets = DatasetDict({"train": train_ds, "validation": val_ds})

# ---------------------------
# 2) Tokenizer and preprocessing (DistilBERT)
# ---------------------------
model_ckpt = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = raw_datasets.map(preprocess_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# 3) Metrics (scikit-learn)
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=label2id.get("phishing", 1), zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------------
# 4) Model and Trainer (legacy-friendly TrainingArguments)
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

# LEGACY-FRIENDLY: remove newer args like evaluation_strategy/save_strategy/load_best_model_at_end
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    fp16=True,  # enable on GPU if supported
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Manual evaluation (since automatic eval per epoch may not be configured)
metrics = trainer.evaluate()
# ---------------------------
# Threshold calibration targeting FPR <= 0.5% on validation set
# ---------------------------
TARGET_FPR = 0.005  # 0.5%
def extract_urls(text: str):
    if not text:
        return []
    return re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_infer_model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR).to(_device)
_infer_model.eval()
_infer_tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
_infer_label_encoder = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
def fused_phishing_probability(email_text: str):
    urls = extract_urls(email_text)
    # Body model probability
    inputs = _infer_tokenizer(email_text, return_tensors="pt", truncation=True).to(_device)
    with torch.no_grad():
        outputs = _infer_model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    probs = softmax(logits)
    #logits = outputs.logits.detach().cpu().numpy() # shape (num_labels,)
    #probs = softmax(logits) # shape (num_labels,)
    #phishing_idx = int(np.where(_infer_label_encoder.classes_ == "phishing"))
    match = np.where(_infer_label_encoder.classes_ == "phishing")
    if len(match) == 0:
      raise ValueError("Class 'phishing' not found in label encoder classes: "
      f"{list(_infer_label_encoder.classes_)}")
    print("match---->>>",match)
    phishing_idx = match
    p_body = float(probs[phishing_idx])

    # URL text model probability
    p_url_text = url_text_score(urls)

    # URL lexical probability
    url_feats = [url_lexical_features(u) for u in urls]
    v = vectorize_url_features(url_feats)
    p_url_lex = url_lexical_score(v)

    # Late-score averaging with availability-aware weights
    scores = [p_body]
    if p_url_text is not None:
        scores.append(p_url_text)
    # Always include lexical (it’s cheap and robust), but you can gate if no URLs
    if urls:
        scores.append(p_url_lex)

    p_fused = float(np.mean(scores)) if scores else p_body
    return {
        "p_body": p_body,
        "p_url_text": p_url_text,
        "p_url_lex": p_url_lex if urls else None,
        "p_fused": p_fused,
        "urls": urls,
    }
# Collect fused probabilities on the validation set
val_texts_list = val_texts  # already created earlier
val_labels_bin = (np.array(val_labels) == "phishing").astype(int)  # 1 for phishing

val_probs = []
for t in val_texts_list:
    p = fused_phishing_probability(t)["p_fused"]
    val_probs.append(p)
val_probs = np.array(val_probs, dtype=np.float32)

# Compute ROC and choose the highest threshold whose FPR <= TARGET_FPR
fpr, tpr, thresholds = roc_curve(val_labels_bin, val_probs)
roc_auc = auc(fpr, tpr)
# Find indices where FPR <= target
ok_idx = np.where(fpr <= TARGET_FPR)[0]
if len(ok_idx) == 0:
    # No threshold meets target FPR; choose the minimal FPR threshold to be conservative
    best_idx = int(np.argmin(fpr))
else:
    best_idx = ok_idx[-1]  # largest threshold that still meets FPR target
CALIBRATED_THRESHOLD = float(thresholds[best_idx])

print(f"Calibrated threshold for FPR<={TARGET_FPR*100:.2f}%: {CALIBRATED_THRESHOLD:.4f} (AUC={roc_auc:.4f})")

#######################################
# ---------------------------
# 6) Confusion matrix on validation set
# ---------------------------

# Get predictions on validation dataset
pred_output = trainer.predict(tokenized["validation"])
# pred_output.predictions is (num_examples, num_labels) logits
from scipy.special import softmax

probs = softmax(pred_output.predictions, axis=1)
phish_idx = label2id["phishing"]
y_pred_ids = (probs[:, phish_idx] >= 0.5).astype(int)  # 0=legit, 1=phishing

#y_pred_ids = np.argmax(pred_output.predictions, axis=-1)
y_true_ids = pred_output.label_ids

# Map numeric ids to string labels using your encoder/classes
# id2label you defined is consistent with LabelEncoder().classes_
# However, for sklearn confusion/report, we can pass the display labels directly:
class_names = list(id2label.values())  # e.g., ['legit', 'phishing']

# Confusion matrix
cm = confusion_matrix(y_true_ids, y_pred_ids, labels=list(id2label.keys()))

# Plot
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - DistilBERT Phishing Classifier")
plt.tight_layout()

# Optionally save to your artifacts directory
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_val.png")
plt.savefig(cm_path, dpi=150)
plt.show()
plt.close()

print(f"Saved confusion matrix to: {cm_path}")

# Per-class metrics for deeper insight
print("Classification report (per-class):")
print(
    classification_report(
        y_true_ids,
        y_pred_ids,
        target_names=class_names,
        zero_division=0
    )
)

#######################################
print("Validation metrics:", metrics)

# Save final model and tokenizer
os.makedirs(BEST_DIR, exist_ok=True)
trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

# ---------------------------
# 5) SDN-style action function (softmax probs)
# ---------------------------
RISK_THRESHOLDS = {"block": 0.8, "monitor": 0.5}
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# ---------------------------
# URL feature extraction
# ---------------------------

def url_lexical_features(url: str):
    # Lightweight lexical features; add/remove as needed
    u = url.strip()
    length = len(u)
    num_digits = sum(c.isdigit() for c in u)
    num_special = sum(c in "-_=?&%./:#" for c in u)
    at_symbol = 1 if "@" in u else 0
    https = 1 if u.lower().startswith("https") else 0
    dots = u.count(".")
    # Suspicious keywords
    kw_list = ["login", "verify", "update", "secure", "confirm", "password", "unlock", "invoice"]
    kw_hit = int(any(k in u.lower() for k in kw_list))
    # TLD and domain age (best-effort; may fail for some TLDs)
    ext = tldextract.extract(u)
    domain = ".".join([ext.domain, ext.suffix]) if ext.suffix else ext.domain
    subdomain = ext.subdomain
    sub_len = len(subdomain) if subdomain else 0
    domain_age_days = -1
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            if not isinstance(created, datetime.datetime):
                created = dateparser.parse(str(created))
            domain_age_days = (datetime.datetime.utcnow() - created.replace(tzinfo=None)).days
    except Exception:
        pass

    return {
        "length": length,
        "num_digits": num_digits,
        "num_special": num_special,
        "at_symbol": at_symbol,
        "https": https,
        "dots": dots,
        "kw_hit": kw_hit,
        "sub_len": sub_len,
        "domain_age_days": domain_age_days,
    }

URL_FEATURE_KEYS = [
    "length","num_digits","num_special","at_symbol","https","dots",
    "kw_hit","sub_len","domain_age_days"
]

def vectorize_url_features(url_features_list):
    # Aggregate multiple-URL email -> simple stats (mean/max) per feature
    if not url_features_list:
        return np.zeros(len(URL_FEATURE_KEYS) * 2, dtype=np.float32)
    feats = np.array([[f[k] for k in URL_FEATURE_KEYS] for f in url_features_list], dtype=np.float32)
    feats_mean = feats.mean(axis=0)
    feats_max = feats.max(axis=0)
    return np.concatenate([feats_mean, feats_max]).astype(np.float32)

# Load fine-tuned model/tokenizer for inference

####################
# ---------------------------
# URL text classifier (DistilBERT) for URL strings
# Replace with a suitable URL-focused checkpoint if available
# Example placeholder: reuse the fine-tuned email model for URL strings
# For best results, use a URL-specific model like a distilled URL classifier.
MODEL_URL_CKPT = BEST_DIR  # TODO: replace with your URL model repo if you have one
_url_device = _device
_url_model = AutoModelForSequenceClassification.from_pretrained(MODEL_URL_CKPT).to(_url_device)
_url_model.eval()
_url_tokenizer = AutoTokenizer.from_pretrained(MODEL_URL_CKPT)

def url_text_score(urls):
    # Returns phishing probability from URL text model averaged across URLs
    if not urls:
        return None
    probs_list = []
    for u in urls:
        inputs = _url_tokenizer(u, return_tensors="pt", truncation=True).to(_url_device)
        with torch.no_grad():
            outputs = _url_model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]
        p = float(softmax(logits)[int(np.where(_infer_label_encoder.classes_ == "phishing"))])
        probs_list.append(p)
    return float(np.mean(probs_list)) if probs_list else None

####################
# ---------------------------
# Fusion of body model + URL models (late-score averaging)
# ---------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Simple linear calibrator placeholder for URL lexical features -> phishing prob
# You can later train a small logistic regression on validation data to learn weights.
_lex_weights = np.zeros(len(URL_FEATURE_KEYS) * 2, dtype=np.float32)  # start as neutral
_lex_bias = 0.0

def url_lexical_score(features_vec: np.ndarray):
    # Convert features to a normalized vector; for now, min-max-esque scaling heuristics
    # Feel free to replace with a proper StandardScaler + LogisticRegression trained on data.
    v = features_vec.copy().astype(np.float32)
    # crude clipping
    v = np.clip(v, -1_000.0, 10_000.0)
    z = float(np.dot(v, _lex_weights) + _lex_bias)
    return float(sigmoid(z))  # returns 0..1


'''
def sdn_action_for_email(text: str):
    inputs = _infer_tokenizer(text, return_tensors="pt", truncation=True).to(_device)
    with torch.no_grad():
        outputs = _infer_model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = softmax(logits)

    phishing_idx = int(np.where(_infer_label_encoder.classes_ == "phishing")[0][0])
    p_phish = float(probs[phishing_idx])

    if p_phish >= RISK_THRESHOLDS["block"]:
        action = "block"
    elif p_phish >= RISK_THRESHOLDS["monitor"]:
        action = "monitor"
    else:
        action = "allow"

    return {
        "type": "phishing_detection",
        "risk_score": round(p_phish, 3),
        "prediction": _infer_label_encoder.classes_[int(np.argmax(probs))],
        "action": action,
        "indicators": {"has_url": bool(URL_REGEX.search(text))},
    }'''
def sdn_action_for_email(text: str):
    fusion = fused_phishing_probability(text)
    p_phish = float(fusion["p_fused"])

    # Use calibrated threshold for allow/monitor/block policy tiers.
    # Keep your monitor/block tiers, but anchor them to calibrated baseline.
    # Example strategy:
    # - Allow if below calibrated threshold
    # - Monitor if between calibrated and calibrated+0.2
    # - Block if above calibrated+0.2
    # Tune deltas to your tolerance.
    thr_allow = CALIBRATED_THRESHOLD
    thr_block = min(1.0, CALIBRATED_THRESHOLD + 0.20)

    if p_phish >= thr_block:
        action = "block"
    elif p_phish >= thr_allow:
        action = "monitor"
    else:
        action = "allow"

    return {
        "type": "phishing_detection",
        "risk_score_calibrated": round(p_phish, 4),
        "risk_components": {
            "body": round(fusion["p_body"], 4),
            "url_text": None if fusion["p_url_text"] is None else round(fusion["p_url_text"], 4),
            "url_lex": None if fusion["p_url_lex"] is None else round(fusion["p_url_lex"], 4),
        },
        "prediction": "phishing" if p_phish >= thr_allow else "legit",
        "action": action,
        "calibration": {
            "target_fpr": TARGET_FPR,
            "threshold": round(CALIBRATED_THRESHOLD, 4),
        },
        "indicators": {
            "has_url": bool(URL_REGEX.search(text)),
            "urls": fusion["urls"],
        },
    }

# Example
if __name__ == "__main__":
    example = "Urgent: Your account is locked. Verify now at http://secure-login-update.com"
    print(json.dumps(sdn_action_for_email(example), indent=2))

## LOAD AND TEST THE MODEL AND PRINT RESULTS
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = "drive/MyDrive/Phishing_Email.csv"
OUTPUT_DIR = "drive/MyDrive/artifacts/distilbert_phish"
BEST_DIR = os.path.join(OUTPUT_DIR, "best_model")

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(DATA_PATH)

# Find text/label columns
cols_lower = {c.lower(): c for c in df.columns}
text_col = next((cols_lower[k] for k in ["email text", "email_text", "text", "body"] if k in cols_lower), None)
label_col = next((cols_lower[k] for k in ["email type", "email_type", "label", "target"] if k in cols_lower), None)
if text_col is None or label_col is None:
    raise ValueError("Could not find expected text/label columns")

# Normalize labels
df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
df["label"] = df["label"].astype(str).str.strip().str.lower()
label_map = {
    "phishing email": "phishing", "phishing": "phishing", "spam": "phishing", "1": "phishing",
    "safe email": "legit", "safe": "legit", "legit": "legit", "ham": "legit", "0": "legit"
}
df["label"] = df["label"].map(lambda x: label_map.get(x, x))
df = df[df["label"].isin(["phishing", "legit"])].dropna(subset=["text"]).reset_index(drop=True)

# ---------------------------
# Train/val split (same as before)
# ---------------------------
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist(),
)

# Encode labels (load existing encoder if saved, else fit again)
label_encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
if os.path.exists(label_encoder_path):
    le = joblib.load(label_encoder_path)
else:
    le = LabelEncoder()
    le.fit(train_labels)
val_label_ids = le.transform(val_labels)

id2label = {i: lab for i, lab in enumerate(le.classes_)}
label2id = {lab: i for i, lab in id2label.items()}

# ---------------------------
# Load fine-tuned model + tokenizer
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR).to(device)

# ---------------------------
# Prepare validation dataset
# ---------------------------
val_ds = Dataset.from_dict({"text": val_texts, "labels": val_label_ids})
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
val_ds = val_ds.map(preprocess_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ---------------------------
# Prediction
# ---------------------------
pred_output = trainer.predict(val_ds)
logits = pred_output.predictions
y_true = pred_output.label_ids
y_pred = np.argmax(logits, axis=-1)

# ---------------------------
# Metrics
# ---------------------------
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=label2id["phishing"])

print("\n=== Validation Metrics ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=list(id2label.keys()))
class_names = list(id2label.values())

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names, cbar=False
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

#This code plot Tsne for dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
OUTPUT_DIR = "drive/MyDrive/artifacts/distilbert_phish"
# ---------------------------
# Load and merge datasets
# ---------------------------
files=[
    'drive/MyDrive/Phishing_email_2nd/SpamAssasin.csv',
    'drive/MyDrive/Phishing_email_2nd/Nigerian_Fraud.csv',
    'drive/MyDrive/Phishing_email_2nd/phishing_email.csv',
    'drive/MyDrive/Phishing_email_2nd/CEAS_08.csv',
    'drive/MyDrive/Phishing_email_2nd/Enron.csv',
    'drive/MyDrive/Phishing_email_2nd/Ling.csv',
    'drive/MyDrive/Phishing_email_2nd/Nazario.csv'
]

dfs = []
for f in files:
    df = pd.read_csv(f)

    cols_lower = {c.lower(): c for c in df.columns}
    text_col = next((cols_lower[k] for k in ["email text", "email_text", "text", "body"] if k in cols_lower), None)
    label_col = next((cols_lower[k] for k in ["email type", "email_type", "label", "target"] if k in cols_lower), None)

    if text_col and label_col:
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data["label"] = data["label"].astype(str).str.strip().str.lower()

# ---------------------------
# Normalize labels
# ---------------------------
label_map = {
    "phishing email": "phishing", "phishing": "phishing", "spam": "phishing", "1": "phishing",
    "safe email": "legit", "safe": "legit", "legit": "legit", "ham": "legit", "0": "legit"
}

data["label"] = data["label"].map(lambda x: label_map.get(x, x))
data = data[data["label"].isin(["phishing", "legit"])].dropna(subset=["text"]).reset_index(drop=True)

print(f"Total merged dataset size: {len(data)}")

# ---------------------------
# Encode labels
# ---------------------------
le = LabelEncoder()
data["label_id"] = le.fit_transform(data["label"])

# ---------------------------
# Convert text to TF-IDF features
# ---------------------------
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(data["text"]).toarray()

# ---------------------------
# Apply t-SNE
# ---------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# ---------------------------
# Plot t-SNE
# ---------------------------
plt.figure(figsize=(8,6))

for label in np.unique(data["label_id"]):
    idx = data["label_id"] == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=le.inverse_transform([label])[0], alpha=0.6, s=10)

plt.title("t-SNE Visualization of Phishing vs Legit Emails")
#plt.xlabel("t-SNE Component 1")
#plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()
save_path = os.path.join(OUTPUT_DIR, "Tsne_dataset1.pdf")
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch, os
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
OUTPUT_DIR = "drive/MyDrive/artifacts/distilbert_phish"
# ---------------------------
# Load and merge datasets
# ---------------------------
files = [
    'drive/MyDrive/Phishing_email_2nd/SpamAssasin.csv',
    'drive/MyDrive/Phishing_email_2nd/Nigerian_Fraud.csv',
    'drive/MyDrive/Phishing_email_2nd/phishing_email.csv',
    'drive/MyDrive/Phishing_email_2nd/CEAS_08.csv',
    'drive/MyDrive/Phishing_email_2nd/Enron.csv',
    'drive/MyDrive/Phishing_email_2nd/Ling.csv',
    'drive/MyDrive/Phishing_email_2nd/Nazario.csv'] #dataset1
#files = ['drive/MyDrive/Phishing_email_2nd/PhiUSIIL_Phishing_URL_Dataset.csv']#2nd dataset

dfs = []

for f in files:
    df = pd.read_csv(f)

    cols_lower = {c.lower(): c for c in df.columns}

    text_col = next(
        (cols_lower[k] for k in ["email text", "email_text", "text", "body"] if k in cols_lower),
        None
    )

    label_col = next(
        (cols_lower[k] for k in ["email type", "email_type", "label", "target"] if k in cols_lower),
        None
    )

    if text_col and label_col:
        df = df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )
        dfs.append(df)
try:
  data = pd.concat(dfs, ignore_index=True)
  data = data.sample(frac=1, random_state=42).reset_index(drop=True)
except: ValueError,
# ---------------------------
# Normalize labels
# ---------------------------
data["label"] = data["label"].astype(str).str.strip().str.lower()

label_map = {
    "phishing email": "phishing",
    "phishing": "phishing",
    "spam": "phishing",
    "1": "phishing",
    "safe email": "legit",
    "safe": "legit",
    "legit": "legit",
    "ham": "legit",
    "0": "legit"
}

data["label"] = data["label"].map(lambda x: label_map.get(x, x))

data = data[data["label"].isin(["phishing", "legit"])]
data = data.dropna(subset=["text"]).reset_index(drop=True)

print(f"Total merged dataset size: {len(data)}")

# ---------------------------
# Optional: Sample for faster t-SNE
# ---------------------------
MAX_SAMPLES = 12000

if len(data) > MAX_SAMPLES:
    data = data.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)

print(f"Dataset size used for t-SNE: {len(data)}")

# ---------------------------
# Encode labels
# ---------------------------
le = LabelEncoder()
data["label_id"] = le.fit_transform(data["label"])

# ---------------------------
# Load BERT model
# ---------------------------
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Using device:", device)

# ---------------------------
# Generate BERT embeddings
# ---------------------------
embeddings = []

BATCH_SIZE = 16

texts = data["text"].tolist()

with torch.no_grad():

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):

        batch_texts = texts[i:i+BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings.append(cls_embeddings.cpu().numpy())

X = np.vstack(embeddings)

print("Embedding shape:", X.shape)

# ---------------------------
# Apply t-SNE
# ---------------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    learning_rate='auto',
    init='pca'
)

X_tsne = tsne.fit_transform(X)

print("t-SNE completed.")

# ---------------------------
# Plot t-SNE
# ---------------------------
plt.figure(figsize=(10, 7))

labels = data["label_id"].values

for label in np.unique(labels):

    idx = labels == label

    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        label=le.inverse_transform([label])[0],
        alpha=0.7,
        s=10
    )

plt.title("t-SNE Visualization of Dataset1")
#plt.xlabel("t-SNE Dimension 1")
#plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)
save_path = os.path.join(OUTPUT_DIR, "Tsne_dataset1_BERT.pdf")
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)
plt.show()

!pip install tldextract

# The code plot T-SNE for phishing dataset

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Encode labels
# ---------------------------
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df["label"])

# ---------------------------
# Load tokenizer & model (feature extractor, NOT classifier)
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# Convert text to embeddings
# ---------------------------
def get_embeddings(texts, batch_size=32):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size].tolist()

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)

print("Extracting embeddings...")
features = get_embeddings(df["text"])

# ---------------------------
# Apply t-SNE
# ---------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(features)

# ---------------------------
# Plot
# ---------------------------
plt.figure()

scatter = plt.scatter(
    tsne_result[:, 0],
    tsne_result[:, 1],
    c=labels
)

plt.colorbar(scatter, ticks=[0, 1])
plt.title("t-SNE of Email Dataset (DistilBERT Embeddings)")
plt.xlabel("Phishing")
plt.ylabel("Legit")

plt.show()

#TRAIN New DATASETS on SAME MODEL
#DATA_PATH = "drive/MyDrive/Phishing_Email.csv"
OUTPUT_DIR = "drive/MyDrive/Phishing_email_2nd/distilbert_phish"
BEST_DIR = os.path.join(OUTPUT_DIR, "best_model")

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import joblib
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------
# Paths
# ---------------------------
OUTPUT_DIR = "drive/MyDrive/Phishing_email_2nd/distilbert_phish"
BEST_DIR = os.path.join(OUTPUT_DIR, "best_model")
label_encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

# ---------------------------
# Load datasets and merge
# ---------------------------
files = [
    "/mnt/data/CEAS_08.csv",
    "/mnt/data/Ling.csv",
    "/mnt/data/Nigerian_Fraud.csv",
]
files=[
    'drive/MyDrive/Phishing_email_2nd/SpamAssasin.csv',
    'drive/MyDrive/Phishing_email_2nd/Nigerian_Fraud.csv',
    'drive/MyDrive/Phishing_email_2nd/phishing_email.csv',
    'drive/MyDrive/Phishing_email_2nd/CEAS_08.csv',
    'drive/MyDrive/Phishing_email_2nd/Enron.csv',
    'drive/MyDrive/Phishing_email_2nd/Ling.csv',
    'drive/MyDrive/Phishing_email_2nd/Nazario.csv'
]
dfs = []
for f in files:
    df = pd.read_csv(f)
    # find text and label columns
    cols_lower = {c.lower(): c for c in df.columns}
    text_col = next((cols_lower[k] for k in ["email text", "email_text", "text", "body"] if k in cols_lower), None)
    label_col = next((cols_lower[k] for k in ["email type", "email_type", "label", "target"] if k in cols_lower), None)
    if text_col and label_col:
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data["label"] = data["label"].astype(str).str.strip().str.lower()

# Normalize labels
label_map = {
    "phishing email": "phishing", "phishing": "phishing", "spam": "phishing", "1": "phishing",
    "safe email": "legit", "safe": "legit", "legit": "legit", "ham": "legit", "0": "legit"
}
data["label"] = data["label"].map(lambda x: label_map.get(x, x))
data = data[data["label"].isin(["phishing", "legit"])].dropna(subset=["text"]).reset_index(drop=True)

print(f"Total merged dataset size: {len(data)}")

# Encode labels
le = LabelEncoder()
data["label_id"] = le.fit_transform(data["label"])
joblib.dump(le, label_encoder_path)

id2label = {i: lab for i, lab in enumerate(le.classes_)}
label2id = {lab: i for i, lab in id2label.items()}

# ---------------------------
# Train/validation split
# ---------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["text"].tolist(),
    data["label_id"].tolist(),
    test_size=0.2,
    stratify=data["label_id"],
    random_state=42
)

train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})

# ---------------------------
# Tokenization
# ---------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=["text"])
val_ds = val_ds.map(preprocess_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# Model
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_),
    id2label=id2label,
    label2id=label2id
).to(device)
'''
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)'''
# ---------------------------
# Training arguments
# ---------------------------
'''
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # one epoch
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
)'''
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    fp16=True,  # enable on GPU if supported
)
# Metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=label2id["phishing"])
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------------
# Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------------------------
# Train
# ---------------------------
trainer.train()

# Save best model
model.save_pretrained(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)

# ---------------------------
# Evaluation per dataset
# ---------------------------
results = []

def evaluate_dataset(file_path):
    print(f"\n=== Evaluating {file_path} ===")
    df = pd.read_csv(file_path)

    cols_lower = {c.lower(): c for c in df.columns}
    text_col = next((cols_lower[k] for k in ["email text", "email_text", "text", "body"] if k in cols_lower), None)
    label_col = next((cols_lower[k] for k in ["email type", "email_type", "label", "target"] if k in cols_lower), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find expected text/label columns in {file_path}")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map(lambda x: label_map.get(x, x))
    df = df[df["label"].isin(["phishing", "legit"])].dropna(subset=["text"]).reset_index(drop=True)

    y_true = le.transform(df["label"].tolist())

    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": y_true})
    ds = ds.map(preprocess_function, batched=True, remove_columns=["text"])

    preds_output = trainer.predict(ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=label2id["phishing"])

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(id2label.keys()))
    class_names = list(id2label.values())
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {os.path.basename(file_path)}")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    results.append({
        "dataset": os.path.basename(file_path),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

for f in files:
    evaluate_dataset(f)

# Save metrics to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False)
print("\nAll evaluation results saved to evaluation_results.csv")

# distilbert_finetune_sdn_full_fixed_multicsv_gpu.py

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import datetime
import tldextract
import whois
from dateutil import parser as dateparser
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from scipy.special import softmax
import joblib

# ---------------------------
# Paths and constants
# ---------------------------
DATA_DIR = "drive/MyDrive/Phishing_email_2nd"  # folder to scan for CSVs
CSV_PATHS = []  # optional explicit list, e.g., [".../file1.csv", ".../file2.csv"]
OUTPUT_DIR = "drive/MyDrive/Phishing_email_2nd/distilbert_phish"
BEST_DIR = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FPR = 0.005  # 0.5%
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# GPU/precision diagnostics
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())
    except Exception:
        pass

# ---------------------------
# Load and combine CSV datasets
# ---------------------------
def discover_csvs(data_dir: str):
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob.glob(pattern))

def load_and_normalize_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Identify columns (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    text_col = None
    for key in ["email text", "email_text", "text", "body", "content", "message", "email_body"]:
        if key in cols_lower:
            text_col = cols_lower[key]
            break
    label_col = None
    for key in ["email type", "email_type", "label", "target", "class", "category", "y"]:
        if key in cols_lower:
            label_col = cols_lower[key]
            break
    if text_col is None or label_col is None:
        raise ValueError(f"{path}: Could not find expected text/label columns. Found: {list(df.columns)}")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    label_map = {
        "phishing email": "phishing",
        "phishing": "phishing",
        "spam": "phishing",
        "1": "phishing",
        "true": "phishing",
        "malicious": "phishing",
        "safe email": "legit",
        "safe": "legit",
        "legit": "legit",
        "ham": "legit",
        "0": "legit",
        "false": "legit",
        "benign": "legit",
    }
    df["label"] = df["label"].map(lambda x: label_map.get(x, x))
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(["phishing", "legit"])]
    return df

def combine_all_csvs(data_dir: str, explicit_paths=None) -> pd.DataFrame:
    paths = explicit_paths if explicit_paths else discover_csvs(data_dir)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir} and no explicit CSV_PATHS provided.")
    dfs = []
    for p in paths:
        try:
            part = load_and_normalize_csv(p)
            part["__source__"] = os.path.basename(p)
            dfs.append(part)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    if not dfs:
        raise RuntimeError("No valid CSVs could be loaded after normalization.")
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"Combined dataset: {len(df_all)} rows from {len(dfs)} files.")
    print("Label distribution:\n", df_all["label"].value_counts())
    return df_all

df = combine_all_csvs(DATA_DIR, CSV_PATHS)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Example: assume df is your dataframe
# If you have labels, separate them
# features = df.drop('label', axis=1)
# labels = df['label']

features = df.values  # if no labels

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)

# Plot
plt.figure()
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title("t-SNE Visualization")
plt.xlabel("Phishing")
plt.ylabel("Legit")
plt.show()
# ---------------------------
# Split into train/val
# ---------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist(),
)

# Encode labels
le = LabelEncoder()
train_label_ids = le.fit_transform(train_labels)
val_label_ids = le.transform(val_labels)
id2label = {i: lab for i, lab in enumerate(le.classes_)}  # e.g., {0:'legit', 1:'phishing'}
label2id = {lab: i for i, lab in id2label.items()}
print("Encoder classes:", list(le.classes_))

# HF datasets
train_ds = Dataset.from_dict({"text": train_texts, "labels": train_label_ids})
val_ds = Dataset.from_dict({"text": val_texts, "labels": val_label_ids})
raw_datasets = DatasetDict({"train": train_ds, "validation": val_ds})

# ---------------------------
# Tokenizer and preprocessing
# ---------------------------
model_ckpt = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = raw_datasets.map(preprocess_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=label2id.get("phishing", 1), zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------------
# Model and Trainer
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # increase for real training
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),  # only if CUDA
    #bf16=(torch.cuda.is_available() or torch.cuda.is_bf16_supported()),  # optional
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print("Validation metrics:", metrics)

# ---------------------------
# Save BEFORE any inference loads
# ---------------------------
os.makedirs(BEST_DIR, exist_ok=True)
trainer.save_model(BEST_DIR)           # saves config with num_labels and label maps
tokenizer.save_pretrained(BEST_DIR)
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

# ---------------------------
# Inference setup (load saved)
# ---------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_infer_model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR).to(_device)
_infer_model.eval()
_infer_tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
_infer_label_encoder = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

print("Loaded from:", BEST_DIR)
print("Infer num_labels:", _infer_model.config.num_labels)
print("Infer label2id:", _infer_model.config.label2id)
print("Infer id2label:", _infer_model.config.id2label)
print("Infer model device:", next(_infer_model.parameters()).device)

# Derive 'phishing' index from model config (preferred), fallback to encoder
if _infer_model.config.label2id and "phishing" in _infer_model.config.label2id:
    PHISHING_IDX = int(_infer_model.config.label2id["phishing"])
else:
    if not hasattr(_infer_label_encoder, "classes_"):
        raise AttributeError("Loaded label encoder missing 'classes_'.")
    m = np.where(_infer_label_encoder.classes_ == "phishing")[0]
    if m.size == 0:
        raise ValueError(f"'phishing' not found in encoder classes: {list(_infer_label_encoder.classes_)}")
    PHISHING_IDX = int(m)
print("PHISHING_IDX:", PHISHING_IDX)

if _infer_model.config.num_labels != 2:
    raise RuntimeError(
        f"Expected 2 output labels, got {_infer_model.config.num_labels}. "
        "Ensure BEST_DIR points to your fine-tuned 2-class model (config.json must have num_labels==2)."
    )

# ---------------------------
# URL utilities and models
# ---------------------------
def extract_urls(text: str):
    if not text:
        return []
    return URL_REGEX.findall(text)

def url_lexical_features(url: str):
    u = url.strip()
    length = len(u)
    num_digits = sum(c.isdigit() for c in u)
    num_special = sum(c in "-_=?&%./:#" for c in u)
    at_symbol = 1 if "@" in u else 0
    https = 1 if u.lower().startswith("https") else 0
    dots = u.count(".")
    kw_list = ["login", "verify", "update", "secure", "confirm", "password", "unlock", "invoice"]
    kw_hit = int(any(k in u.lower() for k in kw_list))
    ext = tldextract.extract(u)
    domain = ".".join([ext.domain, ext.suffix]) if ext.suffix else ext.domain
    subdomain = ext.subdomain
    sub_len = len(subdomain) if subdomain else 0
    domain_age_days = -1
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            if not isinstance(created, datetime.datetime):
                created = dateparser.parse(str(created))
            domain_age_days = (datetime.datetime.utcnow() - created.replace(tzinfo=None)).days
    except Exception:
        pass
    return {
        "length": length,
        "num_digits": num_digits,
        "num_special": num_special,
        "at_symbol": at_symbol,
        "https": https,
        "dots": dots,
        "kw_hit": kw_hit,
        "sub_len": sub_len,
        "domain_age_days": domain_age_days,
    }

URL_FEATURE_KEYS = [
    "length","num_digits","num_special","at_symbol","https","dots",
    "kw_hit","sub_len","domain_age_days"
]

def vectorize_url_features(url_features_list):
    if not url_features_list:
        return np.zeros(len(URL_FEATURE_KEYS) * 2, dtype=np.float32)
    feats = np.array([[f[k] for k in URL_FEATURE_KEYS] for f in url_features_list], dtype=np.float32)
    feats_mean = feats.mean(axis=0)
    feats_max = feats.max(axis=0)
    return np.concatenate([feats_mean, feats_max]).astype(np.float32)

# URL text model (reusing the fine-tuned model by default)
MODEL_URL_CKPT = BEST_DIR  # replace with a proper 2-class URL model when available
_url_device = _device
_url_model = AutoModelForSequenceClassification.from_pretrained(MODEL_URL_CKPT).to(_url_device)
_url_model.eval()
_url_tokenizer = AutoTokenizer.from_pretrained(MODEL_URL_CKPT)

if _url_model.config.label2id and "phishing" in _url_model.config.label2id:
    URL_PHISH_IDX = int(_url_model.config.label2id["phishing"])
else:
    URL_PHISH_IDX = PHISHING_IDX
print("URL_PHISH_IDX:", URL_PHISH_IDX)
print("URL model device:", next(_url_model.parameters()).device)

def url_text_score(urls):
    if not urls:
        return None
    probs_list = []
    for u in urls:
        inputs = _url_tokenizer(u, return_tensors="pt", truncation=True).to(_url_device)
        with torch.no_grad():
            outputs = _url_model(**inputs)
        logits_np = outputs.logits.detach().cpu().numpy()  # expected (1, 2)
        if logits_np.ndim != 2 or logits_np.shape[1] != 2:
            continue  # skip non-2-class checkpoints
        probs = softmax(logits_np, axis=1)  # (2,)
        p = float(probs[URL_PHISH_IDX])
        probs_list.append(p)
    return float(np.mean(probs_list)) if probs_list else None

# Lexical scoring placeholder
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

_lex_weights = np.zeros(len(URL_FEATURE_KEYS) * 2, dtype=np.float32)
_lex_bias = 0.0

def url_lexical_score(features_vec: np.ndarray):
    v = features_vec.copy().astype(np.float32)
    v = np.clip(v, -1_000.0, 10_000.0)
    z = float(np.dot(v, _lex_weights) + _lex_bias)
    return float(sigmoid(z))  # returns 0..1

# ---------------------------
# Fusion and probability computation
# ---------------------------
def fused_phishing_probability(email_text: str):
    urls = extract_urls(email_text)

    # Body model probability
    inputs = _infer_tokenizer(email_text, return_tensors="pt", truncation=True).to(_device)
    with torch.no_grad():
        outputs = _infer_model(**inputs)

    logits_np = outputs.logits.detach().cpu().numpy()  # (1, num_labels)
    if logits_np.ndim != 2 or logits_np.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected logits shape: {logits_np.shape}. Expected (1,2). "
            f"Check {BEST_DIR}/config.json and ensure num_labels==2."
        )
    probs = softmax(logits_np, axis=1)  # (2,)
    p_body = float(probs[PHISHING_IDX])

    # URL text model probability
    p_url_text = url_text_score(urls)

    # URL lexical probability
    url_feats = [url_lexical_features(u) for u in urls]
    v = vectorize_url_features(url_feats)
    p_url_lex = url_lexical_score(v) if urls else None

    # Late-score averaging with availability-aware inclusion
    scores = [p_body]
    if p_url_text is not None:
        scores.append(p_url_text)
    if p_url_lex is not None:
        scores.append(p_url_lex)

    p_fused = float(np.mean(scores)) if scores else p_body
    return {
        "p_body": p_body,
        "p_url_text": p_url_text,
        "p_url_lex": p_url_lex,
        "p_fused": p_fused,
        "urls": urls,
    }

# ---------------------------
# ROC-based threshold calibration (validation)
# ---------------------------
val_texts_list = val_texts
val_labels_bin = (np.array(val_labels) == "phishing").astype(int)

val_probs = []
for t in val_texts_list:
    p = fused_phishing_probability(t)["p_fused"]
    val_probs.append(p)
val_probs = np.array(val_probs, dtype=np.float32)

fpr, tpr, thresholds = roc_curve(val_labels_bin, val_probs)
roc_auc = auc(fpr, tpr)
ok_idx = np.where(fpr <= TARGET_FPR)[0]
if len(ok_idx) == 0:
    best_idx = int(np.argmin(fpr))
else:
    best_idx = ok_idx[-1]
CALIBRATED_THRESHOLD = float(thresholds[best_idx])
print(f"Calibrated threshold for FPR<={TARGET_FPR*100:.2f}%: {CALIBRATED_THRESHOLD:.4f} (AUC={roc_auc:.4f})")

# ---------------------------
# Confusion matrix on validation set (body model baseline, thresholded)
# ---------------------------
pred_output = trainer.predict(tokenized["validation"])
probs_val = softmax(pred_output.predictions, axis=1)  # (N, 2)
phish_idx = label2id["phishing"]
y_pred_ids = (probs_val[:, phish_idx] >= 0.5).astype(int)
y_true_ids = pred_output.label_ids
class_names = list(id2label.values())
cm = confusion_matrix(y_true_ids, y_pred_ids, labels=list(id2label.keys()))

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - DistilBERT Phishing Classifier (Validation)")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_val.png")
plt.savefig(cm_path, dpi=150)
plt.show()
plt.close()
print(f"Saved confusion matrix to: {cm_path}")

print("Classification report (per-class):")
print(
    classification_report(
        y_true_ids,
        y_pred_ids,
        target_names=class_names,
        zero_division=0
    )
)

# ---------------------------
# SDN-style action function using calibrated risk
# ---------------------------
def sdn_action_for_email(text: str):
    fusion = fused_phishing_probability(text)
    p_phish = float(fusion["p_fused"])

    thr_allow = CALIBRATED_THRESHOLD
    thr_block = min(1.0, CALIBRATED_THRESHOLD + 0.20)

    if p_phish >= thr_block:
        action = "block"
    elif p_phish >= thr_allow:
        action = "monitor"
    else:
        action = "allow"

    return {
        "type": "phishing_detection",
        "risk_score_calibrated": round(p_phish, 4),
        "risk_components": {
            "body": round(fusion["p_body"], 4),
            "url_text": None if fusion["p_url_text"] is None else round(fusion["p_url_text"], 4),
            "url_lex": None if fusion["p_url_lex"] is None else round(fusion["p_url_lex"], 4),
        },
        "prediction": "phishing" if p_phish >= thr_allow else "legit",
        "action": action,
        "calibration": {
            "target_fpr": TARGET_FPR,
            "threshold": round(CALIBRATED_THRESHOLD, 4),
        },
        "indicators": {
            "has_url": bool(URL_REGEX.search(text)),
            "urls": fusion["urls"],
        },
    }

# ---------------------------
# Example
# ---------------------------
if __name__ == "__main__":
    example = "Urgent: Your account is locked. Verify now at http://secure-login-update.com"
    print(json.dumps(sdn_action_for_email(example), indent=2))