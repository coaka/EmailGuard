# =========================================================
# EmailGuard v2 — Multi-Agent Ensemble-Based Multi-Modal
# Email Phishing Detection Framework with SDN Enforcement
# =========================================================
# Dataset 1 (7 source files, merged):
#   SpamAssassin.csv  — sender, receiver, date, subject, body, urls, label
#   Nigerian_Fraud.csv — sender, receiver, date, subject, body, urls, label
#   phishing_email.csv — Email Text, Email Type
#   CEAS_08.csv       — sender, receiver, date, subject, body, urls, label
#   Enron.csv         — subject, body, label
#   Ling.csv          — subject, body, label
#   Nazario.csv       — sender, receiver, date, subject, body, urls, label
#
# Dataset 2 (PhiUSIIL_Phishing_URL_Dataset.csv):
#   54 pre-computed URL features (exact column list from the dataset),
#   used directly to train the Lexical & Structural Feature Agent.
# =========================================================

import os
import re
import math
import string
import logging
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset as TorchDataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger("EmailGuard")

# =========================================================
# CONFIG
# =========================================================

# Dataset 1 — the 7 source files (update paths as needed)
DATASET1_FILES = [
    "SpamAssassin.csv",
    "Nigerian_Fraud.csv",
    "phishing_email.csv",
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
]

# Dataset 2 — PhiUSIIL pre-computed feature CSV
DATASET2_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

OUTPUT_DIR = "./EmailGuard_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
N_FOLDS      = 4
VAL_FRACTION = 0.20      # 20 % of train fold → calibration-validation

# ── Semantic Content Agent (DistilBERT) ──────────────────
SEM_MODEL_NAME   = "distilbert-base-uncased"
SEM_MAX_LEN      = 512
SEM_BATCH_SIZE   = 32
SEM_EPOCHS       = 5
SEM_LR           = 2e-5
SEM_WEIGHT_DECAY = 0.01
SEM_DROPOUT      = 0.3
SEM_EARLY_STOP   = 2

# ── Textual Pattern Agent (BiLSTM) ───────────────────────
TXT_EMBED_DIM  = 128
TXT_HIDDEN_DIM = 64      # per direction → 128 after BiLSTM
TXT_MLP_HIDDEN = 128
TXT_DROPOUT    = 0.2
TXT_EPOCHS     = 20
TXT_BATCH_SIZE = 64
TXT_LR         = 1e-3
TXT_EARLY_STOP = 3
TXT_MIN_FREQ   = 3
TXT_MAX_VOCAB  = 30_000
TXT_MAX_SEQ    = 256

# ── Lexical & Structural Agent (Logistic Regression) ─────
LEX_C        = 1.0
LEX_SOLVER   = "lbfgs"
LEX_MAX_ITER = 1000

# ── Ensemble / calibration ───────────────────────────────
FUSION_WEIGHTS = (1/3, 1/3, 1/3)
ALPHA          = 0.005       # FPR ceiling (0.5 %)
DELTA_BLOCK    = 0.10        # τ_block = τ_cal + Δ_block

# =========================================================
# SECTION 1 — DATASET 1 LOADING & MERGING
# The 7 source files have two distinct schemas:
#
# Schema A (SpamAssassin, Nigerian_Fraud, CEAS_08, Nazario):
#   sender | receiver | date | subject | body | urls | label
#   label values: 0 = legitimate, 1 = phishing / spam
#
# Schema B (Enron, Ling):
#   subject | body | label
#   label values: 0 = legitimate, 1 = spam/phishing
#
# Schema C (phishing_email.csv):
#   Email Text | Email Type
#   Email Type: "Phishing Email" or "Safe Email"
#
# All schemas are normalised to two columns: text, y
# Text = subject (if present) + " " + body
# =========================================================

# Normalise label values to 0 / 1
LABEL_MAP = {
    "phishing email": 1, "phishing": 1, "spam": 1,
    "1": 1, 1: 1, True: 1,
    "safe email": 0, "legit": 0, "ham": 0, "safe": 0,
    "legitimate": 0, "0": 0, 0: 0, False: 0,
}


def preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords."""
    STOPWORDS = {
        "a","an","the","and","or","but","in","on","at","to","for",
        "of","is","it","its","this","that","with","as","by","from",
        "was","are","be","been","have","has","had","not","do","did",
        "will","would","could","should","may","might","shall","can"
    }
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(
        t for t in text.split() if t not in STOPWORDS and len(t) > 1
    )


def _load_one(path: str) -> pd.DataFrame:
    """
    Load a single Dataset 1 source file and normalise it to (text, y).
    Handles all three schemas automatically by column detection.
    """
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]

    # ── Schema C: phishing_email.csv ─────────────────────
    if "email text" in df.columns and "email type" in df.columns:
        df = df[["email text", "email type"]].copy()
        df.columns = ["text", "raw_label"]

    # ── Schema A: has both subject + body columns ─────────
    elif "body" in df.columns and "subject" in df.columns:
        df["text"] = (
            df["subject"].fillna("") + " " + df["body"].fillna("")
        ).str.strip()
        # label column
        label_col = next(
            (c for c in df.columns if c in ("label", "labels", "class",
                                             "type", "category")), None
        )
        df["raw_label"] = df[label_col] if label_col else 0
        df = df[["text", "raw_label"]].copy()

    # ── Schema B: body only, no subject ──────────────────
    elif "body" in df.columns:
        df["text"] = df["body"].fillna("")
        label_col = next(
            (c for c in df.columns if c in ("label", "labels", "class",
                                             "type", "category")), None
        )
        df["raw_label"] = df[label_col] if label_col else 0
        df = df[["text", "raw_label"]].copy()

    else:
        log.warning("Unrecognised schema in %s — skipping", path)
        return pd.DataFrame(columns=["text", "y"])

    # ── Normalise label to 0 / 1 ─────────────────────────
    df["y"] = (
        df["raw_label"]
        .astype(str).str.strip().str.lower()
        .map(lambda v: LABEL_MAP.get(v, LABEL_MAP.get(
            int(v) if v.lstrip("-").isdigit() else v, np.nan
        )))
    )
    df = df[["text", "y"]].dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    log.info(
        "  Loaded %-30s  %5d rows  (%d phishing, %d legit)",
        os.path.basename(path), len(df), df.y.sum(), (df.y == 0).sum()
    )
    return df


def load_dataset1(file_paths: list) -> pd.DataFrame:
    """Merge all 7 source files into a single normalised DataFrame."""
    log.info("=== Loading Dataset 1 (7 source files) ===")
    parts = []
    for p in file_paths:
        if not os.path.exists(p):
            log.warning("File not found, skipping: %s", p)
            continue
        parts.append(_load_one(p))

    df = pd.concat(parts, ignore_index=True).dropna()
    df["text"] = df["text"].apply(preprocess_text)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    log.info(
        "Dataset 1 merged: %d instances  (%d phishing, %d legit)",
        len(df), df.y.sum(), (df.y == 0).sum()
    )
    return df

# =========================================================
# SECTION 2 — DATASET 2 LOADING (PhiUSIIL pre-computed features)
#
# The CSV already contains 54 engineered URL features.
# We select only the numeric feature columns (excluding
# metadata columns: FILENAME, URL, Domain, TLD, Title)
# and the label column.
#
# Full column list (from the dataset specification):
#   FILENAME, URL, URLLength, Domain, DomainLength, IsDomainIP,
#   TLD, URLSimilarityIndex, CharContinuationRate, TLDLegitimateProb,
#   URLCharProb, TLDLength, NoOfSubDomain, HasObfuscation,
#   NoOfObfuscatedChar, ObfuscationRatio, NoOfLettersInURL,
#   LetterRatioInURL, NoOfDegitsInURL, DegitRatioInURL,
#   NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL,
#   NoOfOtherSpecialCharsInURL, SpacialCharRatioInURL, IsHTTPS,
#   LineOfCode, LargestLineLength, HasTitle, Title,
#   DomainTitleMatchScore, URLTitleMatchScore, HasFavicon, Robots,
#   IsResponsive, NoOfURLRedirect, NoOfSelfRedirect, HasDescription,
#   NoOfPopup, NoOfiFrame, HasExternalFormSubmit, HasSocialNet,
#   HasSubmitButton, HasHiddenFields, HasPasswordField, Bank, Pay,
#   Crypto, HasCopyrightInfo, NoOfImage, NoOfCSS, NoOfJS,
#   NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef, label
# =========================================================

# Columns to drop before feature extraction (non-numeric / metadata)
D2_DROP_COLS = {
    "filename", "url", "domain", "tld", "title",
}

# The target column in Dataset 2
D2_LABEL_COL = "label"


def load_dataset2(path: str) -> tuple:
    """
    Load PhiUSIIL dataset and return (X, y) where X is a float32 array
    of all numeric feature columns and y is binary (1=phishing, 0=legit).
    """
    log.info("=== Loading Dataset 2 (PhiUSIIL) from %s ===", path)
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    # Separate label
    y = df[D2_LABEL_COL].astype(int).values

    # Drop non-numeric / metadata columns
    feature_cols = [
        c for c in df.columns
        if c.lower() not in D2_DROP_COLS
        and c != D2_LABEL_COL
    ]

    # Keep only columns that can be coerced to numeric
    numeric_cols = []
    for c in feature_cols:
        try:
            pd.to_numeric(df[c], errors="raise")
            numeric_cols.append(c)
        except (ValueError, TypeError):
            pass

    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce") \
                         .fillna(0).values.astype(np.float32)

    log.info(
        "Dataset 2 loaded: %d instances, %d features  "
        "(%d phishing, %d legit)",
        len(y), X.shape[1], int(y.sum()), int((y == 0).sum())
    )
    log.info("Feature columns used (%d): %s", len(numeric_cols),
             numeric_cols)
    return X, y, numeric_cols

# =========================================================
# SECTION 3 — LEXICAL & STRUCTURAL FEATURE AGENT
# Uses Dataset 2's pre-computed features directly.
# Wrapped in a class that also supports live inference
# by computing a minimal 5-dim fallback if no Dataset 2
# features are available at inference time.
# =========================================================

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def extract_urls(text: str) -> list:
    return URL_REGEX.findall(str(text))


def _fallback_url_features(url: str) -> np.ndarray:
    """
    Compute a minimal set of URL features for live inference
    when pre-computed Dataset 2 features are not available.
    Returns a vector whose length matches the number of
    features the LexicalAgent was trained on (padded with zeros
    if fewer features can be computed).
    """
    u = url.lower()
    # 5 basic features always computable
    feats = [
        len(url),
        sum(c.isdigit() for c in url),
        sum(c in "-_?=&%./@!~" for c in url),
        url.count("."),
        1 if u.startswith("https") else 0,
    ]
    return np.array(feats, dtype=np.float32)


class LexicalAgent:
    """
    Logistic Regression on Dataset 2's pre-computed feature vectors.
    StandardScaler fitted on training fold; L2, C=1.0, lbfgs.
    """

    def __init__(self, n_features: int = None):
        self.scaler     = StandardScaler()
        self.model      = LogisticRegression(
            penalty="l2", C=LEX_C,
            solver=LEX_SOLVER, max_iter=LEX_MAX_ITER,
            random_state=SEED
        )
        self.n_features = n_features   # set after first fit

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score_from_urls(self, text: str) -> float:
        """
        Live-inference path: extract URLs from email text,
        compute fallback features, pad/truncate to n_features,
        and return p_lexical.
        """
        urls = extract_urls(text)
        if not urls:
            # No URLs → neutral prior (0.5)
            return 0.5
        feats = np.stack([_fallback_url_features(u) for u in urls])
        agg   = np.concatenate([feats.mean(axis=0), feats.max(axis=0)])
        # Pad or truncate to match training dimensionality
        n = self.n_features or len(agg)
        if len(agg) < n:
            agg = np.pad(agg, (0, n - len(agg)))
        else:
            agg = agg[:n]
        X_scaled = self.scaler.transform(agg.reshape(1, -1))
        return float(self.model.predict_proba(X_scaled)[0][1])

    def save(self, path: str):
        joblib.dump({
            "scaler":     self.scaler,
            "model":      self.model,
            "n_features": self.n_features,
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj  = cls(n_features=data["n_features"])
        obj.scaler = data["scaler"]
        obj.model  = data["model"]
        return obj

# =========================================================
# SECTION 4 — TEXTUAL PATTERN AGENT (BiLSTM)
# =========================================================

class Vocabulary:
    PAD, UNK = 0, 1

    def __init__(self):
        self.w2i = {"<PAD>": 0, "<UNK>": 1}
        self.i2w = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts: list):
        freq  = Counter(w for t in texts for w in t.split())
        vocab = [w for w, c in freq.most_common(TXT_MAX_VOCAB)
                 if c >= TXT_MIN_FREQ]
        for idx, w in enumerate(vocab, start=2):
            self.w2i[w] = idx
            self.i2w[idx] = w
        log.info("Vocabulary built: %d tokens", len(self.w2i))

    def encode(self, text: str) -> list:
        ids = [self.w2i.get(w, self.UNK)
               for w in text.split()][:TXT_MAX_SEQ]
        ids += [self.PAD] * (TXT_MAX_SEQ - len(ids))
        return ids

    def __len__(self):
        return len(self.w2i)


class _EmailDS(TorchDataset):
    def __init__(self, texts, labels, vocab):
        self.ids    = [torch.tensor(vocab.encode(t), dtype=torch.long)
                       for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    """
    embedding_dim=128, 2 bidirectional LSTM layers of 64 units,
    mean pooling, MLP 128→ReLU→Dropout(0.2)→2.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, TXT_EMBED_DIM, padding_idx=Vocabulary.PAD
        )
        self.bilstm = nn.LSTM(
            input_size    = TXT_EMBED_DIM,
            hidden_size   = TXT_HIDDEN_DIM,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = TXT_DROPOUT,
        )
        self.mlp = nn.Sequential(
            nn.Linear(TXT_HIDDEN_DIM * 2, TXT_MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(TXT_DROPOUT),
            nn.Linear(TXT_MLP_HIDDEN, 2),
        )

    def forward(self, x):
        emb    = self.embedding(x)
        out, _ = self.bilstm(emb)
        return self.mlp(out.mean(dim=1))


class TextualPatternAgent:
    def __init__(self):
        self.vocab = Vocabulary()
        self.net   = None

    def fit(self, train_texts, train_labels,
            val_texts=None, val_labels=None):
        self.vocab.build(list(train_texts))
        train_dl = DataLoader(
            _EmailDS(train_texts, train_labels, self.vocab),
            batch_size=TXT_BATCH_SIZE, shuffle=True
        )
        self.net = BiLSTMClassifier(len(self.vocab)).to(DEVICE)
        opt  = torch.optim.Adam(self.net.parameters(), lr=TXT_LR)
        crit = nn.CrossEntropyLoss()

        best_loss, patience, best_state = float("inf"), 0, None
        for epoch in range(1, TXT_EPOCHS + 1):
            self.net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                crit(self.net(xb), yb).backward()
                opt.step()

            if val_texts is not None:
                vl = self._val_loss(val_texts, val_labels, crit)
                if vl < best_loss:
                    best_loss, patience = vl, 0
                    best_state = {k: v.clone()
                                  for k, v in self.net.state_dict().items()}
                else:
                    patience += 1
                    if patience >= TXT_EARLY_STOP:
                        log.info("Textual Agent early stop at epoch %d",
                                 epoch)
                        break

        if best_state:
            self.net.load_state_dict(best_state)

    def _val_loss(self, texts, labels, crit):
        dl = DataLoader(
            _EmailDS(texts, labels, self.vocab),
            batch_size=TXT_BATCH_SIZE
        )
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += crit(self.net(xb), yb).item()
        return total / len(dl)

    def predict_proba(self, texts) -> np.ndarray:
        dl = DataLoader(
            _EmailDS(texts, np.zeros(len(texts), int), self.vocab),
            batch_size=TXT_BATCH_SIZE
        )
        self.net.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in dl:
                logits = self.net(xb.to(DEVICE))
                probs.append(
                    torch.softmax(logits, dim=1).cpu().numpy()
                )
        return np.concatenate(probs, axis=0)

    def score(self, text: str) -> float:
        return float(self.predict_proba([text])[0][1])

    def save(self, path: str):
        joblib.dump({"vocab": self.vocab,
                     "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj  = cls()
        obj.vocab = data["vocab"]
        obj.net   = BiLSTMClassifier(len(obj.vocab)).to(DEVICE)
        obj.net.load_state_dict(data["state"])
        return obj

# =========================================================
# SECTION 5 — SEMANTIC CONTENT AGENT (DistilBERT)
# =========================================================

class _HFDS(TorchDataset):
    def __init__(self, encodings, labels):
        self.enc    = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class SemanticContentAgent:
    """
    DistilBERT-base-uncased + 2-layer head:
    768→256→ReLU→Dropout(0.3)→2→softmax
    AdamW lr=2e-5, wd=0.01, warmup=10%, batch=32,
    max_epochs=5, early_stop patience=2.
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(SEM_MODEL_NAME)
        self.model     = None
        self._path     = None

    def _tok(self, texts):
        return self.tokenizer(
            list(texts), truncation=True,
            padding=True, max_length=SEM_MAX_LEN
        )

    def fit(self, train_texts, train_labels,
            val_texts=None, val_labels=None, fold_id=0):
        train_ds = _HFDS(self._tok(train_texts), list(train_labels))
        val_ds   = (_HFDS(self._tok(val_texts), list(val_labels))
                    if val_texts is not None else None)

        self._path = os.path.join(
            OUTPUT_DIR, f"semantic_fold{fold_id}"
        )
        steps        = math.ceil(len(train_ds) / SEM_BATCH_SIZE)
        warmup_steps = int(0.10 * steps * SEM_EPOCHS)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            SEM_MODEL_NAME,
            num_labels          = 2,
            hidden_dropout_prob = SEM_DROPOUT,
        ).to(DEVICE)

        args = TrainingArguments(
            output_dir                  = self._path,
            learning_rate               = SEM_LR,
            weight_decay                = SEM_WEIGHT_DECAY,
            per_device_train_batch_size = SEM_BATCH_SIZE,
            num_train_epochs            = SEM_EPOCHS,
            warmup_steps                = warmup_steps,
            evaluation_strategy         = "epoch" if val_ds else "no",
            save_strategy               = "epoch" if val_ds else "no",
            load_best_model_at_end      = val_ds is not None,
            metric_for_best_model       = "loss",
            greater_is_better           = False,
            report_to                   = "none",
            seed                        = SEED,
        )
        callbacks = (
            [EarlyStoppingCallback(early_stopping_patience=SEM_EARLY_STOP)]
            if val_ds else []
        )
        Trainer(
            model         = self.model,
            args          = args,
            train_dataset = train_ds,
            eval_dataset  = val_ds,
            tokenizer     = self.tokenizer,
            callbacks     = callbacks,
        ).train()

    def predict_proba(self, texts) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(texts), SEM_BATCH_SIZE):
            enc = self.tokenizer(
                list(texts[i:i+SEM_BATCH_SIZE]),
                return_tensors="pt",
                truncation=True, padding=True,
                max_length=SEM_MAX_LEN
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(**enc).logits
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def score(self, text: str) -> float:
        return float(self.predict_proba(np.array([text]))[0][1])

    def save(self, path: str):
        if self._path and os.path.isdir(self._path):
            import shutil
            shutil.copytree(self._path, path, dirs_exist_ok=True)

    @classmethod
    def load(cls, path: str):
        obj           = cls.__new__(cls)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model     = AutoModelForSequenceClassification.from_pretrained(
            path
        ).to(DEVICE)
        obj._path     = path
        return obj

# =========================================================
# SECTION 6 — CALIBRATED LATE-FUSION ENSEMBLE
# =========================================================

def late_fusion(p_sem: float, p_txt: float, p_lex: float,
                weights=FUSION_WEIGHTS) -> float:
    """Equation 1: p_fused = w_s·p_sem + w_t·p_txt + w_l·p_lex"""
    w_s, w_t, w_l = weights
    assert abs(w_s + w_t + w_l - 1.0) < 1e-6
    return w_s * p_sem + w_t * p_txt + w_l * p_lex


def roc_guided_calibration(scores: np.ndarray,
                            labels: np.ndarray,
                            alpha: float = ALPHA) -> float:
    """
    Algorithm 1 / Equations 2-3:
    τ_cal = argmax_τ TPR(τ)  s.t.  FPR(τ) ≤ α
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores, pos_label=1)
    best_tau, best_tpr = thresholds[-1], 0.0
    for fpr_v, tpr_v, tau in zip(fpr_arr, tpr_arr, thresholds):
        if fpr_v <= alpha and tpr_v > best_tpr:
            best_tpr, best_tau = tpr_v, tau
    log.info("Calibration: τ_cal=%.4f  best_TPR=%.4f", best_tau, best_tpr)
    return float(best_tau)

# =========================================================
# SECTION 7 — CROSS-VALIDATION PIPELINE
# =========================================================

def compute_metrics(y_true, y_pred, y_scores, tau_cal):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(
        accuracy=round(acc, 4), precision=round(prec, 4),
        recall=round(rec, 4),   f1=round(f1, 4),
        auc=round(auc, 4),      fpr=round(fpr * 100, 2),
        tau_cal=round(tau_cal, 4),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
    )


def train_evaluate_fold(
    fold_id:      int,
    train_texts:  np.ndarray,
    train_labels: np.ndarray,
    test_texts:   np.ndarray,
    test_labels:  np.ndarray,
    X_url_train:  np.ndarray,   # Dataset 2 features for this fold's train
    y_url_train:  np.ndarray,
) -> dict:
    log.info("=== Fold %d ===", fold_id)

    # ── 80/20 within-fold split ────────────────────────────
    n      = len(train_texts)
    n_val  = int(n * VAL_FRACTION)
    rng    = np.random.RandomState(SEED + fold_id)
    idx    = rng.permutation(n)
    vi, fi = idx[:n_val], idx[n_val:]

    fit_t,  fit_l  = train_texts[fi],  train_labels[fi]
    val_t,  val_l  = train_texts[vi],  train_labels[vi]

    log.info("  model-train=%d  cal-val=%d  test=%d",
             len(fit_t), len(val_t), len(test_texts))

    # ── train Semantic Agent ───────────────────────────────
    sem = SemanticContentAgent()
    sem.fit(fit_t, fit_l, val_texts=val_t, val_labels=val_l,
            fold_id=fold_id)

    # ── train Textual Agent ────────────────────────────────
    txt = TextualPatternAgent()
    txt.fit(fit_t, fit_l, val_texts=val_t, val_labels=val_l)

    # ── train Lexical Agent on Dataset 2 train split ───────
    lex = LexicalAgent()
    lex.fit(X_url_train, y_url_train)

    # ── calibration-validation fused scores ───────────────
    p_sem_v = sem.predict_proba(val_t)[:, 1]
    p_txt_v = txt.predict_proba(val_t)[:, 1]
    p_lex_v = np.array([lex.score_from_urls(t) for t in val_t])
    p_fused_v = np.array([
        late_fusion(s, t, l)
        for s, t, l in zip(p_sem_v, p_txt_v, p_lex_v)
    ])

    # ── Algorithm 1: ROC-guided calibration ───────────────
    tau_cal   = roc_guided_calibration(p_fused_v, val_l)
    tau_block = min(1.0, tau_cal + DELTA_BLOCK)

    # ── test-fold evaluation ───────────────────────────────
    p_sem_te = sem.predict_proba(test_texts)[:, 1]
    p_txt_te = txt.predict_proba(test_texts)[:, 1]
    p_lex_te = np.array([lex.score_from_urls(t) for t in test_texts])
    p_fused_te = np.array([
        late_fusion(s, t, l)
        for s, t, l in zip(p_sem_te, p_txt_te, p_lex_te)
    ])

    y_pred  = (p_fused_te >= tau_cal).astype(int)
    metrics = compute_metrics(test_labels, y_pred, p_fused_te, tau_cal)
    log.info("  Fold %d results: %s", fold_id, metrics)

    # ── save models and thresholds ─────────────────────────
    sem.save(os.path.join(OUTPUT_DIR, f"semantic_fold{fold_id}"))
    txt.save(os.path.join(OUTPUT_DIR, f"textual_fold{fold_id}.joblib"))
    lex.save(os.path.join(OUTPUT_DIR, f"lexical_fold{fold_id}.joblib"))
    joblib.dump({"tau_cal": tau_cal, "tau_block": tau_block},
                os.path.join(OUTPUT_DIR,
                             f"thresholds_fold{fold_id}.joblib"))
    return metrics


def run_cross_validation(df: pd.DataFrame,
                         X_url: np.ndarray,
                         y_url: np.ndarray) -> list:
    """
    Stratified 4-fold CV on Dataset 1 (emails).
    Dataset 2 (URLs) is split with the same fold indices so
    each fold's Lexical Agent trains on 75% of Dataset 2.
    """
    texts  = df["text"].values
    labels = df["y"].values

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)
    skf_url = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)

    url_splits = list(skf_url.split(X_url, y_url))
    results    = []

    for fold_id, (tr_idx, te_idx) in enumerate(
            skf.split(texts, labels), 1):
        url_tr_idx, _ = url_splits[fold_id - 1]
        metrics = train_evaluate_fold(
            fold_id      = fold_id,
            train_texts  = texts[tr_idx],
            train_labels = labels[tr_idx],
            test_texts   = texts[te_idx],
            test_labels  = labels[te_idx],
            X_url_train  = X_url[url_tr_idx],
            y_url_train  = y_url[url_tr_idx],
        )
        results.append(metrics)

    # ── aggregate summary ──────────────────────────────────
    log.info("=== Cross-Validation Summary ===")
    for k in ["accuracy", "precision", "recall", "f1", "fpr", "auc"]:
        vals = [r[k] for r in results]
        log.info("  %-12s %.4f ± %.4f", k.upper(),
                 np.mean(vals), np.std(vals))
    # Pooled confusion matrix
    log.info("  Pooled — TP=%d  FN=%d  FP=%d  TN=%d",
             sum(r["TP"] for r in results),
             sum(r["FN"] for r in results),
             sum(r["FP"] for r in results),
             sum(r["TN"] for r in results))
    return results

# =========================================================
# SECTION 8 — SIMULATED SDN ENFORCEMENT LAYER
# =========================================================

def sdn_enforce(text:       str,
                sem:        SemanticContentAgent,
                txt:        TextualPatternAgent,
                lex:        LexicalAgent,
                tau_cal:    float,
                tau_block:  float) -> dict:
    """
    Algorithm 2 — simulated SDN tiered enforcement.
    score >= tau_block  →  BLOCK   (quarantine)
    score >= tau_cal    →  MONITOR (mirror + tag)
    score <  tau_cal    →  ALLOW   (deliver)
    """
    p_sem    = sem.score(text)
    p_txt    = txt.score(text)
    p_lex    = lex.score_from_urls(text)
    p_fused  = late_fusion(p_sem, p_txt, p_lex)

    if p_fused >= tau_block:
        action, sdn = "BLOCK",   "quarantine_via_simulated_southbound_API"
    elif p_fused >= tau_cal:
        action, sdn = "MONITOR", "mirror_to_analysis_vlan_tag_for_review"
    else:
        action, sdn = "ALLOW",   "deliver_without_intervention"

    return {
        "p_fused":    round(p_fused, 4),
        "p_semantic": round(p_sem,   4),
        "p_textual":  round(p_txt,   4),
        "p_lexical":  round(p_lex,   4),
        "tau_cal":    round(tau_cal,   4),
        "tau_block":  round(tau_block, 4),
        "action":     action,
        "sdn_action": sdn,
        "timestamp":  datetime.datetime.utcnow().isoformat(),
    }

# =========================================================
# SECTION 9 — MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":

    # ── Dataset 1: load and merge 7 source files ──────────
    df_email = load_dataset1(DATASET1_FILES)

    # ── Dataset 2: load pre-computed PhiUSIIL features ────
    X_url, y_url, url_feature_cols = load_dataset2(DATASET2_PATH)

    # ── Run stratified 4-fold cross-validation ────────────
    cv_results = run_cross_validation(df_email, X_url, y_url)

    # ── Demo: SDN enforcement on a sample email ───────────
    log.info("=== Demo inference (Fold 1 models) ===")

    thresholds = joblib.load(
        os.path.join(OUTPUT_DIR, "thresholds_fold1.joblib")
    )
    sem_demo = SemanticContentAgent.load(
        os.path.join(OUTPUT_DIR, "semantic_fold1")
    )
    txt_demo = TextualPatternAgent.load(
        os.path.join(OUTPUT_DIR, "textual_fold1.joblib")
    )
    lex_demo = LexicalAgent.load(
        os.path.join(OUTPUT_DIR, "lexical_fold1.joblib")
    )

    sample = preprocess_text(
        "Urgent: Your bank account has been locked. "
        "Please verify immediately: http://secure-login-verify.com"
        "/confirm?user=you&token=abc123"
    )

    result = sdn_enforce(
        text      = sample,
        sem       = sem_demo,
        txt       = txt_demo,
        lex       = lex_demo,
        tau_cal   = thresholds["tau_cal"],
        tau_block = thresholds["tau_block"],
    )

    print("\n=== EmailGuard SDN Enforcement Result ===")
    for k, v in result.items():
        print(f"  {k:<16}: {v}")
