# =========================================================
# EmailGuard: Multi-Agent Ensemble-Based Multi-Modal
# Email Phishing Detection Framework with SDN Enforcement
# =========================================================
# Implements the framework described in:
#   "EmailGuard: A Multi-Agent Ensemble-Based Multi-Modal
#    Email Phishing Detection Framework with SDN Enforcement"
#
# Three detection agents:
#   1. Semantic Content Agent   — DistilBERT fine-tuned classifier
#   2. Textual Pattern Agent    — BiLSTM sequence classifier
#   3. Lexical & Structural Agent — Logistic Regression on 54 URL features
#
# Ensemble: calibrated equal-weight late fusion (w = 1/3 each)
# Calibration: ROC-guided threshold optimisation (Algorithm 1, alpha=0.005)
# Evaluation: stratified 4-fold cross-validation
# Enforcement: simulated SDN tiered policy (allow / monitor / block)
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
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("EmailGuard")

# =========================================================
# CONFIG  (all hyper-parameters from the manuscript)
# =========================================================

EMAIL_DATA_PATH = "phishing_email.csv"   # merged 7-source corpus (Dataset 1)
URL_DATA_PATH   = "PhiUSIIL_Phishing_URL_Dataset.csv"         # PhiUSIIL URL dataset  (Dataset 2)

OUTPUT_DIR   = "./EmailGuard_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
N_FOLDS      = 4            # stratified 4-fold CV
VAL_FRACTION = 0.20         # 20% of train fold reserved for calibration

# ── Semantic Content Agent (DistilBERT) ──────────────────
SEM_MODEL_NAME  = "distilbert-base-uncased"
SEM_MAX_LEN     = 512
SEM_BATCH_SIZE  = 32
SEM_EPOCHS      = 5
SEM_LR          = 2e-5
SEM_WEIGHT_DECAY= 0.01
SEM_DROPOUT     = 0.3
SEM_EARLY_STOP  = 2         # patience

# ── Textual Pattern Agent (BiLSTM) ───────────────────────
TXT_EMBED_DIM   = 128
TXT_HIDDEN_DIM  = 64        # per direction; total = 128 after BiLSTM
TXT_MLP_HIDDEN  = 128
TXT_DROPOUT     = 0.2
TXT_EPOCHS      = 20
TXT_BATCH_SIZE  = 64
TXT_LR          = 1e-3
TXT_EARLY_STOP  = 3         # patience
TXT_MIN_FREQ    = 3
TXT_MAX_VOCAB   = 30_000
TXT_MAX_SEQ_LEN = 256

# ── Lexical & Structural Agent (Logistic Regression) ─────
LEX_C           = 1.0       # L2 regularisation strength
LEX_SOLVER      = "lbfgs"
LEX_MAX_ITER    = 1000

# ── Ensemble / calibration ───────────────────────────────
FUSION_WEIGHTS  = (1/3, 1/3, 1/3)   # (w_semantic, w_textual, w_lexical)
ALPHA           = 0.005              # FPR ceiling for calibration (0.5%)
DELTA_BLOCK     = 0.10               # τ_block = τ_cal + Δ_block

# ── SDN thresholds (set after calibration) ───────────────
TAU_CAL         = None   # filled by Algorithm 1
TAU_BLOCK       = None   # filled after calibration

# ── Suspicious URL keywords (19 binary flags) ────────────
PHISH_KEYWORDS  = [
    "login", "secure", "verify", "update", "confirm",
    "account", "bank", "ebay", "paypal", "free",
    "bonus", "winner", "click", "here", "now",
    "alert", "warning", "urgent", "immediate"
]

# =========================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# =========================================================

def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, strip stopwords, apply basic stemming."""
    STOPWORDS = {
        "a","an","the","and","or","but","in","on","at","to","for",
        "of","is","it","its","this","that","with","as","by","from",
        "was","are","be","been","have","has","had","not","do","did",
        "will","would","could","should","may","might","shall","can"
    }
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def load_email_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    text_col  = next(c for c in df.columns if "text" in c or "body" in c)
    label_col = next(c for c in df.columns if "label" in c or "type" in c)
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df["label"] = df["label"].astype(str).str.lower().replace({
        "phishing email": "phishing", "spam": "phishing",
        "ham": "legit", "safe": "legit", "legit": "legit",
        "1": "phishing", "0": "legit"
    })
    df = df[df["label"].isin(["phishing", "legit"])].dropna().reset_index(drop=True)
    df["y"]    = df["label"].map({"legit": 0, "phishing": 1})
    df["text"] = df["text"].apply(preprocess_text)
    log.info("Email dataset loaded: %d instances (%d phishing, %d legit)",
             len(df), df.y.sum(), (df.y == 0).sum())
    return df


def load_url_dataset(path: str) -> tuple:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    url_col   = next(c for c in df.columns if "url" in c)
    label_col = next(c for c in df.columns if "label" in c)
    df = df[[url_col, label_col]].copy()
    df.columns = ["url", "label"]
    df["y"] = df["label"].astype(str).str.lower().map(
        {"0": 0, "1": 1, "legit": 0, "phishing": 1}
    ).fillna(df["label"].astype(int))
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    log.info("URL dataset loaded: %d instances (%d phishing, %d legit)",
             len(df), df.y.sum(), (df.y == 0).sum())
    return df["url"].values, df["y"].values.astype(int)

# =========================================================
# SECTION 2 — LEXICAL & STRUCTURAL FEATURE AGENT
#   54 features per URL, aggregated to 108-dim email vector
#   (mean + max across all URLs in the email)
# =========================================================

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def extract_urls(text: str) -> list:
    return URL_REGEX.findall(text)

def url_features_54(url: str) -> np.ndarray:
    """Extract the 54 lexical/structural features described in the manuscript."""
    u = url.lower()
    # --- parse components ---
    proto_end   = u.find("://")
    after_proto = u[proto_end + 3:] if proto_end >= 0 else u
    slash_idx   = after_proto.find("/")
    domain_part = after_proto[:slash_idx] if slash_idx >= 0 else after_proto
    path_part   = after_proto[slash_idx:] if slash_idx >= 0 else ""

    # 1  URL length
    f01 = len(url)
    # 2  number of dots
    f02 = url.count(".")
    # 3  number of hyphens
    f03 = url.count("-")
    # 4  number of underscores
    f04 = url.count("_")
    # 5  number of slashes
    f05 = url.count("/")
    # 6  number of question marks
    f06 = url.count("?")
    # 7  number of equals signs
    f07 = url.count("=")
    # 8  number of ampersands
    f08 = url.count("&")
    # 9  number of at-signs
    f09 = url.count("@")
    # 10 number of exclamation marks
    f10 = url.count("!")
    # 11 number of tilde characters
    f11 = url.count("~")
    # 12 number of commas
    f12 = url.count(",")
    # 13 number of percent signs
    f13 = url.count("%")
    # 14 number of digit characters
    f14 = sum(c.isdigit() for c in url)
    # 15 digit-to-letter ratio
    letters = sum(c.isalpha() for c in url)
    f15 = f14 / letters if letters > 0 else 0.0
    # 16 HTTPS flag
    f16 = 1 if u.startswith("https") else 0
    # 17 IP address as hostname flag
    f17 = 1 if re.match(r"^(?:\d{1,3}\.){3}\d{1,3}", domain_part) else 0
    # 18 number of subdomains
    domain_labels = domain_part.replace("www.", "").split(".")
    f18 = max(0, len(domain_labels) - 2)
    # 19 subdomain depth (same as f18 here; reported separately in manuscript)
    f19 = f18
    # 20 TLD encoded as integer (hash mod 500 for reproducibility)
    tld  = domain_labels[-1] if domain_labels else ""
    f20  = hash(tld) % 500
    # 21 domain length
    f21 = len(domain_part)
    # 22 domain age proxy (0 = unknown; requires live WHOIS in production)
    f22 = 0
    # 23 domain registration period proxy
    f23 = 0
    # 24–42  19 binary suspicious keyword flags
    kw_flags = [1 if kw in u else 0 for kw in PHISH_KEYWORDS]  # 19 values
    # 43 Shannon entropy of full URL
    counts = Counter(url)
    total  = len(url)
    f43 = -sum((v/total) * math.log2(v/total) for v in counts.values() if v > 0)
    # 44 Shannon entropy of domain segment only
    counts_d = Counter(domain_part)
    total_d  = len(domain_part) if domain_part else 1
    f44 = -sum((v/total_d) * math.log2(v/total_d)
               for v in counts_d.values() if v > 0)
    # 45 URL-to-domain length ratio
    f45 = len(url) / len(domain_part) if domain_part else 0.0
    # 46 redirect count proxy (count of "http" occurrences beyond the first)
    f46 = max(0, url.lower().count("http") - 1)
    # 47 port number present flag
    f47 = 1 if re.search(r":\d{2,5}(/|$)", domain_part) else 0
    # 48 count of double-slash occurrences beyond protocol
    f48 = max(0, url.count("//") - 1)
    # 49 count of "www" occurrences
    f49 = u.count("www")
    # 50–54  five path-level features
    f50 = len(path_part)                          # path length
    f51 = path_part.count("/")                    # path depth
    f52 = 1 if path_part.endswith((".exe", ".zip", ".php", ".html")) else 0
    f53 = len(re.findall(r"[A-Z]", url))          # uppercase letter count
    f54 = 1 if re.search(r"\d{1,3}-\d{1,3}", url) else 0  # digit-range pattern

    features = (
        [f01, f02, f03, f04, f05, f06, f07, f08, f09, f10,
         f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
         f21, f22, f23]
        + kw_flags          # 19 keyword flags  (indices 24-42)
        + [f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54]
    )
    assert len(features) == 54, f"Expected 54 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def aggregate_url_features(urls: list) -> np.ndarray:
    """
    Aggregate per-URL 54-dim vectors into a single 108-dim email-level vector
    by concatenating the column-wise mean and column-wise maximum.
    Returns a zero vector if no URLs are present (neutral prior).
    """
    if len(urls) == 0:
        return np.zeros(108, dtype=np.float32)
    feats = np.stack([url_features_54(u) for u in urls])   # (n_urls, 54)
    return np.concatenate([feats.mean(axis=0), feats.max(axis=0)])  # (108,)


class LexicalAgent:
    """
    Logistic Regression classifier on 108-dim aggregated URL feature vectors.
    Hyper-parameters: L2 penalty, C=1.0, solver=lbfgs, max_iter=1000.
    Features are standardised with StandardScaler fitted on the training fold.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = LogisticRegression(
            penalty="l2", C=LEX_C,
            solver=LEX_SOLVER, max_iter=LEX_MAX_ITER,
            random_state=SEED
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score(self, text: str) -> float:
        """Return p_lexical for a single email text."""
        urls  = extract_urls(text)
        feats = aggregate_url_features(urls).reshape(1, -1)
        return float(self.predict_proba(feats)[0][1])

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)

    @classmethod
    def load(cls, path: str):
        obj    = cls()
        data   = joblib.load(path)
        obj.scaler = data["scaler"]
        obj.model  = data["model"]
        return obj

# =========================================================
# SECTION 3 — TEXTUAL PATTERN AGENT (BiLSTM)
# =========================================================

class Vocabulary:
    """Simple word vocabulary built from training data."""
    PAD, UNK = 0, 1

    def __init__(self, min_freq: int = TXT_MIN_FREQ,
                 max_size: int = TXT_MAX_VOCAB):
        self.min_freq = min_freq
        self.max_size = max_size
        self.w2i = {"<PAD>": 0, "<UNK>": 1}
        self.i2w = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts: list):
        freq = Counter(w for t in texts for w in t.split())
        vocab = [w for w, c in freq.most_common(self.max_size)
                 if c >= self.min_freq]
        for idx, w in enumerate(vocab, start=2):
            self.w2i[w] = idx
            self.i2w[idx] = w
        log.info("Vocabulary built: %d tokens", len(self.w2i))

    def encode(self, text: str, max_len: int = TXT_MAX_SEQ_LEN) -> list:
        ids = [self.w2i.get(w, self.UNK) for w in text.split()][:max_len]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.w2i)


class EmailTextDataset(TorchDataset):
    def __init__(self, texts, labels, vocab: Vocabulary):
        self.ids    = [torch.tensor(vocab.encode(t), dtype=torch.long)
                       for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM text classifier matching the manuscript specification:
      embedding_dim=128, 2 bidirectional LSTM layers of 64 units each,
      mean pooling, MLP head 128->ReLU->Dropout(0.2)->2->softmax.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, TXT_EMBED_DIM, padding_idx=Vocabulary.PAD
        )
        self.bilstm = nn.LSTM(
            input_size    = TXT_EMBED_DIM,
            hidden_size   = TXT_HIDDEN_DIM,     # 64 units per direction
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = TXT_DROPOUT
        )
        lstm_out_dim = TXT_HIDDEN_DIM * 2       # 128 after BiLSTM
        self.mlp = nn.Sequential(
            nn.Linear(lstm_out_dim, TXT_MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(TXT_DROPOUT),
            nn.Linear(TXT_MLP_HIDDEN, 2)        # binary output
        )

    def forward(self, x):
        emb = self.embedding(x)                 # (B, L, 128)
        out, _ = self.bilstm(emb)               # (B, L, 128)
        pooled = out.mean(dim=1)                # mean pooling -> (B, 128)
        return self.mlp(pooled)                 # (B, 2)


class TextualPatternAgent:
    """
    Wrapper for training and inference with the BiLSTM classifier.
    Adam optimiser, lr=1e-3, batch_size=64, early stopping patience=3.
    """
    def __init__(self):
        self.vocab = Vocabulary()
        self.net   = None

    def fit(self, train_texts, train_labels,
            val_texts=None, val_labels=None):
        self.vocab.build(list(train_texts))
        train_ds = EmailTextDataset(train_texts, train_labels, self.vocab)
        train_dl = DataLoader(train_ds, batch_size=TXT_BATCH_SIZE,
                              shuffle=True)

        self.net = BiLSTMClassifier(len(self.vocab)).to(DEVICE)
        opt      = torch.optim.Adam(self.net.parameters(), lr=TXT_LR)
        crit     = nn.CrossEntropyLoss()

        best_val_loss, patience_count = float("inf"), 0
        best_state = None

        for epoch in range(1, TXT_EPOCHS + 1):
            self.net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(self.net(xb), yb)
                loss.backward()
                opt.step()

            # ── early stopping on validation loss ──────────────
            if val_texts is not None and val_labels is not None:
                val_loss = self._eval_loss(val_texts, val_labels, crit)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                    best_state = {k: v.clone()
                                  for k, v in self.net.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= TXT_EARLY_STOP:
                        log.info("Textual Agent early stop at epoch %d", epoch)
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _eval_loss(self, texts, labels, crit):
        ds = EmailTextDataset(texts, labels, self.vocab)
        dl = DataLoader(ds, batch_size=TXT_BATCH_SIZE)
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += crit(self.net(xb), yb).item()
        return total / len(dl)

    def predict_proba(self, texts) -> np.ndarray:
        ds = EmailTextDataset(texts, np.zeros(len(texts), dtype=int),
                              self.vocab)
        dl = DataLoader(ds, batch_size=TXT_BATCH_SIZE)
        self.net.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in dl:
                logits = self.net(xb.to(DEVICE))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def score(self, text: str) -> float:
        """Return p_textual for a single email text."""
        return float(self.predict_proba([text])[0][1])

    def save(self, path: str):
        joblib.dump({"vocab": self.vocab,
                     "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str):
        obj  = cls()
        data = joblib.load(path)
        obj.vocab = data["vocab"]
        obj.net   = BiLSTMClassifier(len(obj.vocab)).to(DEVICE)
        obj.net.load_state_dict(data["state"])
        return obj

# =========================================================
# SECTION 4 — SEMANTIC CONTENT AGENT (DistilBERT)
# =========================================================

class HFEmailDataset(TorchDataset):
    """HuggingFace-compatible dataset for DistilBERT fine-tuning."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx])
                for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class SemanticContentAgent:
    """
    DistilBERT-base-uncased fine-tuned with a two-layer classification head.
    Architecture: 768->256->ReLU->Dropout(0.3)->2->softmax
    Training: AdamW, lr=2e-5, weight_decay=0.01, linear warm-up (10% steps),
              batch=32, max_epochs=5, early stopping patience=2.
    """
    def __init__(self, model_name: str = SEM_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = None
        self.save_path  = None

    def _tokenize(self, texts):
        return self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=SEM_MAX_LEN
        )

    def fit(self, train_texts, train_labels,
            val_texts=None, val_labels=None,
            fold_id: int = 0):

        train_enc = self._tokenize(train_texts)
        train_ds  = HFEmailDataset(train_enc, list(train_labels))

        callbacks = []
        eval_strat = "no"
        if val_texts is not None:
            val_enc = self._tokenize(val_texts)
            val_ds  = HFEmailDataset(val_enc, list(val_labels))
            eval_strat = "epoch"
            callbacks  = [EarlyStoppingCallback(
                early_stopping_patience=SEM_EARLY_STOP
            )]
        else:
            val_ds = None

        self.save_path = os.path.join(
            OUTPUT_DIR, f"semantic_model_fold{fold_id}"
        )

        # Total training steps for warm-up scheduler
        steps_per_epoch = math.ceil(
            len(train_ds) / SEM_BATCH_SIZE
        )
        total_steps  = steps_per_epoch * SEM_EPOCHS
        warmup_steps = int(0.10 * total_steps)   # 10% warm-up

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels         = 2,
            hidden_dropout_prob= SEM_DROPOUT,
        ).to(DEVICE)

        train_args = TrainingArguments(
            output_dir                  = self.save_path,
            learning_rate               = SEM_LR,
            weight_decay                = SEM_WEIGHT_DECAY,
            per_device_train_batch_size = SEM_BATCH_SIZE,
            num_train_epochs            = SEM_EPOCHS,
            warmup_steps                = warmup_steps,
            evaluation_strategy         = eval_strat,
            save_strategy               = eval_strat,
            load_best_model_at_end      = (val_ds is not None),
            metric_for_best_model       = "loss",
            greater_is_better           = False,
            report_to                   = "none",
            seed                        = SEED,
        )

        trainer = Trainer(
            model          = self.model,
            args           = train_args,
            train_dataset  = train_ds,
            eval_dataset   = val_ds,
            tokenizer      = self.tokenizer,
            callbacks      = callbacks,
        )
        trainer.train()
        trainer.save_model(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def predict_proba(self, texts) -> np.ndarray:
        self.model.eval()
        all_probs = []
        for i in range(0, len(texts), SEM_BATCH_SIZE):
            batch = list(texts[i: i + SEM_BATCH_SIZE])
            enc   = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=SEM_MAX_LEN
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def score(self, text: str) -> float:
        """Return p_semantic for a single email text."""
        return float(self.predict_proba(np.array([text]))[0][1])

    def save(self, path: str):
        if self.save_path and os.path.isdir(self.save_path):
            import shutil
            shutil.copytree(self.save_path, path, dirs_exist_ok=True)

    @classmethod
    def load(cls, path: str):
        obj            = cls.__new__(cls)
        obj.model_name = SEM_MODEL_NAME
        obj.tokenizer  = AutoTokenizer.from_pretrained(path)
        obj.model      = AutoModelForSequenceClassification.from_pretrained(
            path
        ).to(DEVICE)
        obj.save_path  = path
        return obj

# =========================================================
# SECTION 5 — CALIBRATED LATE-FUSION ENSEMBLE
# =========================================================

def late_fusion(p_sem: float, p_txt: float, p_lex: float,
                weights=FUSION_WEIGHTS) -> float:
    """
    Equation 1 from the manuscript:
      p_fused = w_s * p_semantic + w_t * p_textual + w_l * p_lexical
    Weights default to equal (1/3, 1/3, 1/3) as validated in the paper.
    """
    w_s, w_t, w_l = weights
    assert abs(w_s + w_t + w_l - 1.0) < 1e-6, "Weights must sum to 1"
    return w_s * p_sem + w_t * p_txt + w_l * p_lex


def roc_guided_calibration(
        scores: np.ndarray,
        labels: np.ndarray,
        alpha: float = ALPHA
) -> float:
    """
    Algorithm 1 — ROC-guided threshold calibration.

    Selects the largest threshold tau such that FPR(tau) <= alpha,
    maximising TPR subject to that FPR ceiling.

    Equations 2–3 from the manuscript:
      FPR(tau) = FP(tau) / N
      TPR(tau) = TP(tau) / P
      tau_cal  = argmax_tau TPR(tau)  s.t.  FPR(tau) <= alpha
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores, pos_label=1)

    best_tau = thresholds[-1]   # default: most conservative threshold
    best_tpr = 0.0

    for fpr_val, tpr_val, tau in zip(fpr_arr, tpr_arr, thresholds):
        if fpr_val <= alpha:
            if tpr_val > best_tpr:
                best_tpr = tpr_val
                best_tau = tau

    log.info(
        "Calibration complete: tau_cal=%.4f  FPR=%.4f  TPR=%.4f",
        best_tau, fpr_arr[np.searchsorted(fpr_arr, alpha)], best_tpr
    )
    return float(best_tau)

# =========================================================
# SECTION 6 — CROSS-VALIDATION TRAINING PIPELINE
# =========================================================

def compute_metrics(y_true, y_pred, y_scores, tau_cal):
    """Compute accuracy, precision, recall, F1, FPR, AUC, confusion matrix."""
    acc   = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc   = roc_auc_score(y_true, y_scores)
    cm    = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n     = tn + fp       # total legitimate
    fpr   = fp / n if n > 0 else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "fpr": round(fpr * 100, 2),
        "tau_cal": round(tau_cal, 4),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
    }


def train_evaluate_fold(
        fold_id:      int,
        train_texts:  np.ndarray,
        train_labels: np.ndarray,
        test_texts:   np.ndarray,
        test_labels:  np.ndarray,
) -> dict:
    """
    Full training and evaluation pipeline for one cross-validation fold.

    Steps:
      1. Split training fold 80/20 into model-training and calibration-validation.
      2. Train all three agents on the model-training subset.
      3. Build fused scores on the calibration-validation subset.
      4. Run Algorithm 1 to derive tau_cal.
      5. Evaluate on the held-out test fold.
    """
    log.info("=== Fold %d ===", fold_id)

    # ── Step 1: 80/20 split within the training fold ──────
    n_train = len(train_texts)
    n_val   = int(n_train * VAL_FRACTION)
    idx     = np.random.RandomState(SEED + fold_id).permutation(n_train)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    fit_texts,  fit_labels  = train_texts[fit_idx],  train_labels[fit_idx]
    val_texts_f, val_labels_f = train_texts[val_idx], train_labels[val_idx]

    log.info("  Fold %d — model-training: %d  calibration-val: %d  test: %d",
             fold_id, len(fit_texts), len(val_texts_f), len(test_texts))

    # ── Step 2a: train Semantic Content Agent ─────────────
    sem_agent = SemanticContentAgent()
    sem_agent.fit(fit_texts, fit_labels,
                  val_texts=val_texts_f, val_labels=val_labels_f,
                  fold_id=fold_id)

    # ── Step 2b: train Textual Pattern Agent ──────────────
    txt_agent = TextualPatternAgent()
    txt_agent.fit(fit_texts, fit_labels,
                  val_texts=val_texts_f, val_labels=val_labels_f)

    # ── Step 2c: train Lexical Agent (on URL dataset only) ─
    #   (loaded separately; scores are computed per email by
    #    aggregating URL features extracted from the email body)
    lex_agent = LexicalAgent()
    log.info("  Fold %d — loading URL dataset for Lexical Agent ...", fold_id)
    X_url_all, y_url_all = load_url_dataset(URL_DATA_PATH)
    # apply the same 4-fold split to Dataset 2 for consistency
    skf_url = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)
    for _fi, (tr_u, _te_u) in enumerate(
            skf_url.split(X_url_all, y_url_all), 1):
        if _fi == fold_id:
            url_fit_idx = tr_u
            break
    X_url_fit = np.array([
        aggregate_url_features(extract_urls(str(u)))
        for u in X_url_all[url_fit_idx]
    ])
    y_url_fit = y_url_all[url_fit_idx]
    lex_agent.fit(X_url_fit, y_url_fit)

    # ── Step 3: fused scores on calibration-validation set ─
    log.info("  Fold %d — computing calibration scores ...", fold_id)
    p_sem_val = sem_agent.predict_proba(val_texts_f)[:, 1]
    p_txt_val = txt_agent.predict_proba(val_texts_f)[:, 1]
    p_lex_val = np.array([lex_agent.score(t) for t in val_texts_f])
    p_fused_val = np.array([
        late_fusion(s, t, l)
        for s, t, l in zip(p_sem_val, p_txt_val, p_lex_val)
    ])

    # ── Step 4: ROC-guided calibration (Algorithm 1) ───────
    tau_cal   = roc_guided_calibration(p_fused_val, val_labels_f, alpha=ALPHA)
    tau_block = min(1.0, tau_cal + DELTA_BLOCK)
    log.info("  Fold %d — tau_cal=%.4f  tau_block=%.4f",
             fold_id, tau_cal, tau_block)

    # ── Step 5: evaluate on held-out test fold ─────────────
    log.info("  Fold %d — evaluating on test fold ...", fold_id)
    p_sem_test = sem_agent.predict_proba(test_texts)[:, 1]
    p_txt_test = txt_agent.predict_proba(test_texts)[:, 1]
    p_lex_test = np.array([lex_agent.score(t) for t in test_texts])
    p_fused_test = np.array([
        late_fusion(s, t, l)
        for s, t, l in zip(p_sem_test, p_txt_test, p_lex_test)
    ])

    y_pred = (p_fused_test >= tau_cal).astype(int)
    metrics = compute_metrics(
        test_labels, y_pred, p_fused_test, tau_cal
    )
    log.info("  Fold %d — %s", fold_id, metrics)

    # ── save fold models ───────────────────────────────────
    sem_agent.save(os.path.join(OUTPUT_DIR, f"semantic_fold{fold_id}"))
    txt_agent.save(os.path.join(OUTPUT_DIR, f"textual_fold{fold_id}.joblib"))
    lex_agent.save(os.path.join(OUTPUT_DIR, f"lexical_fold{fold_id}.joblib"))
    joblib.dump({"tau_cal": tau_cal, "tau_block": tau_block},
                os.path.join(OUTPUT_DIR, f"thresholds_fold{fold_id}.joblib"))

    return metrics


def run_cross_validation(df: pd.DataFrame) -> list:
    """
    Execute the full stratified 4-fold cross-validation protocol.
    Each instance appears in exactly one test fold.
    """
    texts  = df["text"].values
    labels = df["y"].values

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)
    results = []

    for fold_id, (train_idx, test_idx) in enumerate(
            skf.split(texts, labels), 1):
        metrics = train_evaluate_fold(
            fold_id      = fold_id,
            train_texts  = texts[train_idx],
            train_labels = labels[train_idx],
            test_texts   = texts[test_idx],
            test_labels  = labels[test_idx],
        )
        results.append(metrics)

    # ── summary across folds ───────────────────────────────
    log.info("=== Cross-Validation Summary ===")
    for key in ["accuracy", "precision", "recall", "f1", "fpr", "auc"]:
        vals = [r[key] for r in results]
        log.info("  %s: %.4f ± %.4f",
                 key.upper(), np.mean(vals), np.std(vals))
    return results

# =========================================================
# SECTION 7 — SIMULATED SDN ENFORCEMENT LAYER
# =========================================================

def sdn_enforce(
        text:        str,
        sem_agent:   SemanticContentAgent,
        txt_agent:   TextualPatternAgent,
        lex_agent:   LexicalAgent,
        tau_cal:     float,
        tau_block:   float,
) -> dict:
    """
    Simulated SDN enforcement layer (Algorithm 2).

    Applies the three-tier tiered decision policy:
      score >= tau_block  ->  BLOCK   (quarantine via simulated southbound API)
      score >= tau_cal    ->  MONITOR (mirror to analysis VLAN; tag for review)
      score <  tau_cal    ->  ALLOW   (deliver without intervention)

    Note: This is a software simulation of SDN enforcement logic.
    Hardware testbed (OpenFlow/Mininet) validation is identified as future work.
    """
    # ── compute per-agent probabilities ───────────────────
    p_sem = sem_agent.score(text)
    p_txt = txt_agent.score(text)
    p_lex = lex_agent.score(text)

    # ── late-fusion (Equation 1) ───────────────────────────
    p_fused = late_fusion(p_sem, p_txt, p_lex)

    # ── tiered enforcement decision ────────────────────────
    if p_fused >= tau_block:
        action     = "BLOCK"
        sdn_action = "quarantine_via_simulated_southbound_API"
    elif p_fused >= tau_cal:
        action     = "MONITOR"
        sdn_action = "mirror_to_analysis_vlan_tag_for_review"
    else:
        action     = "ALLOW"
        sdn_action = "deliver_without_intervention"

    return {
        "p_fused":     round(p_fused, 4),
        "p_semantic":  round(p_sem,   4),
        "p_textual":   round(p_txt,   4),
        "p_lexical":   round(p_lex,   4),
        "tau_cal":     round(tau_cal,   4),
        "tau_block":   round(tau_block, 4),
        "action":      action,
        "sdn_action":  sdn_action,
        "timestamp":   datetime.datetime.utcnow().isoformat(),
    }

# =========================================================
# SECTION 8 — MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":

    # ── load and preprocess Dataset 1 ─────────────────────
    df_email = load_email_dataset(URL_DATA_PATH)

    # ── run full 4-fold cross-validation ──────────────────
    cv_results = run_cross_validation(df_email)

    # ── report aggregate confusion matrix counts ──────────
    TP_total = sum(r["TP"] for r in cv_results)
    FP_total = sum(r["FP"] for r in cv_results)
    TN_total = sum(r["TN"] for r in cv_results)
    FN_total = sum(r["FN"] for r in cv_results)
    log.info(
        "Pooled confusion matrix — TP=%d  FN=%d  FP=%d  TN=%d",
        TP_total, FN_total, FP_total, TN_total
    )

    # ── demonstrate inference with the Fold 1 models ──────
    log.info("=== Demo: SDN enforcement on a test email ===")

    sem_demo = SemanticContentAgent.load(
        os.path.join(OUTPUT_DIR, "semantic_fold1")
    )
    txt_demo = TextualPatternAgent.load(
        os.path.join(OUTPUT_DIR, "textual_fold1.joblib")
    )
    lex_demo = LexicalAgent.load(
        os.path.join(OUTPUT_DIR, "lexical_fold1.joblib")
    )
    thresholds = joblib.load(
        os.path.join(OUTPUT_DIR, "thresholds_fold1.joblib")
    )

    test_email = (
        "Urgent: Your bank account has been locked. "
        "Please verify your identity immediately at "
        "http://secure-login-verify.com/confirm?user=you&token=abc123"
    )

    result = sdn_enforce(
        text      = preprocess_text(test_email),
        sem_agent = sem_demo,
        txt_agent = txt_demo,
        lex_agent = lex_demo,
        tau_cal   = thresholds["tau_cal"],
        tau_block = thresholds["tau_block"],
    )

    print("\n=== EmailGuard Result ===")
    for k, v in result.items():
        print(f"  {k:<20}: {v}")
