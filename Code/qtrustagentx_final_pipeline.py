"""
QTrustAgent-X Final Pipeline
============================
Explainable Agentic Quantum-Graph Inspired Multi-Channel Phishing Detection

Expected dataset root:
D:\\other\\QTrustPhish\\Dataset_Reorganized

Main outputs:
D:\\other\\QTrustPhish\\QTrustAgentX_Results

Run examples:
    conda activate qtrustphish
    cd D:\\other\\QTrustPhish\\code
    python qtrustagentx_final_pipeline.py --mode all --qr_limit 20000 --qr_epochs 10

Recommended installs:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib pillow tqdm openpyxl
    pip install torch torchvision torchaudio
    pip install shap lime networkx

Notes:
- This file is intentionally practical. It uses fast, reproducible models first.
- The "quantum" module is a quantum-inspired random feature map plus trust-graph interaction encoder.
- It contains grouped ablation perspectives A-F suitable for paper tables.
"""

import argparse
import json
import math
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

warnings.filterwarnings("ignore")


def identity_passthrough(X):
    """Pickle-safe identity function for sklearn FunctionTransformer."""
    return X


def safe_joblib_dump(model, path: Path):
    """Save sklearn models without crashing the full run if a custom step is not serializable."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return True, ""
    except Exception as e:
        warn_path = path.with_suffix(path.suffix + ".save_error.txt")
        with open(warn_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"WARNING: Could not save model {path.name}: {e}")
        return False, str(e)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Paths and configuration
# =============================================================================

@dataclass
class Config:
    data_root: Path
    out_root: Path
    qr_limit: int = 20000
    qr_epochs: int = 10
    qr_batch_size: int = 64
    test_size: float = 0.20
    val_size: float = 0.10
    seed: int = 42
    max_text_features: int = 25000
    svd_components: int = 256
    quantum_dim: int = 256


DEFAULT_DATA_ROOT = Path(r"D:\other\QTrustPhish\Dataset_Reorganized")
DEFAULT_OUT_ROOT = Path(r"D:\other\QTrustPhish\QTrustAgentX_Results")


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dirs(cfg: Config):
    for sub in [
        "models", "results", "figures", "reports", "timing", "predictions",
        "explanations", "splits", "logs", "tables"
    ]:
        (cfg.out_root / sub).mkdir(parents=True, exist_ok=True)


def timer():
    return time.perf_counter()


def elapsed(start):
    return round(time.perf_counter() - start, 4)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            pass
    raise RuntimeError(f"Could not read CSV: {path}")


def safe_read_txt(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]
    for enc in encodings:
        for sep in ["\t", None]:
            try:
                if sep is None:
                    return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip", header=None)
                return pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip", header=None)
            except Exception:
                pass
    raise RuntimeError(f"Could not read TXT: {path}")


def binary_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            out["auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            out["auc"] = np.nan
    else:
        out["auc"] = np.nan
    return out


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            s = model.decision_function(X)
            s = np.asarray(s, dtype=float)
            return (s - s.min()) / (s.max() - s.min() + 1e-12)
        except Exception:
            pass
    return None


def plot_bar_metrics(df: pd.DataFrame, out_path: Path, title: str):
    try:
            metrics = ["accuracy", "precision", "recall", "f1", "auc"]
            use = df.copy()
            use = use[["experiment"] + [m for m in metrics if m in use.columns]]
            x = np.arange(len(use))
            width = 0.15
            plt.figure(figsize=(max(12, len(use) * 0.8), 6))
            for i, m in enumerate(metrics):
                if m in use.columns:
                    plt.bar(x + (i - 2) * width, use[m].astype(float), width, label=m)
            plt.xticks(x, use["experiment"], rotation=45, ha="right")
            plt.ylim(0, 1.05)
            plt.ylabel("Score")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
    except Exception as e:
        print(f"WARNING: plot failed in plot_bar_metrics: {e}")
    finally:
        try:
            plt.close('all')
        except Exception:
            pass


def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, interpolation="nearest")
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["benign", "phishing"], rotation=45)
            plt.yticks(tick_marks, ["benign", "phishing"])
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
    except Exception as e:
        print(f"WARNING: plot failed in plot_confusion: {e}")
    finally:
        try:
            plt.close('all')
        except Exception:
            pass


def plot_roc(y_true, y_score, out_path: Path, title: str):
    try:
            if y_score is None:
                return
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()

    except Exception as e:
        print(f"WARNING: plot failed in plot_roc: {e}")
    finally:
        try:
            plt.close('all')
        except Exception:
            pass

# =============================================================================
# Dataset loaders
# =============================================================================

def load_url_dataset(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    path = cfg.data_root / "01_url" / "url_phishing_11430_89features.csv"
    df = safe_read_csv(path)
    label_col = "status" if "status" in df.columns else df.columns[-1]
    y_raw = df[label_col].astype(str).str.lower()
    y = y_raw.map({"legitimate": 0, "benign": 0, "0": 0, "false": 0, "phishing": 1, "malicious": 1, "1": 1, "true": 1})
    if y.isna().any():
        # fallback factorize, highest risk term as 1 where possible
        y = pd.Series(pd.factorize(y_raw)[0], index=df.index)
    X = df.drop(columns=[label_col])
    # Drop obvious non-numeric URL string if present, preserve numeric features
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    return X, y.astype(int)


def load_sms_dataset(cfg: Config, use_duplicate_spam: bool = False) -> Tuple[pd.Series, pd.Series]:
    frames = []
    # main SMS phishing
    p1 = cfg.data_root / "02_sms" / "sms_phishing_5971.csv"
    if p1.exists():
        df = safe_read_csv(p1)
        text_col = "TEXT" if "TEXT" in df.columns else next((c for c in df.columns if str(c).lower() in ["text", "message", "sms"]), df.columns[-1])
        label_col = "LABEL" if "LABEL" in df.columns else next((c for c in df.columns if str(c).lower() in ["label", "class", "target"]), df.columns[0])
        tmp = pd.DataFrame({"text": df[text_col].astype(str), "label_raw": df[label_col].astype(str)})
        frames.append(tmp)
    # smishing txt
    p2 = cfg.data_root / "02_sms" / "sms_smishing_5571.txt"
    if p2.exists():
        df = safe_read_txt(p2)
        if len(df.columns) >= 2:
            tmp = pd.DataFrame({"text": df.iloc[:, 1].astype(str), "label_raw": df.iloc[:, 0].astype(str)})
            frames.append(tmp)
    # duplicate spam optional
    p3 = cfg.data_root / "02_sms" / "sms_spam_raw_duplicate_check.csv"
    if use_duplicate_spam and p3.exists():
        df = safe_read_csv(p3)
        label_col = df.columns[0]
        text_col = df.columns[1]
        tmp = pd.DataFrame({"text": df[text_col].astype(str), "label_raw": df[label_col].astype(str)})
        frames.append(tmp)
    data = pd.concat(frames, ignore_index=True).dropna()
    data["text_norm"] = data["text"].astype(str).str.strip().str.lower()
    data = data.drop_duplicates("text_norm")
    y = data["label_raw"].astype(str).str.lower().map({
        "ham": 0, "legit": 0, "legitimate": 0, "benign": 0, "0": 0,
        "spam": 1, "smish": 1, "phishing": 1, "malicious": 1, "1": 1
    })
    y = y.fillna(data["label_raw"].astype(str).str.contains("smish|spam|phish", case=False, regex=True).astype(int))
    return data["text"].reset_index(drop=True), y.astype(int).reset_index(drop=True)


def load_email_dataset(cfg: Config) -> Tuple[pd.Series, pd.Series, pd.Series]:
    root = cfg.data_root / "03_email_human_llm"
    files = [
        (root / "human_legit" / "human_legit_email_1000.csv", 0, "human"),
        (root / "human_phishing" / "human_phishing_email_1000.csv", 1, "human"),
        (root / "llm_legit" / "llm_legit_email_1000.csv", 0, "llm"),
        (root / "llm_phishing" / "llm_phishing_email_595.csv", 1, "llm"),
    ]
    texts, labels, authors = [], [], []
    for path, label, author in files:
        if not path.exists():
            continue
        df = safe_read_csv(path)
        text_col = "body" if "body" in df.columns else next((c for c in df.columns if str(c).lower() in ["text", "message", "email", "content", "body"]), df.columns[-1])
        for v in df[text_col].fillna("").astype(str).tolist():
            texts.append(v)
            labels.append(label)
            authors.append(author)
    data = pd.DataFrame({"text": texts, "label": labels, "author": authors})
    data["text_norm"] = data["text"].str.strip().str.lower()
    data = data.drop_duplicates("text_norm")
    return data["text"].reset_index(drop=True), data["label"].astype(int).reset_index(drop=True), data["author"].reset_index(drop=True)


def list_qr_images(cfg: Config, limit: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for label_name, label in [("benign", 0), ("malicious", 1)]:
        folder = cfg.data_root / "04_qr" / label_name
        imgs = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
            imgs.extend(folder.glob(ext))
        imgs = sorted(imgs)
        if limit:
            per_class = max(1, limit // 2)
            imgs = imgs[:per_class]
        for p in imgs:
            rows.append({"path": str(p), "label": label, "label_name": label_name})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)


# =============================================================================
# Quantum-inspired and agentic modules
# =============================================================================

class QuantumFeatureMap(BaseEstimator, TransformerMixin):
    """Quantum-inspired cosine/sine random feature map.

    This approximates nonlinear Hilbert-space mapping and is used as a lightweight
    variational quantum surrogate for tabular and compressed text embeddings.
    """
    def __init__(self, n_components=256, gamma=1.0, seed=42):
        self.n_components = n_components
        self.gamma = gamma
        self.seed = seed

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        self.W_ = rng.normal(0, self.gamma, size=(X.shape[1], self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=(self.n_components,))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        proj = X @ self.W_ + self.b_
        return np.concatenate([np.cos(proj), np.sin(proj)], axis=1) / math.sqrt(self.n_components)


class TrustGraphInteractionMap(BaseEstimator, TransformerMixin):
    """Creates cross-feature trust interactions as graph-inspired relational edges."""
    def __init__(self, max_pairs=128, seed=42):
        self.max_pairs = max_pairs
        self.seed = seed

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        n = X.shape[1]
        pairs = []
        for _ in range(min(self.max_pairs, n * (n - 1) // 2)):
            i, j = rng.choice(n, size=2, replace=False)
            pairs.append((int(i), int(j)))
        self.pairs_ = pairs
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        feats = []
        for i, j in self.pairs_:
            feats.append((X[:, i] * X[:, j]).reshape(-1, 1))
            feats.append(np.abs(X[:, i] - X[:, j]).reshape(-1, 1))
        if not feats:
            return np.empty((X.shape[0], 0))
        return np.hstack(feats)


def build_text_agent(model_type="svm", max_features=25000, svd_components=None, quantum=False, quantum_dim=256):
    base_steps = [
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True))
    ]
    if svd_components:
        base_steps += [("svd", TruncatedSVD(n_components=svd_components, random_state=SEED)), ("scale", StandardScaler())]
        if quantum:
            base_steps += [("qmap", QuantumFeatureMap(n_components=quantum_dim, gamma=0.5, seed=SEED))]
    if model_type == "svm":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), cv=3)
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", random_state=SEED)
    else:
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", random_state=SEED)
    return Pipeline(base_steps + [("clf", clf)])


def build_url_agent(kind="rf", quantum=False, graph=False, quantum_dim=256):
    preprocess_steps = [("scale", StandardScaler())]
    if graph:
        # combine original scaled features with graph interaction edges
        union = FeatureUnion([
            ("identity", FunctionTransformer(identity_passthrough, validate=False)),
            ("graph", TrustGraphInteractionMap(max_pairs=128, seed=SEED)),
        ])
        preprocess_steps.append(("graph_union", union))
        preprocess_steps.append(("scale2", StandardScaler()))
    if quantum:
        preprocess_steps.append(("qmap", QuantumFeatureMap(n_components=quantum_dim, gamma=0.3, seed=SEED)))
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=SEED, class_weight="balanced")
    elif kind == "extra":
        clf = ExtraTreesClassifier(n_estimators=700, max_depth=None, n_jobs=-1, random_state=SEED, class_weight="balanced")
    elif kind == "gb":
        clf = GradientBoostingClassifier(random_state=SEED)
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced", random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED, class_weight="balanced")
    return Pipeline(preprocess_steps + [("clf", clf)])


# =============================================================================
# Training agents
# =============================================================================

def train_and_eval(name: str, model, X_train, X_test, y_train, y_test, cfg: Config) -> Dict[str, Any]:
    start = timer()
    model.fit(X_train, y_train)
    train_time = elapsed(start)
    start_pred = timer()
    y_pred = model.predict(X_test)
    y_score = get_score(model, X_test)
    pred_time = elapsed(start_pred)
    metrics = binary_metrics(y_test, y_pred, y_score)
    metrics.update({"experiment": name, "train_time_sec": train_time, "predict_time_sec": pred_time, "n_train": len(y_train), "n_test": len(y_test)})
    safe_joblib_dump(model, cfg.out_root / "models" / f"{name}.joblib")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_score": y_score if y_score is not None else np.nan}).to_csv(
        cfg.out_root / "predictions" / f"{name}_predictions.csv", index=False
    )
    plot_confusion(y_test, y_pred, cfg.out_root / "figures" / f"{name}_confusion.png", f"{name} Confusion Matrix")
    if y_score is not None:
        plot_roc(y_test, y_score, cfg.out_root / "figures" / f"{name}_roc.png", f"{name} ROC Curve")
    with open(cfg.out_root / "reports" / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, target_names=["benign", "phishing"], zero_division=0))
    return metrics


def run_url_experiments(cfg: Config) -> pd.DataFrame:
    X, y = load_url_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    experiments = {
        "URL_RF_Baseline": build_url_agent("rf", quantum=False, graph=False, quantum_dim=cfg.quantum_dim),
        "URL_ExtraTrees_Strong": build_url_agent("extra", quantum=False, graph=False, quantum_dim=cfg.quantum_dim),
        "URL_GraphTrust_Agent": build_url_agent("extra", quantum=False, graph=True, quantum_dim=cfg.quantum_dim),
        "URL_QuantumGraph_Agent": build_url_agent("extra", quantum=True, graph=True, quantum_dim=cfg.quantum_dim),
        "URL_NoQuantum_GraphOnly": build_url_agent("extra", quantum=False, graph=True, quantum_dim=cfg.quantum_dim),
        "URL_NoGraph_QuantumOnly": build_url_agent("extra", quantum=True, graph=False, quantum_dim=cfg.quantum_dim),
    }
    rows = []
    for name, model in experiments.items():
        print(f"Training {name}")
        rows.append(train_and_eval(name, model, X_train, X_test, y_train.values, y_test.values, cfg))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_root / "results" / "url_agent_results.csv", index=False)
    plot_bar_metrics(df, cfg.out_root / "figures" / "url_agent_metrics.png", "URL Agent Results")
    return df


def run_sms_experiments(cfg: Config) -> pd.DataFrame:
    X, y = load_sms_dataset(cfg, use_duplicate_spam=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    experiments = {
        "SMS_TFIDF_SVM_Baseline": build_text_agent("svm", cfg.max_text_features, None, False, cfg.quantum_dim),
        "SMS_SemanticCompressed_Agent": build_text_agent("svm", cfg.max_text_features, cfg.svd_components, False, cfg.quantum_dim),
        "SMS_QuantumText_Agent": build_text_agent("logreg", cfg.max_text_features, cfg.svd_components, True, cfg.quantum_dim),
        "SMS_NoQuantum_TextOnly": build_text_agent("logreg", cfg.max_text_features, cfg.svd_components, False, cfg.quantum_dim),
    }
    rows = []
    for name, model in experiments.items():
        print(f"Training {name}")
        rows.append(train_and_eval(name, model, X_train, X_test, y_train.values, y_test.values, cfg))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_root / "results" / "sms_agent_results.csv", index=False)
    plot_bar_metrics(df, cfg.out_root / "figures" / "sms_agent_metrics.png", "SMS Agent Results")
    return df


def run_email_experiments(cfg: Config) -> pd.DataFrame:
    X, y, author = load_email_dataset(cfg)
    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(X, y, author, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    experiments = {
        "EMAIL_TFIDF_SVM_Baseline": build_text_agent("svm", cfg.max_text_features, None, False, cfg.quantum_dim),
        "EMAIL_SemanticCompressed_Agent": build_text_agent("svm", cfg.max_text_features, cfg.svd_components, False, cfg.quantum_dim),
        "EMAIL_QuantumText_Agent": build_text_agent("logreg", cfg.max_text_features, cfg.svd_components, True, cfg.quantum_dim),
        "EMAIL_NoQuantum_TextOnly": build_text_agent("logreg", cfg.max_text_features, cfg.svd_components, False, cfg.quantum_dim),
    }
    rows = []
    for name, model in experiments.items():
        print(f"Training {name}")
        row = train_and_eval(name, model, X_train, X_test, y_train.values, y_test.values, cfg)
        # Human vs LLM subgroup evaluation
        pred = model.predict(X_test)
        score = get_score(model, X_test)
        for subgroup in ["human", "llm"]:
            mask = (a_test.values == subgroup)
            if mask.sum() > 0:
                subm = binary_metrics(y_test.values[mask], pred[mask], score[mask] if score is not None else None)
                for k, v in subm.items():
                    row[f"{subgroup}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_root / "results" / "email_agent_results.csv", index=False)
    plot_bar_metrics(df, cfg.out_root / "figures" / "email_agent_metrics.png", "Email Agent Results")
    return df


# =============================================================================
# QR model with torch, plus fallback handcrafted QR features
# =============================================================================

def qr_handcrafted_features(paths: List[str]) -> np.ndarray:
    feats = []
    for p in tqdm(paths, desc="QR handcrafted features"):
        try:
            img = Image.open(p).convert("L").resize((128, 128))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            # density and grid features
            density = arr.mean()
            std = arr.std()
            edge_h = np.abs(np.diff(arr, axis=1)).mean()
            edge_v = np.abs(np.diff(arr, axis=0)).mean()
            blocks = []
            for i in range(0, 128, 16):
                for j in range(0, 128, 16):
                    blocks.append(arr[i:i+16, j:j+16].mean())
            feats.append([density, std, edge_h, edge_v] + blocks)
        except Exception:
            feats.append([0.0] * (4 + 64))
    return np.asarray(feats, dtype=np.float32)


def run_qr_handcrafted(cfg: Config) -> pd.DataFrame:
    df = list_qr_images(cfg, limit=cfg.qr_limit)
    X_train, X_test, y_train, y_test = train_test_split(df["path"], df["label"], test_size=cfg.test_size, stratify=df["label"], random_state=cfg.seed)
    Xtr = qr_handcrafted_features(X_train.tolist())
    Xte = qr_handcrafted_features(X_test.tolist())
    experiments = {
        "QR_Handcrafted_RF": Pipeline([("scale", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED, class_weight="balanced"))]),
        "QR_Handcrafted_Quantum": Pipeline([("scale", StandardScaler()), ("qmap", QuantumFeatureMap(n_components=cfg.quantum_dim, gamma=0.5, seed=SEED)), ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", random_state=SEED))]),
    }
    rows = []
    for name, model in experiments.items():
        print(f"Training {name}")
        rows.append(train_and_eval(name, model, Xtr, Xte, y_train.values, y_test.values, cfg))
    out = pd.DataFrame(rows)
    out.to_csv(cfg.out_root / "results" / "qr_handcrafted_results.csv", index=False)
    plot_bar_metrics(out, cfg.out_root / "figures" / "qr_handcrafted_metrics.png", "QR Handcrafted Agent Results")
    return out


def run_qr_cnn(cfg: Config) -> pd.DataFrame:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, models
    except Exception as e:
        print("PyTorch/torchvision unavailable. Running handcrafted QR only.")
        return run_qr_handcrafted(cfg)

    class QRDataset(Dataset):
        def __init__(self, frame, transform=None):
            self.frame = frame.reset_index(drop=True)
            self.transform = transform
        def __len__(self):
            return len(self.frame)
        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            img = Image.open(row["path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, int(row["label"])

    df = list_qr_images(cfg, limit=cfg.qr_limit)
    train_df, test_df = train_test_split(df, test_size=cfg.test_size, stratify=df["label"], random_state=cfg.seed)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_loader = DataLoader(QRDataset(train_df, train_tf), batch_size=cfg.qr_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(QRDataset(test_df, test_tf), batch_size=cfg.qr_batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer model, robust and fast enough.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.qr_epochs))

    start = timer()
    epoch_rows = []
    for epoch in range(cfg.qr_epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"QR ResNet epoch {epoch+1}/{cfg.qr_epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        epoch_loss = float(np.mean(losses)) if losses else 0
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")
        epoch_rows.append({"epoch": epoch+1, "loss": epoch_loss})
    train_time = elapsed(start)

    model.eval()
    y_true, y_pred, y_score = [], [], []
    start_pred = timer()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())
            y_score.extend(prob.tolist())
    pred_time = elapsed(start_pred)
    metrics = binary_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
    metrics.update({"experiment": "QR_ResNet18_Agent", "train_time_sec": train_time, "predict_time_sec": pred_time, "n_train": len(train_df), "n_test": len(test_df)})

    torch.save(model.state_dict(), cfg.out_root / "models" / "QR_ResNet18_Agent.pt")
    pd.DataFrame(epoch_rows).to_csv(cfg.out_root / "timing" / "qr_resnet_training_log.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_score": y_score}).to_csv(cfg.out_root / "predictions" / "QR_ResNet18_Agent_predictions.csv", index=False)
    plot_confusion(y_true, y_pred, cfg.out_root / "figures" / "QR_ResNet18_Agent_confusion.png", "QR ResNet18 Confusion Matrix")
    plot_roc(y_true, y_score, cfg.out_root / "figures" / "QR_ResNet18_Agent_roc.png", "QR ResNet18 ROC")

    cnn_df = pd.DataFrame([metrics])
    hand_df = run_qr_handcrafted(cfg)
    out = pd.concat([cnn_df, hand_df], ignore_index=True)
    out.to_csv(cfg.out_root / "results" / "qr_agent_results.csv", index=False)
    plot_bar_metrics(out, cfg.out_root / "figures" / "qr_agent_metrics.png", "QR Agent Results")
    return out


# =============================================================================
# Agentic arbitration and ablations
# =============================================================================

def extract_agent_scores(cfg: Config) -> pd.DataFrame:
    """Build a common meta-dataset from saved best agent predictions.

    Since modality datasets are disjoint, this creates an evidence-level risk table rather than forced sample-level fusion.
    Each row represents one evidence item from one channel with its agent risk score and metadata.
    """
    rows = []
    pred_files = list((cfg.out_root / "predictions").glob("*_predictions.csv"))
    for p in pred_files:
        df = pd.read_csv(p)
        name = p.name.replace("_predictions.csv", "")
        # Use only best-like agents to avoid too many duplicates in arbitration.
        keep_keywords = ["URL_QuantumGraph_Agent", "SMS_QuantumText_Agent", "EMAIL_QuantumText_Agent", "QR_ResNet18_Agent", "QR_Handcrafted_Quantum"]
        if not any(k == name for k in keep_keywords):
            continue
        modality = "url" if name.startswith("URL") else "sms" if name.startswith("SMS") else "email" if name.startswith("EMAIL") else "qr"
        for _, r in df.iterrows():
            score = r.get("y_score", np.nan)
            if pd.isna(score):
                score = float(r.get("y_pred", 0))
            rows.append({
                "agent": name,
                "modality": modality,
                "risk_score": float(score),
                "agent_pred": int(r.get("y_pred", 0)),
                "label": int(r.get("y_true", 0)),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_agentic_arbitration(cfg: Config) -> pd.DataFrame:
    meta = extract_agent_scores(cfg)
    if meta.empty:
        print("No agent predictions found for arbitration. Run agents first.")
        return pd.DataFrame()
    # Agentic risk arbitration features
    # A single evidence item is scored by its specialist agent. We enrich with modality priors and confidence distance.
    modality_dummies = pd.get_dummies(meta["modality"], prefix="mod")
    X = pd.concat([
        meta[["risk_score", "agent_pred"]].reset_index(drop=True),
        pd.DataFrame({
            "risk_margin": np.abs(meta["risk_score"].values - 0.5),
            "high_confidence": (np.abs(meta["risk_score"].values - 0.5) > 0.35).astype(int),
        }),
        modality_dummies.reset_index(drop=True)
    ], axis=1)
    y = meta["label"].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)

    experiments = {
        "Agentic_MajorityVote_Baseline": LogisticRegression(max_iter=1000, random_state=SEED),
        "Agentic_RiskArbitration_X": GradientBoostingClassifier(random_state=SEED),
        "Agentic_QuantumArbitration_X": Pipeline([
            ("scale", StandardScaler()),
            ("qmap", QuantumFeatureMap(n_components=cfg.quantum_dim, gamma=0.7, seed=SEED)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
        ]),
    }
    rows = []
    for name, model in experiments.items():
        print(f"Training {name}")
        rows.append(train_and_eval(name, model, X_train, X_test, y_train, y_test, cfg))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_root / "results" / "agentic_arbitration_results.csv", index=False)
    plot_bar_metrics(df, cfg.out_root / "figures" / "agentic_arbitration_metrics.png", "Agentic Risk Arbitration")
    # Save evidence table for explanations
    meta.to_csv(cfg.out_root / "explanations" / "agent_evidence_scores.csv", index=False)
    return df


def run_ablation_suite(cfg: Config) -> pd.DataFrame:
    """Create grouped ablation tables A-F.

    Perspective A: Agentic orchestration and arbitration.
    Perspective B: Quantum and graph reasoning.
    Perspective C: Modality contribution.
    Perspective D: Human-vs-LLM generalization.
    Perspective E: Robustness/noisy evidence.
    Perspective F: Explainability faithfulness proxies.
    """
    all_result_files = list((cfg.out_root / "results").glob("*_results.csv"))
    parts = []
    for p in all_result_files:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            pass
    if not parts:
        print("No result files found. Run training first.")
        return pd.DataFrame()
    results = pd.concat(parts, ignore_index=True)

    rows = []

    def add(perspective, ablation, variant_name, source_exp, claim):
        match = results[results["experiment"] == source_exp]
        if len(match) == 0:
            return
        r = match.iloc[0].to_dict()
        rows.append({
            "perspective": perspective,
            "ablation": ablation,
            "variant": variant_name,
            "source_experiment": source_exp,
            "accuracy": r.get("accuracy", np.nan),
            "precision": r.get("precision", np.nan),
            "recall": r.get("recall", np.nan),
            "f1": r.get("f1", np.nan),
            "auc": r.get("auc", np.nan),
            "train_time_sec": r.get("train_time_sec", np.nan),
            "what_it_tests": claim,
        })

    # A. Agentic orchestration perspective
    add("A. Agentic Orchestration", "A1", "Single URL specialist", "URL_QuantumGraph_Agent", "Single specialist evidence agent")
    add("A. Agentic Orchestration", "A2", "Single SMS specialist", "SMS_QuantumText_Agent", "Single specialist evidence agent")
    add("A. Agentic Orchestration", "A3", "Single Email specialist", "EMAIL_QuantumText_Agent", "Single specialist evidence agent")
    add("A. Agentic Orchestration", "A4", "Single QR specialist", "QR_ResNet18_Agent", "Single specialist evidence agent")
    add("A. Agentic Orchestration", "A5", "Agentic majority baseline", "Agentic_MajorityVote_Baseline", "Simple arbitration baseline")
    add("A. Agentic Orchestration", "A6", "Agentic risk arbitration", "Agentic_RiskArbitration_X", "Agentic decision arbitration")
    add("A. Agentic Orchestration", "A7", "Agentic quantum arbitration", "Agentic_QuantumArbitration_X", "Quantum arbitration over agent evidence")

    # B. Quantum graph reasoning perspective
    add("B. Quantum-Graph Reasoning", "B1", "URL baseline RF", "URL_RF_Baseline", "No graph, no quantum")
    add("B. Quantum-Graph Reasoning", "B2", "URL graph only", "URL_NoQuantum_GraphOnly", "Trust-graph interactions without quantum mapping")
    add("B. Quantum-Graph Reasoning", "B3", "URL quantum only", "URL_NoGraph_QuantumOnly", "Quantum mapping without graph interactions")
    add("B. Quantum-Graph Reasoning", "B4", "URL quantum graph", "URL_QuantumGraph_Agent", "Combined quantum graph encoder")
    add("B. Quantum-Graph Reasoning", "B5", "SMS compressed no quantum", "SMS_NoQuantum_TextOnly", "Semantic compression only")
    add("B. Quantum-Graph Reasoning", "B6", "SMS quantum text", "SMS_QuantumText_Agent", "Quantum text map")
    add("B. Quantum-Graph Reasoning", "B7", "Email compressed no quantum", "EMAIL_NoQuantum_TextOnly", "Semantic compression only")
    add("B. Quantum-Graph Reasoning", "B8", "Email quantum text", "EMAIL_QuantumText_Agent", "Quantum text map")

    # C. Modality contribution perspective
    add("C. Modality Contribution", "C1", "URL channel", "URL_QuantumGraph_Agent", "URL/domain evidence")
    add("C. Modality Contribution", "C2", "SMS channel", "SMS_QuantumText_Agent", "Smishing evidence")
    add("C. Modality Contribution", "C3", "Email channel", "EMAIL_QuantumText_Agent", "Human and LLM email evidence")
    add("C. Modality Contribution", "C4", "QR channel", "QR_ResNet18_Agent", "QR image evidence")
    add("C. Modality Contribution", "C5", "QR handcrafted", "QR_Handcrafted_Quantum", "QR structural evidence without CNN")

    # D. Human vs LLM generalization perspective
    e = results[results["experiment"] == "EMAIL_QuantumText_Agent"]
    if len(e):
        r = e.iloc[0]
        for subgroup in ["human", "llm"]:
            rows.append({
                "perspective": "D. Human-vs-LLM Generalization",
                "ablation": "D1" if subgroup == "human" else "D2",
                "variant": f"Email quantum text on {subgroup}",
                "source_experiment": "EMAIL_QuantumText_Agent",
                "accuracy": r.get(f"{subgroup}_accuracy", np.nan),
                "precision": r.get(f"{subgroup}_precision", np.nan),
                "recall": r.get(f"{subgroup}_recall", np.nan),
                "f1": r.get(f"{subgroup}_f1", np.nan),
                "auc": r.get(f"{subgroup}_auc", np.nan),
                "train_time_sec": r.get("train_time_sec", np.nan),
                "what_it_tests": f"Generalization on {subgroup}-generated email evidence",
            })

    # E. Robustness/noisy evidence: compute from prediction files by flipping/masking scores
    meta = extract_agent_scores(cfg)
    if not meta.empty:
        rng = np.random.default_rng(SEED)
        for noise_rate in [0.05, 0.10, 0.20, 0.30]:
            temp = meta.copy()
            mask = rng.random(len(temp)) < noise_rate
            # simulate poisoned evidence by inverting a fraction of risk scores
            temp.loc[mask, "risk_score"] = 1 - temp.loc[mask, "risk_score"]
            y_true = temp["label"].values
            y_pred = (temp["risk_score"].values >= 0.5).astype(int)
            m = binary_metrics(y_true, y_pred, temp["risk_score"].values)
            rows.append({
                "perspective": "E. Robustness to Poisoned Evidence",
                "ablation": f"E{int(noise_rate*100)}",
                "variant": f"{int(noise_rate*100)}% evidence poisoning",
                "source_experiment": "agent_evidence_scores",
                "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "auc": m["auc"],
                "train_time_sec": np.nan,
                "what_it_tests": "Risk-score stability under misleading modality evidence",
            })

    # F. Explainability faithfulness proxies: confidence-risk alignment and top evidence reliability
    if not meta.empty:
        for modality in sorted(meta["modality"].unique()):
            sub = meta[meta["modality"] == modality]
            conf = np.abs(sub["risk_score"].values - 0.5)
            correct = (sub["agent_pred"].values == sub["label"].values).astype(int)
            # high confidence correctness proxy
            high = conf >= np.quantile(conf, 0.75)
            acc_high = correct[high].mean() if high.sum() else np.nan
            rows.append({
                "perspective": "F. Explainability Faithfulness",
                "ablation": f"F-{modality}",
                "variant": f"High-confidence explanation reliability: {modality}",
                "source_experiment": "agent_evidence_scores",
                "accuracy": acc_high,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "auc": np.nan,
                "train_time_sec": np.nan,
                "what_it_tests": "Whether high-confidence evidence explanations align with correct predictions",
            })

    ab = pd.DataFrame(rows)
    ab.to_csv(cfg.out_root / "results" / "grouped_ablation_A_to_F.csv", index=False)
    with pd.ExcelWriter(cfg.out_root / "tables" / "grouped_ablation_A_to_F.xlsx", engine="openpyxl") as writer:
        ab.to_excel(writer, sheet_name="all_ablation_results", index=False)
        for perspective, g in ab.groupby("perspective"):
            safe = perspective[:31].replace(".", "")
            g.to_excel(writer, sheet_name=safe, index=False)
    # plots per perspective
    for perspective, g in ab.groupby("perspective"):
        if "accuracy" in g.columns and g["accuracy"].notna().any():
            plt.figure(figsize=(max(10, len(g) * 0.7), 5))
            plt.bar(g["ablation"].astype(str), g["accuracy"].astype(float))
            plt.ylim(0, 1.05)
            plt.ylabel("Accuracy / Reliability")
            plt.title(perspective)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fname = perspective.lower().replace(" ", "_").replace(".", "").replace("-", "_")
            plt.savefig(cfg.out_root / "figures" / f"ablation_{fname}.png", dpi=300)
            plt.close()
    return ab


# =============================================================================
# Explainability reports
# =============================================================================

def generate_explainability_report(cfg: Config) -> pd.DataFrame:
    meta = extract_agent_scores(cfg)
    if meta.empty:
        return pd.DataFrame()
    # Reason codes based on modality and risk score thresholds.
    reason_map = {
        "url": "URL/domain structure indicates phishing risk patterns.",
        "sms": "SMS text contains smishing-style lexical or intent patterns.",
        "email": "Email body contains phishing or LLM-like deception patterns.",
        "qr": "QR visual/structural pattern indicates malicious QR evidence.",
    }
    rows = []
    for i, r in meta.head(1000).iterrows():
        risk = float(r["risk_score"])
        if risk >= 0.85:
            level = "critical"
        elif risk >= 0.65:
            level = "high"
        elif risk >= 0.45:
            level = "uncertain"
        else:
            level = "low"
        rows.append({
            "evidence_id": i,
            "agent": r["agent"],
            "modality": r["modality"],
            "risk_score": risk,
            "predicted_label": "phishing" if int(r["agent_pred"]) == 1 else "benign",
            "true_label": "phishing" if int(r["label"]) == 1 else "benign",
            "risk_level": level,
            "reason_code": reason_map.get(r["modality"], "Agent evidence indicates suspicious behavior."),
            "faithfulness_proxy": "high" if abs(risk - 0.5) > 0.35 else "medium/low",
        })
    out = pd.DataFrame(rows)
    out.to_csv(cfg.out_root / "explanations" / "explainability_agent_reason_codes.csv", index=False)
    return out


def generate_final_report(cfg: Config):
    result_files = list((cfg.out_root / "results").glob("*.csv"))
    sections = []
    all_results = []
    for p in result_files:
        try:
            df = pd.read_csv(p)
            all_results.append(df)
            sections.append(f"\n## {p.name}\n\n" + df.head(30).to_markdown(index=False))
        except Exception:
            pass
    if all_results:
        merged = pd.concat([d for d in all_results if "experiment" in d.columns], ignore_index=True)
        merged.to_csv(cfg.out_root / "results" / "ALL_RESULTS_MERGED.csv", index=False)
        plot_bar_metrics(merged.drop_duplicates("experiment"), cfg.out_root / "figures" / "ALL_RESULTS_METRICS.png", "All QTrustAgent-X Results")
    report = "# QTrustAgent-X Experimental Report\n\n"
    report += "This report summarizes trained agents, proposed quantum/graph variants, agentic arbitration, ablation perspectives A-F, timing, and saved figures.\n"
    report += "\n".join(sections)
    with open(cfg.out_root / "reports" / "QTrustAgentX_Final_Report.md", "w", encoding="utf-8") as f:
        f.write(report)


# =============================================================================
# Main orchestration
# =============================================================================

def run_all(cfg: Config):
    ensure_dirs(cfg)
    start_all = timer()
    outputs = {}
    outputs["url"] = run_url_experiments(cfg).to_dict(orient="records")
    outputs["sms"] = run_sms_experiments(cfg).to_dict(orient="records")
    outputs["email"] = run_email_experiments(cfg).to_dict(orient="records")
    outputs["qr"] = run_qr_cnn(cfg).to_dict(orient="records")
    outputs["agentic"] = run_agentic_arbitration(cfg).to_dict(orient="records")
    outputs["ablation"] = run_ablation_suite(cfg).to_dict(orient="records")
    outputs["explainability"] = generate_explainability_report(cfg).to_dict(orient="records")
    outputs["total_runtime_sec"] = elapsed(start_all)
    save_json(outputs, cfg.out_root / "results" / "qtrustagentx_all_outputs.json")
    pd.DataFrame([{"total_runtime_sec": outputs["total_runtime_sec"]}]).to_csv(cfg.out_root / "timing" / "total_runtime.csv", index=False)
    generate_final_report(cfg)
    print("\nDONE: QTrustAgent-X full pipeline completed.")
    print(f"Outputs saved to: {cfg.out_root}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "url", "sms", "email", "qr", "agentic", "ablation", "explain", "report"])
    ap.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--qr_limit", type=int, default=20000)
    ap.add_argument("--qr_epochs", type=int, default=10)
    ap.add_argument("--qr_batch_size", type=int, default=64)
    ap.add_argument("--max_text_features", type=int, default=25000)
    ap.add_argument("--svd_components", type=int, default=256)
    ap.add_argument("--quantum_dim", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_root=Path(args.data_root),
        out_root=Path(args.out_root),
        qr_limit=args.qr_limit,
        qr_epochs=args.qr_epochs,
        qr_batch_size=args.qr_batch_size,
        max_text_features=args.max_text_features,
        svd_components=args.svd_components,
        quantum_dim=args.quantum_dim,
    )
    ensure_dirs(cfg)
    if args.mode == "all":
        run_all(cfg)
    elif args.mode == "url":
        run_url_experiments(cfg)
    elif args.mode == "sms":
        run_sms_experiments(cfg)
    elif args.mode == "email":
        run_email_experiments(cfg)
    elif args.mode == "qr":
        run_qr_cnn(cfg)
    elif args.mode == "agentic":
        run_agentic_arbitration(cfg)
    elif args.mode == "ablation":
        run_ablation_suite(cfg)
    elif args.mode == "explain":
        generate_explainability_report(cfg)
    elif args.mode == "report":
        generate_final_report(cfg)


if __name__ == "__main__":
    main()
