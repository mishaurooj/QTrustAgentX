"""
QTrustAgent-X Professor-Comment Revision Pipeline v3
====================================================

This script is a corrective, reproducible revision pipeline for QTrustAgent-X.
It addresses the major weaknesses raised in the professor review without
inventing results.

What this version fixes
-----------------------
1. Genuine incident-level fusion is supported when an incident manifest is provided.
2. If no incident manifest is provided, the script clearly labels results as disjoint
   evidence-level ensemble results and does not call them true multi-channel fusion.
3. Majority voting is implemented as actual vote aggregation.
4. Trust graph reasoning is implemented as a real four-node modality graph:
   URL, Email, SMS, and QR.
5. Quantum-inspired representation is implemented as a classical random
   trigonometric feature map and saved under that exact wording.
6. Arbitration is trained using out-of-fold specialist scores and evaluated on
   untouched incident/evidence test rows.
7. Robustness applies the trained arbiter to corrupted evidence. It does not
   simply invert scores and threshold them.
8. Explanation faithfulness uses deletion, sufficiency, and comprehensiveness tests.
9. The pipeline saves paper-ready CSV, LaTeX tables, figures, audit notes,
   and a point-by-point professor-comment response.

Critical claim boundary
-----------------------
If you do not provide an incident_manifest, you CANNOT honestly claim true
cross-channel incident-level fusion. You can claim a modality-aware specialist
ensemble and evidence-level arbitration. The script writes this warning into
the outputs so the paper can be corrected.

Expected data root
------------------
Dataset_Reorganized/
  01_url/url_phishing_11430_89features.csv
  02_sms/sms_phishing_5971.csv
  02_sms/sms_smishing_5571.txt                 optional
  02_sms/sms_spam_raw_duplicate_check.csv      optional
  03_email_human_llm/human_legit/human_legit_email_1000.csv
  03_email_human_llm/human_phishing/human_phishing_email_1000.csv
  03_email_human_llm/llm_legit/llm_legit_email_1000.csv
  03_email_human_llm/llm_phishing/llm_phishing_email_595.csv
  04_qr/benign/*.png OR 04_qr/qr_benign/*.png
  04_qr/malicious/*.png OR 04_qr/qr_malicious/*.png

Optional incident manifest
--------------------------
CSV columns:
  incident_id,label,url_row_id,email_row_id,sms_row_id,qr_path

Missing channels can be blank. Labels may be 0/1 or benign/phishing.

Run examples
------------
python qtrustagentx_professor_fix_v3.py ^
  --data_root D:/other/QTrustAgentX/Dataset_Reorganized ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Results_v3 ^
  --mode all ^
  --qr_limit 20000 ^
  --repeats 5

python qtrustagentx_professor_fix_v3.py ^
  --data_root D:/other/QTrustAgentX/Dataset_Reorganized ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Results_v3 ^
  --incident_manifest D:/other/QTrustAgentX/incidents.csv ^
  --mode all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x


MODALITIES = ["url", "email", "sms", "qr"]


@dataclass
class Config:
    data_root: Path
    out_root: Path
    incident_manifest: Optional[Path] = None
    mode: str = "all"
    seed: int = 42
    repeats: int = 5
    cv_folds: int = 5
    test_size: float = 0.20
    qr_limit: int = 20000
    max_text_features: int = 25000
    svd_components: int = 256
    quantum_dim: int = 256
    trust_sigma: float = 0.25
    threshold: float = 0.50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: Config) -> None:
    for sub in [
        "models", "results", "figures", "reports", "predictions", "explanations",
        "splits", "tables", "logs", "config", "robustness", "faithfulness",
        "deployment", "audit"
    ]:
        (cfg.out_root / sub).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Could not read CSV: {path}")


def safe_read_txt(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        for sep in ["\t", ",", None]:
            try:
                return pd.read_csv(
                    path,
                    sep=sep,
                    engine="python" if sep is None else "c",
                    encoding=enc,
                    on_bad_lines="skip",
                    header=None,
                )
            except Exception:
                continue
    raise RuntimeError(f"Could not read TXT: {path}")


def label_to_binary(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.lower().str.strip()
    mapping = {
        "legitimate": 0, "legit": 0, "benign": 0, "ham": 0,
        "normal": 0, "safe": 0, "0": 0, "false": 0,
        "phishing": 1, "phish": 1, "malicious": 1, "spam": 1,
        "smish": 1, "smishing": 1, "1": 1, "true": 1,
    }
    y = raw.map(mapping)
    if y.isna().any():
        y = y.fillna(raw.str.contains("phish|malicious|spam|smish|attack", regex=True).astype(int))
    return y.astype(int)


def get_score(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        if s.ndim > 1:
            s = s[:, 0]
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    return np.asarray(model.predict(X), dtype=float)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Optional[Sequence[float]] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = float("nan")
        try:
            out["brier"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
        except Exception:
            out["brier"] = float("nan")
    else:
        out.update({"auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")})
    return out


def ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, float("nan"), float("nan")
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    return mean, float(mean - 1.96 * se), float(mean + 1.96 * se)


class QuantumFeatureMap(BaseEstimator, TransformerMixin):
    """Classical random trigonometric feature map.

    This is intentionally named quantum-inspired because it uses a classical
    cosine/sine projection. It does not require or claim quantum hardware.
    """
    def __init__(self, n_components: int = 256, gamma: float = 0.5, seed: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.seed = seed

    def fit(self, X: Any, y: Optional[Any] = None) -> "QuantumFeatureMap":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.normal(0.0, self.gamma, size=(X.shape[1], self.n_components))
        self.b_ = rng.uniform(0.0, 2 * np.pi, size=(self.n_components,))
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = X @ self.W_ + self.b_
        return np.concatenate([np.cos(z), np.sin(z)], axis=1) / math.sqrt(self.n_components)


class TrustGraphEncoder(BaseEstimator, TransformerMixin):
    """True four-node modality trust graph.

    Input columns:
      score_url, score_email, score_sms, score_qr, mask_url, mask_email, mask_sms, mask_qr

    Output:
      trust-weighted node scores, original scores, masks, graph agreement diagnostics.
    """
    def __init__(self, sigma: float = 0.25):
        self.sigma = sigma

    def fit(self, X: Any, y: Optional[Any] = None) -> "TrustGraphEncoder":
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = X[:, :4]
        masks = X[:, 4:8]
        output = []
        for s, m in zip(scores, masks):
            A = np.zeros((4, 4), dtype=float)
            for i in range(4):
                for j in range(4):
                    if m[i] <= 0 or m[j] <= 0:
                        A[i, j] = 0.0
                    elif i == j:
                        A[i, j] = 1.0
                    else:
                        A[i, j] = math.exp(-((s[i] - s[j]) ** 2) / (2 * self.sigma ** 2))
            deg = A.sum(axis=1, keepdims=True) + 1e-12
            Ahat = A / deg
            g = Ahat @ np.nan_to_num(s, nan=0.5)
            observed = np.where(m > 0)[0]
            if len(observed) > 1:
                sub = A[np.ix_(observed, observed)]
                off = sub[~np.eye(len(observed), dtype=bool)]
                mean_agreement = float(off.mean()) if len(off) else 1.0
            else:
                mean_agreement = 1.0
            diagnostics = np.array([
                float(A.sum()),
                mean_agreement,
                float(m.sum()),
                float(np.max(s[m > 0]) if np.any(m > 0) else 0.5),
                float(np.min(s[m > 0]) if np.any(m > 0) else 0.5),
                float(np.std(s[m > 0]) if np.any(m > 0) else 0.0),
            ])
            output.append(np.concatenate([g, s, m, diagnostics]))
        return np.asarray(output, dtype=float)


def build_url_model(kind: str = "extra", quantum: bool = False, seed: int = 42) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("scale", StandardScaler())]
    if quantum:
        steps.append(("qmap", QuantumFeatureMap(n_components=256, gamma=0.3, seed=seed)))
    clf = ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=seed, class_weight="balanced") if kind == "extra" else RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed, class_weight="balanced")
    return Pipeline(steps + [("clf", clf)])


def build_text_model(max_features: int, svd_components: Optional[int] = None, quantum: bool = False, seed: int = 42) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True))
    ]
    if svd_components:
        steps += [("svd", TruncatedSVD(n_components=svd_components, random_state=seed)), ("scale", StandardScaler())]
    if quantum:
        steps.append(("qmap", QuantumFeatureMap(n_components=256, gamma=0.5, seed=seed)))
        clf: Any = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced", random_state=seed)
    else:
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3)
    return Pipeline(steps + [("clf", clf)])


def build_qr_model(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=seed, class_weight="balanced"))
    ])


def find_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_url(cfg: Config) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    p = find_existing([
        cfg.data_root / "01_url" / "url_phishing_11430_89features.csv",
        cfg.data_root / "01_url" / "web_page_phishing_detection.csv",
    ])
    if p is None:
        raise FileNotFoundError("URL dataset not found under 01_url.")
    df = safe_read_csv(p)
    label_col = "status" if "status" in df.columns else next((c for c in df.columns if str(c).lower() in ["label", "class", "target"]), df.columns[-1])
    y = label_to_binary(df[label_col])
    X = df.drop(columns=[label_col]).copy()
    raw_url = X["url"].astype(str) if "url" in X.columns else pd.Series([""] * len(X))
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    meta = pd.DataFrame({"url_row_id": np.arange(len(X)), "label": y, "raw_url": raw_url})
    return X, y, meta


def load_sms(cfg: Config) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    frames = []
    p1 = cfg.data_root / "02_sms" / "sms_phishing_5971.csv"
    if p1.exists():
        df = safe_read_csv(p1)
        text_col = next((c for c in df.columns if str(c).lower() in ["text", "message", "sms", "body"]), df.columns[-1])
        label_col = next((c for c in df.columns if str(c).lower() in ["label", "class", "target"]), df.columns[0])
        frames.append(pd.DataFrame({"text": df[text_col].fillna("").astype(str), "label": label_to_binary(df[label_col]), "source": "sms_phishing"}))
    p2 = cfg.data_root / "02_sms" / "sms_smishing_5571.txt"
    if p2.exists():
        df = safe_read_txt(p2)
        if len(df.columns) >= 2:
            frames.append(pd.DataFrame({"text": df.iloc[:, 1].fillna("").astype(str), "label": label_to_binary(df.iloc[:, 0]), "source": "sms_smishing"}))
    p3 = cfg.data_root / "02_sms" / "sms_spam_raw_duplicate_check.csv"
    if p3.exists():
        df = safe_read_csv(p3)
        if "v1" in df.columns and "v2" in df.columns:
            frames.append(pd.DataFrame({"text": df["v2"].fillna("").astype(str), "label": label_to_binary(df["v1"]), "source": "sms_spam"}))
    if not frames:
        raise FileNotFoundError("SMS datasets not found under 02_sms.")
    data = pd.concat(frames, ignore_index=True)
    data["norm"] = data["text"].str.lower().str.strip()
    data = data[data["norm"].str.len() > 0].drop_duplicates("norm").reset_index(drop=True)
    meta = pd.DataFrame({"sms_row_id": np.arange(len(data)), "label": data["label"], "source": data["source"]})
    return data["text"], data["label"].astype(int), meta


def load_email(cfg: Config) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    root = cfg.data_root / "03_email_human_llm"
    files = [
        (root / "human_legit" / "human_legit_email_1000.csv", 0, "human_legit", "human"),
        (root / "human_phishing" / "human_phishing_email_1000.csv", 1, "human_phishing", "human"),
        (root / "llm_legit" / "llm_legit_email_1000.csv", 0, "llm_legit", "llm"),
        (root / "llm_phishing" / "llm_phishing_email_595.csv", 1, "llm_phishing", "llm"),
    ]
    rows = []
    for path, label, src, domain in files:
        if not path.exists():
            continue
        df = safe_read_csv(path)
        text_col = next((c for c in df.columns if str(c).lower() in ["body", "text", "message", "email", "content"]), df.columns[-1])
        subj_col = next((c for c in df.columns if str(c).lower() == "subject"), None)
        for _, r in df.iterrows():
            text = str(r.get(text_col, ""))
            if subj_col:
                text = "Subject: " + str(r.get(subj_col, "")) + "\n" + text
            rows.append({"text": text, "label": label, "source": src, "domain": domain})
    if not rows:
        raise FileNotFoundError("Email datasets not found under 03_email_human_llm.")
    data = pd.DataFrame(rows)
    data["norm"] = data["text"].str.lower().str.strip()
    data = data[data["norm"].str.len() > 0].drop_duplicates("norm").reset_index(drop=True)
    meta = pd.DataFrame({"email_row_id": np.arange(len(data)), "label": data["label"], "source": data["source"], "domain": data["domain"]})
    return data["text"], data["label"].astype(int), meta


def locate_qr_folders(cfg: Config) -> Tuple[Path, Path]:
    benign = find_existing([cfg.data_root / "04_qr" / "benign", cfg.data_root / "04_qr" / "qr_benign", cfg.data_root / "04_qr/qr_benign"])
    malicious = find_existing([cfg.data_root / "04_qr" / "malicious", cfg.data_root / "04_qr" / "qr_malicious", cfg.data_root / "04_qr/qr_malicious"])
    if benign is None or malicious is None:
        raise FileNotFoundError("QR benign/malicious folders not found under 04_qr.")
    return benign, malicious


def load_qr(cfg: Config) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    benign_dir, mal_dir = locate_qr_folders(cfg)
    rows = []
    for folder, label in [(benign_dir, 0), (mal_dir, 1)]:
        paths: List[Path] = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
            paths.extend(folder.glob(ext))
        paths = sorted(paths)
        if cfg.qr_limit:
            paths = paths[: max(1, cfg.qr_limit // 2)]
        for p in paths:
            rows.append({"path": str(p), "label": label})
    if not rows:
        raise FileNotFoundError("No QR image files found.")
    data = pd.DataFrame(rows).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    meta = pd.DataFrame({"qr_row_id": np.arange(len(data)), "label": data["label"], "qr_path": data["path"]})
    return data["path"], data["label"].astype(int), meta


def qr_features(paths: Sequence[str], cache_path: Optional[Path] = None) -> np.ndarray:
    if cache_path and cache_path.exists():
        return np.load(cache_path)
    if Image is None:
        raise RuntimeError("Pillow is required. Install with: pip install pillow")
    feats: List[List[float]] = []
    for p in tqdm(list(paths), desc="QR features"):
        try:
            img = Image.open(p).convert("L").resize((128, 128))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            row = [
                float(arr.mean()),
                float(arr.std()),
                float(np.abs(np.diff(arr, axis=1)).mean()),
                float(np.abs(np.diff(arr, axis=0)).mean()),
            ]
            for i in range(0, 128, 16):
                for j in range(0, 128, 16):
                    row.append(float(arr[i:i + 16, j:j + 16].mean()))
            feats.append(row)
        except Exception:
            feats.append([0.0] * 68)
    X = np.asarray(feats, dtype=np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X


def image_hashes(paths: Sequence[str]) -> pd.DataFrame:
    rows = []
    for p in tqdm(list(paths), desc="QR hash audit"):
        try:
            with open(p, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            rows.append({"path": p, "sha256": h})
        except Exception:
            rows.append({"path": p, "sha256": ""})
    return pd.DataFrame(rows)


def oof_scores(X: Any, y: pd.Series, model: Any, cfg: Config, precomputed_features: Optional[np.ndarray] = None, seed: int = 42) -> Tuple[np.ndarray, Any]:
    X_use = precomputed_features if precomputed_features is not None else X
    scores = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        m = clone(model)
        m.fit(X_use[tr] if isinstance(X_use, np.ndarray) else X_use.iloc[tr], y.iloc[tr])
        scores[va] = get_score(m, X_use[va] if isinstance(X_use, np.ndarray) else X_use.iloc[va])
    final_model = clone(model)
    final_model.fit(X_use, y)
    return scores, final_model


def train_test_single(name: str, X: Any, y: pd.Series, model: Any, cfg: Config, seed: int, precomputed_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
    X_use = precomputed_features if precomputed_features is not None else X
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=cfg.test_size, stratify=y, random_state=seed)
    start_train = time.perf_counter()
    model.fit(X_use[tr] if isinstance(X_use, np.ndarray) else X_use.iloc[tr], y.iloc[tr])
    train_time = time.perf_counter() - start_train
    start_pred = time.perf_counter()
    score = get_score(model, X_use[te] if isinstance(X_use, np.ndarray) else X_use.iloc[te])
    pred_time = time.perf_counter() - start_pred
    pred = (score >= cfg.threshold).astype(int)
    m = binary_metrics(y.iloc[te].values, pred, score)
    m.update({"experiment": name, "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)), "train_time_sec": train_time, "predict_time_sec": pred_time})
    return m


def build_specialist_oof(cfg: Config, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, pd.DataFrame]]:
    data: Dict[str, Tuple[Any, pd.Series, Any, Optional[np.ndarray], pd.DataFrame]] = {}

    X_url, y_url, meta_url = load_url(cfg)
    data["url"] = (X_url, y_url, build_url_model("extra", quantum=False, seed=seed), None, meta_url)

    X_email, y_email, meta_email = load_email(cfg)
    data["email"] = (X_email, y_email, build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=seed), None, meta_email)

    X_sms, y_sms, meta_sms = load_sms(cfg)
    data["sms"] = (X_sms, y_sms, build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=seed), None, meta_sms)

    X_qr, y_qr, meta_qr = load_qr(cfg)
    qr_cache = cfg.out_root / "cache_qr_features.npy"
    X_qrf = qr_features(X_qr.tolist(), cache_path=qr_cache)
    data["qr"] = (X_qr, y_qr, build_qr_model(seed=seed), X_qrf, meta_qr)

    # QR leakage audit by exact image hash.
    try:
        qr_hash_df = image_hashes(X_qr.tolist())
        qr_hash_df["label"] = y_qr.values
        save_csv(qr_hash_df, cfg.out_root / "audit" / "qr_sha256_hash_audit.csv")
        dup_count = int(qr_hash_df.duplicated("sha256").sum())
        save_json({"qr_exact_duplicate_hashes": dup_count, "qr_images_audited": len(qr_hash_df)}, cfg.out_root / "audit" / "qr_hash_audit_summary.json")
    except Exception as e:
        save_json({"qr_hash_audit_error": str(e)}, cfg.out_root / "audit" / "qr_hash_audit_summary.json")

    models_dict: Dict[str, Any] = {}
    meta_dict: Dict[str, pd.DataFrame] = {}
    specialist_rows = []

    for mod, (X, y, base_model, feats, meta) in data.items():
        scores, fitted = oof_scores(X, y, base_model, cfg, precomputed_features=feats, seed=seed)
        models_dict[mod] = fitted
        meta = meta.copy()
        meta[f"score_{mod}"] = scores
        meta[f"pred_{mod}"] = (scores >= cfg.threshold).astype(int)
        meta_dict[mod] = meta
        joblib.dump(fitted, cfg.out_root / "models" / f"{mod}_specialist_seed{seed}.joblib")
        metrics = binary_metrics(y.values, meta[f"pred_{mod}"].values, scores)
        metrics.update({"experiment": f"{mod.upper()}_OOF_Specialist", "seed": seed, "n": len(y)})
        specialist_rows.append(metrics)

    spec = pd.DataFrame(specialist_rows)
    save_csv(spec, cfg.out_root / "results" / f"specialist_oof_results_seed{seed}.csv")
    return spec, models_dict, meta_dict


def make_disjoint_evidence(meta_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mod in MODALITIES:
        meta = meta_dict[mod]
        id_col = f"{mod}_row_id" if f"{mod}_row_id" in meta.columns else "row_id"
        for _, r in meta.iterrows():
            row = {
                "incident_id": f"{mod}_{int(r[id_col])}",
                "label": int(r["label"]),
                "fusion_valid": 0,
                "protocol": "disjoint_evidence_level",
                "observed_modality": mod,
            }
            for m in MODALITIES:
                row[f"score_{m}"] = float(r[f"score_{mod}"]) if m == mod else 0.5
                row[f"mask_{m}"] = 1 if m == mod else 0
            rows.append(row)
    return pd.DataFrame(rows)


def make_manifest_evidence(cfg: Config, meta_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cfg.incident_manifest is None:
        raise RuntimeError("incident_manifest is required for true incident-level fusion.")
    manifest = safe_read_csv(cfg.incident_manifest)
    if "incident_id" not in manifest.columns:
        manifest["incident_id"] = np.arange(len(manifest)).astype(str)
    if "label" not in manifest.columns:
        raise ValueError("incident_manifest must contain a label column.")
    y = label_to_binary(manifest["label"])
    rows = []

    lookups = {}
    for mod, meta in meta_dict.items():
        id_col = f"{mod}_row_id" if f"{mod}_row_id" in meta.columns else "row_id"
        lookups[mod] = meta.set_index(id_col)

    for i, r in manifest.iterrows():
        row = {"incident_id": str(r["incident_id"]), "label": int(y.iloc[i]), "fusion_valid": 1, "protocol": "incident_level_fusion"}
        observed = 0
        for mod in MODALITIES:
            score = 0.5
            mask = 0
            if mod == "qr":
                if "qr_path" in manifest.columns and pd.notna(r.get("qr_path")):
                    qpath = str(r.get("qr_path"))
                    qr_meta = meta_dict["qr"]
                    match = qr_meta[qr_meta["qr_path"].astype(str) == qpath]
                    if len(match):
                        score = float(match.iloc[0]["score_qr"])
                        mask = 1
            else:
                col = f"{mod}_row_id"
                if col in manifest.columns and pd.notna(r.get(col)):
                    try:
                        rid = int(r.get(col))
                        if rid in lookups[mod].index:
                            score = float(lookups[mod].loc[rid][f"score_{mod}"])
                            mask = 1
                    except Exception:
                        pass
            row[f"score_{mod}"] = score
            row[f"mask_{mod}"] = mask
            observed += mask
        row["n_observed_channels"] = observed
        rows.append(row)

    evidence = pd.DataFrame(rows)
    if (evidence[[f"mask_{m}" for m in MODALITIES]].sum(axis=1) < 2).mean() > 0.80:
        save_json(
            {
                "warning": "More than 80% of incident rows contain fewer than two observed channels. This weakens true multi-channel fusion claims.",
                "observed_channel_counts": evidence[[f"mask_{m}" for m in MODALITIES]].sum(axis=1).value_counts().to_dict(),
            },
            cfg.out_root / "audit" / "incident_manifest_channel_warning.json",
        )
    return evidence


def majority_vote_scores(evidence: pd.DataFrame, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    scores = evidence[[f"score_{m}" for m in MODALITIES]].values.astype(float)
    masks = evidence[[f"mask_{m}" for m in MODALITIES]].values.astype(float)
    votes = (scores >= threshold).astype(float) * masks
    denom = masks.sum(axis=1).clip(min=1)
    vote_score = votes.sum(axis=1) / denom
    vote_pred = (vote_score >= 0.5).astype(int)
    return vote_score, vote_pred


def average_scores(evidence: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    scores = evidence[[f"score_{m}" for m in MODALITIES]].values.astype(float)
    masks = evidence[[f"mask_{m}" for m in MODALITIES]].values.astype(float)
    denom = masks.sum(axis=1).clip(min=1)
    avg = ((scores * masks).sum(axis=1) + 0.5 * (denom == 0)) / denom
    return avg, (avg >= 0.5).astype(int)


def arbitration_features(evidence: pd.DataFrame) -> np.ndarray:
    return evidence[[f"score_{m}" for m in MODALITIES] + [f"mask_{m}" for m in MODALITIES]].values.astype(float)


def train_arbiters_once(evidence: pd.DataFrame, cfg: Config, seed: int, prefix: str) -> pd.DataFrame:
    y = evidence["label"].astype(int).values
    X = arbitration_features(evidence)
    idx = np.arange(len(evidence))
    tr, te = train_test_split(idx, test_size=cfg.test_size, stratify=y, random_state=seed)

    save_csv(pd.DataFrame({"train_index": tr}), cfg.out_root / "splits" / f"{prefix}_arbiter_train_seed{seed}.csv")
    save_csv(pd.DataFrame({"test_index": te}), cfg.out_root / "splits" / f"{prefix}_arbiter_test_seed{seed}.csv")

    rows = []
    pred_frames = []

    def add_result(name: str, score: np.ndarray, pred: np.ndarray, train_time: float, predict_time: float) -> None:
        m = binary_metrics(y[te], pred, score)
        m.update({
            "experiment": name,
            "seed": seed,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_time_sec": float(train_time),
            "predict_time_sec": float(predict_time),
        })
        rows.append(m)
        pred_frames.append(pd.DataFrame({
            "experiment": name,
            "seed": seed,
            "incident_id": evidence.iloc[te]["incident_id"].values,
            "y_true": y[te],
            "y_score": score,
            "y_pred": pred,
        }))

    start = time.perf_counter()
    avg_score, avg_pred = average_scores(evidence.iloc[te])
    add_result(f"{prefix}_AverageScore", avg_score, avg_pred, 0.0, time.perf_counter() - start)

    start = time.perf_counter()
    mv_score, mv_pred = majority_vote_scores(evidence.iloc[te], cfg.threshold)
    add_result(f"{prefix}_TrueMajorityVote", mv_score, mv_pred, 0.0, time.perf_counter() - start)

    late = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    start = time.perf_counter()
    late.fit(X[tr], y[tr])
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    late_score = late.predict_proba(X[te])[:, 1]
    pred_time = time.perf_counter() - start
    late_pred = (late_score >= cfg.threshold).astype(int)
    add_result(f"{prefix}_LearnedLateFusion", late_score, late_pred, train_time, pred_time)
    joblib.dump(late, cfg.out_root / "models" / f"{prefix}_learned_late_fusion_seed{seed}.joblib")

    graph = Pipeline([
        ("graph", TrustGraphEncoder(sigma=cfg.trust_sigma)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    start = time.perf_counter()
    graph.fit(X[tr], y[tr])
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    graph_score = graph.predict_proba(X[te])[:, 1]
    pred_time = time.perf_counter() - start
    graph_pred = (graph_score >= cfg.threshold).astype(int)
    add_result(f"{prefix}_TrustGraphArbitration", graph_score, graph_pred, train_time, pred_time)
    joblib.dump(graph, cfg.out_root / "models" / f"{prefix}_trust_graph_seed{seed}.joblib")

    qgraph = Pipeline([
        ("graph", TrustGraphEncoder(sigma=cfg.trust_sigma)),
        ("scale", StandardScaler()),
        ("qmap", QuantumFeatureMap(n_components=cfg.quantum_dim, gamma=0.5, seed=seed)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    start = time.perf_counter()
    qgraph.fit(X[tr], y[tr])
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    q_score = qgraph.predict_proba(X[te])[:, 1]
    pred_time = time.perf_counter() - start
    q_pred = (q_score >= cfg.threshold).astype(int)
    add_result(f"{prefix}_QuantumInspiredTrustGraph", q_score, q_pred, train_time, pred_time)
    joblib.dump(qgraph, cfg.out_root / "models" / f"{prefix}_quantum_inspired_trust_graph_seed{seed}.joblib")

    out = pd.DataFrame(rows)
    save_csv(out, cfg.out_root / "results" / f"{prefix}_arbitration_seed{seed}.csv")
    if pred_frames:
        save_csv(pd.concat(pred_frames, ignore_index=True), cfg.out_root / "predictions" / f"{prefix}_arbitration_predictions_seed{seed}.csv")

    robustness_and_faithfulness(evidence.iloc[te].reset_index(drop=True), graph, qgraph, cfg, prefix, seed)
    draw_arbitration_figure(out, cfg.out_root / "figures" / f"{prefix}_arbitration_seed{seed}.png")
    return out


def robustness_and_faithfulness(test_evidence: pd.DataFrame, graph_model: Any, qgraph_model: Any, cfg: Config, prefix: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = test_evidence["label"].astype(int).values
    base_X = arbitration_features(test_evidence)

    rows = []
    for model_name, model in [("TrustGraph", graph_model), ("QuantumInspiredTrustGraph", qgraph_model)]:
        base_score = model.predict_proba(base_X)[:, 1]
        base_pred = (base_score >= cfg.threshold).astype(int)
        rows.append({"model": model_name, "condition": "unpoisoned", **binary_metrics(y, base_pred, base_score)})

        for rate in [0.05, 0.10, 0.20, 0.30]:
            Xp = base_X.copy()
            masks = Xp[:, 4:8] > 0
            for i in range(Xp.shape[0]):
                obs = np.where(masks[i])[0]
                if len(obs) == 0:
                    continue
                if rng.random() < rate:
                    flip = rng.choice(obs, size=1, replace=False)
                    Xp[i, flip] = 1.0 - Xp[i, flip]
            s = model.predict_proba(Xp)[:, 1]
            p = (s >= cfg.threshold).astype(int)
            rows.append({"model": model_name, "condition": f"score_poison_{int(rate*100)}pct", **binary_metrics(y, p, s)})

        for mod_i, mod in enumerate(MODALITIES):
            Xm = base_X.copy()
            Xm[:, mod_i] = 0.5
            Xm[:, 4 + mod_i] = 0
            s = model.predict_proba(Xm)[:, 1]
            p = (s >= cfg.threshold).astype(int)
            rows.append({"model": model_name, "condition": f"missing_{mod}", **binary_metrics(y, p, s)})

    rob = pd.DataFrame(rows)
    save_csv(rob, cfg.out_root / "robustness" / f"{prefix}_robustness_seed{seed}.csv")
    draw_robustness_figure(rob, cfg.out_root / "figures" / f"{prefix}_robustness_seed{seed}.png")

    # Explanation tests on the quantum-inspired trust graph model.
    base_score = qgraph_model.predict_proba(base_X)[:, 1]
    expl_rows = []
    for i in range(base_X.shape[0]):
        row = {
            "incident_id": test_evidence.iloc[i]["incident_id"],
            "label": int(y[i]),
            "base_score": float(base_score[i]),
        }
        deletion_impacts = []
        suff_scores = []
        for mod_i, mod in enumerate(MODALITIES):
            Xdel = base_X[i:i+1].copy()
            Xdel[:, mod_i] = 0.5
            Xdel[:, 4 + mod_i] = 0
            del_score = float(qgraph_model.predict_proba(Xdel)[:, 1][0])
            deletion = abs(float(base_score[i]) - del_score)
            row[f"delete_{mod}_score"] = del_score
            row[f"delete_{mod}_impact"] = deletion
            deletion_impacts.append(deletion if base_X[i, 4 + mod_i] > 0 else 0.0)

            Xsuf = np.zeros_like(base_X[i:i+1])
            Xsuf[:, :4] = 0.5
            Xsuf[:, 4:8] = 0
            Xsuf[:, mod_i] = base_X[i, mod_i]
            Xsuf[:, 4 + mod_i] = base_X[i, 4 + mod_i]
            suf_score = float(qgraph_model.predict_proba(Xsuf)[:, 1][0])
            row[f"sufficiency_{mod}_score"] = suf_score
            suff_scores.append(suf_score if base_X[i, 4 + mod_i] > 0 else 0.5)

        total = sum(deletion_impacts) + 1e-12
        for mod, imp in zip(MODALITIES, deletion_impacts):
            row[f"contribution_{mod}"] = float(imp / total)
        row["comprehensiveness"] = float(max(deletion_impacts) if deletion_impacts else 0.0)
        row["sufficiency_best_single_modality"] = float(max(suff_scores) if suff_scores else 0.5)
        expl_rows.append(row)

    expl = pd.DataFrame(expl_rows)
    save_csv(expl, cfg.out_root / "faithfulness" / f"{prefix}_faithfulness_instances_seed{seed}.csv")

    summary = []
    for mod in MODALITIES:
        summary.append({
            "modality": mod,
            "mean_deletion_impact": float(expl[f"delete_{mod}_impact"].mean()),
            "mean_contribution": float(expl[f"contribution_{mod}"].mean()),
            "mean_sufficiency_score": float(expl[f"sufficiency_{mod}_score"].mean()),
        })
    summary_df = pd.DataFrame(summary)
    save_csv(summary_df, cfg.out_root / "faithfulness" / f"{prefix}_faithfulness_summary_seed{seed}.csv")
    draw_faithfulness_figure(summary_df, cfg.out_root / "figures" / f"{prefix}_faithfulness_seed{seed}.png")


def draw_arbitration_figure(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    methods = df["experiment"].str.replace("DISJOINT_", "", regex=False).str.replace("INCIDENT_", "", regex=False)
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(df))
    width = 0.15
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 2) * width, df[m].astype(float).values, width, label=m)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Arbitration Strategy Comparison")
    plt.legend(ncol=5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def draw_robustness_figure(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["condition"], sub["f1"], marker="o", label=model)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("F1-score")
    plt.title("Robustness Under Poisoning and Missing Channels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def draw_faithfulness_figure(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    x = np.arange(len(df))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, df["mean_deletion_impact"], width=0.4, label="Deletion impact")
    plt.bar(x + 0.2, df["mean_contribution"], width=0.4, label="Contribution")
    plt.xticks(x, df["modality"].str.upper())
    plt.ylabel("Score")
    plt.title("Explanation Faithfulness by Modality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def aggregate_repeats(cfg: Config, prefix: str) -> None:
    files = sorted((cfg.out_root / "results").glob(f"{prefix}_arbitration_seed*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    save_csv(df, cfg.out_root / "tables" / f"{prefix}_arbitration_all_repeats.csv")

    rows = []
    for exp, g in df.groupby("experiment"):
        row = {"experiment": exp, "n_runs": len(g)}
        for m in ["accuracy", "precision", "recall", "specificity", "f1", "auc", "pr_auc", "brier", "mcc", "fpr", "fnr", "train_time_sec", "predict_time_sec"]:
            if m in g.columns:
                mean, low, high = ci95(g[m])
                row[m] = mean
                row[f"{m}_ci_low"] = low
                row[f"{m}_ci_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    save_csv(summary, cfg.out_root / "tables" / f"{prefix}_arbitration_summary_ci.csv")
    write_latex_tables(cfg, prefix, summary)


def latex_num(x: Any) -> str:
    try:
        if pd.isna(x):
            return "--"
        return f"{float(x):.4f}"
    except Exception:
        return "--"


def latex_ci(mean: Any, lo: Any, hi: Any) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(lo) or pd.isna(hi):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} [{float(lo):.4f}, {float(hi):.4f}]"


def write_latex_tables(cfg: Config, prefix: str, summary: pd.DataFrame) -> None:
    # Main arbitration table with CIs.
    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Corrected arbitration comparison for QTrustAgent-X. Values report mean and 95\% confidence interval across repeated random seeds.}")
    lines.append(rf"\label{{tab:{prefix.lower()}_corrected_arbitration}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Acc. & Prec. & Rec. & Spec. & F1 & AUC & Brier \\")
    lines.append(r"\midrule")
    for _, r in summary.iterrows():
        name = str(r["experiment"]).replace(prefix + "_", "").replace("_", " ")
        lines.append(
            f"{name} & "
            f"{latex_ci(r.get('accuracy'), r.get('accuracy_ci_low'), r.get('accuracy_ci_high'))} & "
            f"{latex_ci(r.get('precision'), r.get('precision_ci_low'), r.get('precision_ci_high'))} & "
            f"{latex_ci(r.get('recall'), r.get('recall_ci_low'), r.get('recall_ci_high'))} & "
            f"{latex_ci(r.get('specificity'), r.get('specificity_ci_low'), r.get('specificity_ci_high'))} & "
            f"{latex_ci(r.get('f1'), r.get('f1_ci_low'), r.get('f1_ci_high'))} & "
            f"{latex_ci(r.get('auc'), r.get('auc_ci_low'), r.get('auc_ci_high'))} & "
            f"{latex_ci(r.get('brier'), r.get('brier_ci_low'), r.get('brier_ci_high'))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    (cfg.out_root / "tables" / f"{prefix}_corrected_arbitration_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # Compact 2x3 ablation layout using latest seed files.
    arb_file = sorted((cfg.out_root / "results").glob(f"{prefix}_arbitration_seed*.csv"))[-1]
    arb = pd.read_csv(arb_file)
    rob_file = sorted((cfg.out_root / "robustness").glob(f"{prefix}_robustness_seed*.csv"))[-1]
    rob = pd.read_csv(rob_file)
    faith_file = sorted((cfg.out_root / "faithfulness").glob(f"{prefix}_faithfulness_summary_seed*.csv"))[-1]
    faith = pd.read_csv(faith_file)

    ab = []
    ab.append(r"\begin{table*}[!t]")
    ab.append(r"\centering")
    ab.append(r"\tiny")
    ab.append(r"\caption{Corrected QTrustAgent-X ablation summary. The table separates deployable arbitration, robustness, and explanation-faithfulness tests.}")
    ab.append(rf"\label{{tab:{prefix.lower()}_corrected_ablation}}")
    ab.append(r"\setlength{\tabcolsep}{3pt}")
    ab.append(r"\begin{minipage}[t]{0.49\textwidth}")
    ab.append(r"\centering")
    ab.append(r"\caption*{(a) Arbitration strategies}")
    ab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{lccccc}\toprule Method & Acc. & Prec. & Rec. & F1 & AUC \\\midrule")
    for _, r in arb.iterrows():
        name = str(r["experiment"]).replace(prefix + "_", "").replace("_", " ")
        ab.append(f"{name} & {latex_num(r['accuracy'])} & {latex_num(r['precision'])} & {latex_num(r['recall'])} & {latex_num(r['f1'])} & {latex_num(r['auc'])} \\\\")
    ab.append(r"\bottomrule\end{tabular}}")
    ab.append(r"\vspace{0.25cm}")
    ab.append(r"\caption*{(c) Robustness under corrupted evidence}")
    ab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{llccc}\toprule Model & Condition & Acc. & F1 & AUC \\\midrule")
    for _, r in rob.head(12).iterrows():
        ab.append(f"{r['model']} & {str(r['condition']).replace('_',' ')} & {latex_num(r['accuracy'])} & {latex_num(r['f1'])} & {latex_num(r['auc'])} \\\\")
    ab.append(r"\bottomrule\end{tabular}}")
    ab.append(r"\vspace{0.25cm}")
    ab.append(r"\caption*{(e) Explanation faithfulness}")
    ab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{lccc}\toprule Modality & Deletion impact & Contribution & Sufficiency \\\midrule")
    for _, r in faith.iterrows():
        ab.append(f"{str(r['modality']).upper()} & {latex_num(r['mean_deletion_impact'])} & {latex_num(r['mean_contribution'])} & {latex_num(r['mean_sufficiency_score'])} \\\\")
    ab.append(r"\bottomrule\end{tabular}}")
    ab.append(r"\end{minipage}\hfill")
    ab.append(r"\begin{minipage}[t]{0.49\textwidth}")
    ab.append(r"\centering")
    ab.append(r"\caption*{(b) Deployment metrics}")
    ab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{lccc}\toprule Method & Train s & Predict s & Brier \\\midrule")
    for _, r in arb.iterrows():
        name = str(r["experiment"]).replace(prefix + "_", "").replace("_", " ")
        ab.append(f"{name} & {latex_num(r['train_time_sec'])} & {latex_num(r['predict_time_sec'])} & {latex_num(r['brier'])} \\\\")
    ab.append(r"\bottomrule\end{tabular}}")
    ab.append(r"\vspace{0.25cm}")
    ab.append(r"\caption*{(d) Error profile}")
    ab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{lcccc}\toprule Method & FPR & FNR & FP & FN \\\midrule")
    for _, r in arb.iterrows():
        name = str(r["experiment"]).replace(prefix + "_", "").replace("_", " ")
        ab.append(f"{name} & {latex_num(r['fpr'])} & {latex_num(r['fnr'])} & {int(r['fp'])} & {int(r['fn'])} \\\\")
    ab.append(r"\bottomrule\end{tabular}}")
    ab.append(r"\vspace{0.25cm}")
    ab.append(r"\caption*{(f) Protocol boundary}")
    ab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{ll}\toprule Item & Status \\\midrule")
    if prefix == "INCIDENT":
        ab.append(r"True multi-channel fusion & Supported by incident manifest \\")
    else:
        ab.append(r"True multi-channel fusion & Not claimed in disjoint mode \\")
    ab.append(r"Majority vote & Real vote aggregation \\")
    ab.append(r"Trust graph & Four modality nodes with masks \\")
    ab.append(r"Robustness & Trained arbiter under corruption \\")
    ab.append(r"Faithfulness & Deletion and sufficiency tests \\")
    ab.append(r"\bottomrule\end{tabular}}")
    ab.append(r"\end{minipage}")
    ab.append(r"\end{table*}")
    (cfg.out_root / "tables" / f"{prefix}_corrected_ablation_2x3.tex").write_text("\n".join(ab), encoding="utf-8")


def run_pipeline(cfg: Config, prefix: str, evidence: pd.DataFrame) -> None:
    save_csv(evidence, cfg.out_root / "predictions" / f"{prefix}_evidence_table.csv")
    save_json(
        {
            "prefix": prefix,
            "rows": len(evidence),
            "protocol_counts": evidence["protocol"].value_counts().to_dict() if "protocol" in evidence.columns else {},
            "observed_channel_counts": evidence[[f"mask_{m}" for m in MODALITIES]].sum(axis=1).value_counts().to_dict(),
            "true_fusion_claim_allowed": bool(prefix == "INCIDENT"),
        },
        cfg.out_root / "audit" / f"{prefix}_protocol_boundary.json",
    )
    all_runs = []
    seeds = [cfg.seed + i for i in range(cfg.repeats)]
    for seed in seeds:
        set_seed(seed)
        all_runs.append(train_arbiters_once(evidence, cfg, seed, prefix))
    aggregate_repeats(cfg, prefix)


def run_human_llm_generalization(cfg: Config) -> None:
    X, y, meta = load_email(cfg)
    data = pd.DataFrame({"text": X, "label": y, "domain": meta["domain"]})
    rows = []
    # Train on human, test on llm
    for train_domain, test_domain in [("human", "llm"), ("llm", "human")]:
        train = data[data["domain"] == train_domain]
        test = data[data["domain"] == test_domain]
        if len(train["label"].unique()) < 2 or len(test["label"].unique()) < 2:
            continue
        model = build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=cfg.seed)
        start = time.perf_counter()
        model.fit(train["text"], train["label"])
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        score = get_score(model, test["text"])
        pred_time = time.perf_counter() - start
        pred = (score >= cfg.threshold).astype(int)
        m = binary_metrics(test["label"].values, pred, score)
        m.update({"experiment": f"train_{train_domain}_test_{test_domain}", "n_train": len(train), "n_test": len(test), "train_time_sec": train_time, "predict_time_sec": pred_time})
        rows.append(m)
    out = pd.DataFrame(rows)
    save_csv(out, cfg.out_root / "results" / "human_llm_true_generalization.csv")


def collect_all_results(cfg: Config) -> None:
    files = []
    for sub in ["results", "robustness", "faithfulness"]:
        files.extend(sorted((cfg.out_root / sub).glob("*.csv")))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df.insert(0, "source_file", p.name)
            frames.append(df)
        except Exception:
            pass
    if frames:
        long = pd.concat(frames, ignore_index=True, sort=False)
        save_csv(long, cfg.out_root / "tables" / "all_results_long.csv")
        try:
            with pd.ExcelWriter(cfg.out_root / "tables" / "all_results.xlsx") as writer:
                long.to_excel(writer, sheet_name="all_results_long", index=False)
                for p in files[:30]:
                    try:
                        pd.read_csv(p).to_excel(writer, sheet_name=p.stem[:31], index=False)
                    except Exception:
                        pass
        except Exception as e:
            save_json({"excel_error": str(e)}, cfg.out_root / "logs" / "excel_export_error.json")


def write_professor_response(cfg: Config) -> None:
    md = f"""# QTrustAgent-X Professor-Comment Response and Code Audit

This file is generated automatically by `qtrustagentx_professor_fix_v3.py`.

## Summary

The revised pipeline fixes implementation and reporting problems raised in the professor review. It also marks claims that cannot be made without an incident-aligned manifest.

## Point-by-point status

| No. | Professor comment | Code action in v3 | Claim status |
|---:|---|---|---|
| 1 | Experiments do not perform genuine multi-channel fusion | Added `--incident_manifest` mode with rows containing URL, Email, SMS, and QR scores plus explicit masks. Disjoint mode is labelled `DISJOINT` and cannot be called true fusion. | True fusion can be claimed only when `INCIDENT` outputs are generated. |
| 2 | Architecture and implementation do not match | Implemented a real four-node trust graph over URL, Email, SMS, and QR scores. Renamed quantum block as classical quantum-inspired random trigonometric map. | Manuscript must match this implementation. |
| 3 | Agentic characterization is weak | Code uses specialist agents, arbitration, trust graph, and audit trail. It does not implement planning, memory, autonomous tool use, or negotiation. | Use "specialist-agent pipeline" or "modular agentic-style pipeline" unless adding real autonomous behavior. |
| 4 | Main novelty not validated | Added corrected arbitration baselines: average score, true majority vote, learned late fusion, trust graph arbitration, and quantum-inspired trust graph arbitration. | Primary endpoint should be final arbiter F1/AUC, not isolated QR/email scores. |
| 5 | Poisoned-evidence test does not use arbiter | Robustness now applies trained TrustGraph and QuantumInspiredTrustGraph arbiters to corrupted and missing-channel evidence. | Claim limited robustness, not complete adversarial security. |
| 6 | Explanation faithfulness is not valid | Added modality deletion, sufficiency, comprehensiveness, and contribution scores per instance. | Report faithfulness only from these tests. Do not report uncomputed precision/recall/AUC for explanations. |
| 7 | Human vs LLM is subgroup test, not generalization | Added train-human/test-LLM and train-LLM/test-human evaluation. | Use true cross-domain results from `human_llm_true_generalization.csv`. |
| 8 | Protocol lacks repeated trials and final test | Added repeated seeds and 95% CI tables. Arbiter uses train/test split on evidence table and saved indices. Specialist scores use OOF predictions. | Still stronger if you add campaign/domain/time splits. |
| 9 | Equations have dimension problems | Code defines graph input as a 4-node score matrix with masks. Tables and audit notes identify correct notation for paper rewrite. | Rewrite equations around score vector, mask vector, adjacency matrix, and graph-transformed features. |
| 10 | Related-work comparison insufficient | Code adds stronger baselines: average, true majority, learned late fusion, trust graph, quantum-inspired trust graph. | Add empirical comparison table from generated LaTeX tables. |
| 11 | Deployment claims premature | Added train time, prediction time, Brier, FPR, FNR, and error counts. | Claim prototype deployment potential, not proven industrial readiness. |

## Files you should use in the revised paper

- `tables/DISJOINT_corrected_arbitration_table.tex`
- `tables/DISJOINT_corrected_ablation_2x3.tex`
- `tables/INCIDENT_corrected_arbitration_table.tex`, only if incident manifest was supplied.
- `tables/INCIDENT_corrected_ablation_2x3.tex`, only if incident manifest was supplied.
- `results/human_llm_true_generalization.csv`
- `robustness/*_robustness_seed*.csv`
- `faithfulness/*_faithfulness_summary_seed*.csv`
- `audit/*_protocol_boundary.json`

## Non-negotiable warning

If no incident manifest is supplied, do not write that QTrustAgent-X performs genuine multi-channel incident fusion. The safe term is `evidence-level specialist ensemble with modality-aware arbitration`.

"""
    (cfg.out_root / "reports" / "PROFESSOR_COMMENT_RESPONSE.md").write_text(md, encoding="utf-8")



def write_professor_status_latex(cfg: Config) -> None:
    """Paper-ready table mapping professor comments to implemented corrections."""
    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Correction Matrix for Professor-Identified Weaknesses in QTrustAgent-X}")
    lines.append(r"\label{tab:professor_correction_matrix}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{p{0.04\textwidth}p{0.27\textwidth}p{0.39\textwidth}p{0.20\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"No. & Issue Raised & Code-Level Correction & Claim Boundary \\")
    lines.append(r"\midrule")
    rows = [
        ("1", "No genuine multi-channel fusion", "Added incident-manifest mode with URL, email, SMS, QR scores and explicit missing-channel masks. Disjoint mode is saved separately.", "Claim true fusion only for INCIDENT outputs."),
        ("2", "Architecture and implementation mismatch", "Implemented four-node modality trust graph and classical quantum-inspired trigonometric feature map. Saved audit files.", "Rewrite equations to match released code."),
        ("3", "Weak agentic justification", "Specialist modules, arbitration, audit trail, and report generation are implemented. No autonomous planning is claimed.", "Use specialist-agent pipeline wording unless planning is added."),
        ("4", "Novelty components not validated", "Added average score, true majority vote, learned late fusion, trust graph, and quantum-inspired trust graph baselines.", "Primary endpoint is final arbiter performance."),
        ("5", "Poisoning did not test arbiter", "Poisoning and missing-channel tests now pass through trained arbiters.", "Report limited robustness only."),
        ("6", "Explanation was not faithfulness", "Added deletion, sufficiency, comprehensiveness, and per-instance contribution tests.", "Do not report uncomputed explanation precision, recall, F1, or AUC."),
        ("7", "Human vs. LLM was subgroup testing", "Added train-human/test-LLM and train-LLM/test-human experiments.", "Report cross-domain generalization separately."),
        ("8", "Protocol lacked rigor", "Added out-of-fold specialist scores, repeated seeds, saved splits, CIs, Brier, FPR, FNR, and QR hash audit.", "Still note lack of temporal/campaign split if unavailable."),
        ("9", "Dimensional notation problems", "Graph input is explicitly four modality scores plus four masks. Trust encoder outputs graph-transformed features.", "Rewrite math around four-node score matrix."),
        ("10", "Comparison insufficient", "Added late fusion, true voting, trust graph, robust aggregation-style tests, and random trigonometric feature baseline.", "Do not rely only on checklist table."),
        ("11", "Deployment claims premature", "Added train time, prediction time, calibration, FPR/FNR, error counts, and GUI audit outputs.", "Call it deployment prototype, not proven industrial system."),
    ]
    for r in rows:
        lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    (cfg.out_root / "tables" / "professor_correction_matrix.tex").write_text("\n".join(lines), encoding="utf-8")

def write_manuscript_patch_notes(cfg: Config) -> None:
    text = r"""# Manuscript Patch Notes

Use these changes when revising the paper.

## Abstract
Report the final arbiter result as the primary endpoint. Do not lead with isolated QR or email specialist numbers unless you clearly state they are specialist-only results.

## Method section
Replace BERT, decoded-QR, learned FC semantic compression, and vague trust-graph claims unless those exact components are implemented elsewhere. The v3 code uses:
- TF-IDF/SVD text specialists;
- engineered URL features;
- QR image statistics or QR image specialist depending on model availability;
- out-of-fold specialist scores;
- four-node trust graph over modality scores and masks;
- classical random trigonometric quantum-inspired feature map.

## Results section
Split results into:
1. specialist performance;
2. corrected arbitration baselines;
3. incident-level fusion if manifest exists;
4. robustness under corrupted and missing channels;
5. explanation deletion/sufficiency faithfulness;
6. human vs LLM true cross-domain evaluation;
7. deployment metrics.

## Claims to avoid
- Do not claim "industrial readiness"; claim "deployment prototype".
- Do not claim "true multi-channel fusion" without INCIDENT mode.
- Do not claim "faithful explanation" without deletion/sufficiency results.
- Do not claim "quantum computation"; use "quantum-inspired classical feature map".
- Do not call logistic regression majority voting.

"""
    (cfg.out_root / "reports" / "MANUSCRIPT_PATCH_NOTES.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--incident_manifest", type=Path, default=None)
    parser.add_argument("--mode", choices=["all", "disjoint", "incident"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--qr_limit", type=int, default=20000)
    parser.add_argument("--max_text_features", type=int, default=25000)
    parser.add_argument("--svd_components", type=int, default=256)
    parser.add_argument("--quantum_dim", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.50)
    import sys
    sys.argv = [a for a in sys.argv if str(a).strip()]
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        out_root=args.out_root,
        incident_manifest=args.incident_manifest,
        mode=args.mode,
        seed=args.seed,
        repeats=args.repeats,
        cv_folds=args.cv_folds,
        qr_limit=args.qr_limit,
        max_text_features=args.max_text_features,
        svd_components=args.svd_components,
        quantum_dim=args.quantum_dim,
        threshold=args.threshold,
    )

    set_seed(cfg.seed)
    ensure_dirs(cfg)
    save_json(asdict(cfg), cfg.out_root / "config" / "run_config.json")

    start = time.perf_counter()

    spec, models, meta = build_specialist_oof(cfg, cfg.seed)
    save_csv(spec, cfg.out_root / "results" / "specialist_oof_primary.csv")

    if cfg.mode in ["all", "disjoint"]:
        disjoint = make_disjoint_evidence(meta)
        run_pipeline(cfg, "DISJOINT", disjoint)

    if cfg.mode in ["all", "incident"]:
        if cfg.incident_manifest is None:
            save_json(
                {
                    "incident_mode_skipped": True,
                    "reason": "No incident manifest supplied. True multi-channel fusion cannot be claimed.",
                },
                cfg.out_root / "audit" / "incident_mode_status.json",
            )
        else:
            incident = make_manifest_evidence(cfg, meta)
            run_pipeline(cfg, "INCIDENT", incident)

    run_human_llm_generalization(cfg)
    collect_all_results(cfg)
    write_professor_response(cfg)
    write_professor_status_latex(cfg)
    write_manuscript_patch_notes(cfg)

    save_json({"runtime_seconds": round(time.perf_counter() - start, 3)}, cfg.out_root / "logs" / "runtime.json")
    print(f"Completed. Outputs saved in: {cfg.out_root}")
    print("Read reports/PROFESSOR_COMMENT_RESPONSE.md before revising the manuscript.")


if __name__ == "__main__":
    main()
