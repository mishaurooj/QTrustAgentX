#!/usr/bin/env python
"""
QTrustAgent-X corrective experiments
====================================

This is a SEPARATE corrective-experiment script. It does not overwrite the v8/v9
results. It addresses three reviewer-facing issues:

1) Stronger, leakage-controlled Human<->LLM email transfer.
   - source-domain-only model selection and threshold selection;
   - word, character, and hybrid TF-IDF controls;
   - DeBERTa ensemble with source-domain early stopping;
   - Brier-weighted and stacked fusion;
   - target labels are used ONLY for final evaluation, never for fitting,
     threshold selection, model selection, or fusion weights.

2) Genuine multimodal graph-arbitration validation.
   - REQUIRES an incident-aligned evidence table containing >=2 observed
     modalities per row; unrelated datasets are never synthetically joined;
   - compares masked average, linear late fusion, self-only graph, full TGA,
     QTG-local, and QTG-pair on the same train/test split;
   - writes pair-activation and local-vs-pair feature-effect audits.

3) Reviewer-requested reliability factorial.
   - ONE parameterized circuit implementation;
   - crosses local phase coefficient {0.5, 1.0} with reliability
     {uniform, estimated};
   - pair-strength lambda stays fixed; specialist evidence and classifier stay
     identical; only 20 quantum observables are used in all four cells so that
     post-quantum reliability summaries do not confound the circuit comparison.

The script writes CSV, JSON, 600-dpi figures, and publication-ready LaTeX tables.
It cannot guarantee that a stronger LLM experiment will yield higher target-domain
metrics. Results are saved exactly as obtained.

Typical Windows run
-------------------
python qtrustagentx_corrective_experiments.py ^
  --data_root D:/other/QTrustAgentX/Dataset_Reorganized ^
  --v8_root D:/other/QTrustAgentX/QTrustAgentX_Results_v8 ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Corrective ^
  --seed 42 ^
  --incident_evidence D:/other/QTrustAgentX/QTrustAgentX_Results_v8/predictions/INCIDENT_evidence_table.csv ^
  --run all

For the strongest GPU LLM run:
  --llm_model microsoft/deberta-v3-base --llm_ensemble 3 --llm_epochs 6

Dependencies
------------
pip install numpy pandas scipy scikit-learn matplotlib joblib qiskit pylatexenc
pip install torch transformers sentencepiece tqdm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp
    QISKIT_AVAILABLE = True
    QISKIT_ERROR = ""
except Exception as exc:
    QuantumCircuit = None
    Statevector = None
    SparsePauliOp = None
    QISKIT_AVAILABLE = False
    QISKIT_ERROR = str(exc)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


MODALITIES = ["url", "email", "sms", "qr"]
EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


@dataclass
class Config:
    data_root: Path
    v8_root: Path
    out_root: Path
    seed: int = 42
    threshold: float = 0.50
    run: str = "all"
    incident_evidence: Optional[Path] = None
    test_size: float = 0.20
    bootstrap_samples: int = 2000

    # LLM transfer
    llm_model: str = "microsoft/deberta-v3-base"
    llm_ensemble: int = 3
    llm_epochs: int = 6
    llm_patience: int = 2
    llm_batch_size: int = 8
    llm_grad_accum: int = 4
    llm_max_length: int = 384
    llm_lr: float = 1e-5
    llm_weight_decay: float = 0.01
    llm_cal_fraction: float = 0.20
    llm_num_workers: int = 0
    allow_cpu_llm: bool = False

    # Quantum
    quantum_layers: int = 2
    pair_strength: float = 0.35
    phase_coeffs: Tuple[float, float] = (0.5, 1.0)
    full_qiskit_factorial: bool = False
    qiskit_validation_rows: int = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: Config) -> None:
    for sub in ["results", "tables", "figures", "predictions", "audit", "models", "logs", "cache"]:
        (cfg.out_root / sub).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            pass
    raise RuntimeError(f"Could not read CSV: {path}")


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
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
        "npv": float(tn / max(tn + fn, 1)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = float("nan")
    try:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier"] = float("nan")
    return out


def choose_threshold(y: np.ndarray, score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Source-calibration-only threshold selection by F1, then MCC, then closeness to 0.5."""
    y = np.asarray(y, dtype=int)
    score = np.clip(np.asarray(score, dtype=float), 0, 1)
    # Include score quantiles plus a fixed regular grid. No target labels are used.
    grid = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 181),
        np.quantile(score, np.linspace(0.02, 0.98, 49)),
        np.array([0.5]),
    ]))
    best = None
    for t in grid:
        pred = (score >= t).astype(int)
        f1 = float(f1_score(y, pred, zero_division=0))
        mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else -1.0
        key = (f1, mcc, -abs(float(t) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t), {"cal_f1": f1, "cal_mcc": mcc})
    return best[1], best[2]


def paired_bootstrap_f1(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                        n_boot: int, seed: int) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    obs = float(f1_score(y, pred_a, zero_division=0) - f1_score(y, pred_b, zero_division=0))
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = f1_score(y[idx], pred_a[idx], zero_division=0) - f1_score(y[idx], pred_b[idx], zero_division=0)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    extreme = 0
    for _ in range(n_boot):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, pred_b, pred_a)
        pb = np.where(swap, pred_a, pred_b)
        d = f1_score(y, pa, zero_division=0) - f1_score(y, pb, zero_division=0)
        if abs(d) >= abs(obs) - 1e-15:
            extreme += 1
    return {
        "delta_f1": obs,
        "bootstrap_ci_low": float(lo),
        "bootstrap_ci_high": float(hi),
        "paired_randomization_p": float((extreme + 1) / (n_boot + 1)),
    }


# -----------------------------------------------------------------------------
# Email data and stronger Human<->LLM transfer
# -----------------------------------------------------------------------------

def load_email(cfg: Config) -> pd.DataFrame:
    root = cfg.data_root / "03_email_human_llm"
    files = [
        (root / "human_legit" / "human_legit_email_1000.csv", 0, "human"),
        (root / "human_phishing" / "human_phishing_email_1000.csv", 1, "human"),
        (root / "llm_legit" / "llm_legit_email_1000.csv", 0, "llm"),
        (root / "llm_phishing" / "llm_phishing_email_595.csv", 1, "llm"),
    ]
    rows: List[Dict[str, Any]] = []
    for path, label, domain in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing email source: {path}")
        df = safe_read_csv(path)
        text_col = next((c for c in df.columns if str(c).lower() in ["body", "text", "message", "email", "content"]), df.columns[-1])
        subj_col = next((c for c in df.columns if str(c).lower() == "subject"), None)
        for _, r in df.iterrows():
            body = str(r.get(text_col, ""))
            subject = str(r.get(subj_col, "")) if subj_col else ""
            text = (f"Subject: {subject}\n{body}" if subject else body).strip()
            if text:
                rows.append({"text": text, "label": int(label), "domain": domain, "source_file": path.name})
    data = pd.DataFrame(rows)
    data["norm"] = data["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    # Global exact normalized deduplication prevents the same message appearing in both domains.
    data = data[data["norm"].str.len() > 0].drop_duplicates("norm", keep="first").reset_index(drop=True)
    return data


def make_classical_candidates(seed: int) -> Dict[str, Any]:
    def calibrated(base):
        try:
            return CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
        except TypeError:
            return CalibratedClassifierCV(base_estimator=base, cv=3, method="sigmoid")

    word = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", sublinear_tf=True,
            ngram_range=(1, 2), min_df=2, max_df=0.995, max_features=70000,
        )),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=seed)),
    ])
    char = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", lowercase=True, sublinear_tf=True,
            ngram_range=(3, 5), min_df=2, max_features=90000,
        )),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=seed)),
    ])
    hybrid = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", sublinear_tf=True,
                ngram_range=(1, 2), min_df=2, max_df=0.995, max_features=50000,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", lowercase=True, sublinear_tf=True,
                ngram_range=(3, 5), min_df=2, max_features=70000,
            )),
        ])),
        ("clf", LinearSVC(C=0.75, class_weight="balanced", random_state=seed)),
    ])
    return {
        "word_svm": calibrated(word),
        "char_svm": calibrated(char),
        "hybrid_svm": calibrated(hybrid),
    }


def _torch_device(cfg: Config):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is unavailable: {exc}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not cfg.allow_cpu_llm:
        raise RuntimeError("CUDA is required for the corrective DeBERTa experiment. Use --allow_cpu_llm only intentionally.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return torch, device


def train_deberta_with_validation(cfg: Config, fit_text: Sequence[str], fit_y: Sequence[int],
                                  cal_text: Sequence[str], cal_y: Sequence[int],
                                  target_text: Sequence[str], seed: int, tag: str
                                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train one source-only DeBERTa member and return calibration/target scores.

    The target labels are never passed to this function. Early stopping uses source-domain
    calibration loss only. The returned best state is the model actually used for both
    calibration and target inference, so there is no target-domain model selection.
    """
    try:
        from torch.utils.data import TensorDataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception as exc:
        raise RuntimeError(f"Transformers dependencies unavailable: {exc}")

    torch, device = _torch_device(cfg)
    set_seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, use_fast=True)
    all_text = list(map(str, fit_text)) + list(map(str, cal_text)) + list(map(str, target_text))
    enc = tokenizer(all_text, truncation=True, padding="max_length", max_length=cfg.llm_max_length, return_tensors="pt")
    n_fit = len(fit_text); n_cal = len(cal_text)
    fit_ids = enc["input_ids"][:n_fit]; fit_mask = enc["attention_mask"][:n_fit]
    cal_ids = enc["input_ids"][n_fit:n_fit+n_cal]; cal_mask = enc["attention_mask"][n_fit:n_fit+n_cal]
    tar_ids = enc["input_ids"][n_fit+n_cal:]; tar_mask = enc["attention_mask"][n_fit+n_cal:]

    yfit = torch.as_tensor(np.asarray(fit_y, dtype=int), dtype=torch.long)
    ycal = torch.as_tensor(np.asarray(cal_y, dtype=int), dtype=torch.long)
    fit_ds = TensorDataset(fit_ids, fit_mask, yfit)
    cal_ds = TensorDataset(cal_ids, cal_mask, ycal)
    tar_ds = TensorDataset(tar_ids, tar_mask)
    pin = device.type == "cuda"
    fit_dl = DataLoader(fit_ds, batch_size=cfg.llm_batch_size, shuffle=True, num_workers=cfg.llm_num_workers, pin_memory=pin)
    cal_dl = DataLoader(cal_ds, batch_size=max(cfg.llm_batch_size, 32), shuffle=False, num_workers=cfg.llm_num_workers, pin_memory=pin)
    tar_dl = DataLoader(tar_ds, batch_size=max(cfg.llm_batch_size, 32), shuffle=False, num_workers=cfg.llm_num_workers, pin_memory=pin)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.llm_model, num_labels=2)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device); model.float()

    counts = np.bincount(np.asarray(fit_y, dtype=int), minlength=2)
    class_weights = torch.tensor([len(fit_y)/(2*max(1,c)) for c in counts], dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.llm_lr, weight_decay=cfg.llm_weight_decay)
    accum = max(1, int(cfg.llm_grad_accum))
    updates_per_epoch = max(1, math.ceil(len(fit_dl) / accum))
    total_updates = updates_per_epoch * int(cfg.llm_epochs)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, int(0.1 * total_updates)), num_training_steps=total_updates)

    amp = device.type == "cuda"
    bf16 = bool(amp and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    use_scaler = bool(amp and not bf16)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    def eval_loader(dl, with_labels: bool) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        model.eval(); scores=[]; ys=[]; losses=[]
        with torch.inference_mode():
            for batch in dl:
                if with_labels:
                    ids, mask, labels = batch
                    labels = labels.to(device, non_blocking=True)
                else:
                    ids, mask = batch
                    labels = None
                ids = ids.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                if labels is not None:
                    losses.append(float(loss_fn(logits.float(), labels.long()).item()))
                    ys.extend(labels.cpu().numpy().tolist())
                scores.extend(torch.softmax(logits.float(), dim=1)[:,1].cpu().numpy().tolist())
        return np.asarray(scores, dtype=float), (np.asarray(ys, dtype=int) if with_labels else None), (float(np.mean(losses)) if losses else float("nan"))

    for ep in range(int(cfg.llm_epochs)):
        model.train(); opt.zero_grad(set_to_none=True); train_losses=[]
        for step, (ids, mask, labels) in enumerate(tqdm(fit_dl, desc=f"{tag} ep {ep+1}/{cfg.llm_epochs}")):
            ids = ids.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), labels.long()) / accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if ((step + 1) % accum == 0) or ((step + 1) == len(fit_dl)):
                if use_scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                scheduler.step(); opt.zero_grad(set_to_none=True)
            train_losses.append(float(loss.item()) * accum)

        cal_scores, cal_labels, cal_loss = eval_loader(cal_dl, True)
        cal_pred = (cal_scores >= 0.5).astype(int)
        cal_f1 = float(f1_score(cal_labels, cal_pred, zero_division=0))
        history.append({"epoch": ep+1, "train_loss": float(np.mean(train_losses)), "cal_loss": cal_loss, "cal_f1_at_0.5": cal_f1})
        if cal_loss < best_loss - 1e-5:
            best_loss = cal_loss; best_epoch = ep+1; wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= int(cfg.llm_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    cal_scores, _, final_cal_loss = eval_loader(cal_dl, True)
    target_scores, _, _ = eval_loader(tar_dl, False)

    meta = {
        "seed": seed, "tag": tag, "model": cfg.llm_model, "device": str(device),
        "best_epoch": best_epoch, "best_cal_loss": best_loss, "final_cal_loss": final_cal_loss,
        "epochs_requested": cfg.llm_epochs, "patience": cfg.llm_patience,
        "batch_size": cfg.llm_batch_size, "grad_accum": cfg.llm_grad_accum,
        "max_length": cfg.llm_max_length, "learning_rate": cfg.llm_lr,
        "history": history,
    }
    del model, opt, scheduler, fit_dl, cal_dl, tar_dl, fit_ds, cal_ds, tar_ds, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cal_scores, target_scores, meta


def run_llm_transfer(cfg: Config) -> None:
    print("\n=== Corrective experiment A: stronger Human<->LLM transfer ===")
    data = load_email(cfg)
    all_results: List[Dict[str, Any]] = []
    all_preds: List[pd.DataFrame] = []
    selection_audit: List[Dict[str, Any]] = []

    for didx, (source, target) in enumerate([("human", "llm"), ("llm", "human")]):
        src = data[data["domain"] == source].reset_index(drop=True)
        tar = data[data["domain"] == target].reset_index(drop=True)
        fit_idx, cal_idx = train_test_split(
            np.arange(len(src)), test_size=cfg.llm_cal_fraction,
            stratify=src["label"].to_numpy(), random_state=cfg.seed + 3100 + didx,
        )
        fit = src.iloc[fit_idx].reset_index(drop=True)
        cal = src.iloc[cal_idx].reset_index(drop=True)
        direction = f"{source}_to_{target}"
        print(f"[{direction}] fit={len(fit)}, cal={len(cal)}, target={len(tar)}")

        cal_scores: Dict[str, np.ndarray] = {}
        target_scores: Dict[str, np.ndarray] = {}
        thresholds: Dict[str, float] = {}
        cal_stats: Dict[str, Dict[str, float]] = {}

        # Strong sparse-text candidates.
        classical_models = make_classical_candidates(cfg.seed + didx)
        for name, model in classical_models.items():
            t0 = time.perf_counter(); model.fit(fit["text"], fit["label"]); fit_sec = time.perf_counter() - t0
            cscore = np.asarray(model.predict_proba(cal["text"])[:,1], dtype=float)
            tscore = np.asarray(model.predict_proba(tar["text"])[:,1], dtype=float)
            th, st = choose_threshold(cal["label"].to_numpy(), cscore)
            cal_scores[name] = cscore; target_scores[name] = tscore; thresholds[name] = th; cal_stats[name] = st
            import joblib
            joblib.dump(model, cfg.out_root / "models" / f"{direction}_{name}.joblib")
            save_json({"direction": direction, "method": name, "source_fit_seconds": fit_sec, "threshold": th, **st},
                      cfg.out_root / "audit" / f"{direction}_{name}_source_selection.json")

        # DeBERTa ensemble. All members see the same source-only fit/cal split.
        d_cal_members=[]; d_tar_members=[]; member_meta=[]
        for e in range(max(1, int(cfg.llm_ensemble))):
            member_seed = cfg.seed + 5000 + didx * 100 + e
            cs, ts, meta = train_deberta_with_validation(
                cfg, fit["text"].tolist(), fit["label"].to_numpy(),
                cal["text"].tolist(), cal["label"].to_numpy(),
                tar["text"].tolist(), member_seed, f"{direction}_deberta_{e+1}",
            )
            d_cal_members.append(cs); d_tar_members.append(ts); member_meta.append(meta)
        dcal = np.mean(np.vstack(d_cal_members), axis=0)
        dtar = np.mean(np.vstack(d_tar_members), axis=0)
        dth, dst = choose_threshold(cal["label"].to_numpy(), dcal)
        cal_scores["deberta_ensemble"] = dcal; target_scores["deberta_ensemble"] = dtar
        thresholds["deberta_ensemble"] = dth; cal_stats["deberta_ensemble"] = dst
        save_json({"direction": direction, "members": member_meta, "threshold": dth, **dst},
                  cfg.out_root / "audit" / f"{direction}_deberta_ensemble_source_selection.json")

        # Choose the strongest classical source-calibration model before constructing fusion.
        classical_names = ["word_svm", "char_svm", "hybrid_svm"]
        best_classical = max(classical_names, key=lambda n: (cal_stats[n]["cal_f1"], cal_stats[n]["cal_mcc"]))

        # Brier-weighted fusion estimated only on source calibration.
        bc = float(brier_score_loss(cal["label"], np.clip(cal_scores[best_classical], 0, 1)))
        bd = float(brier_score_loss(cal["label"], np.clip(dcal, 0, 1)))
        wc = 1.0 / max(bc, 1e-6); wd = 1.0 / max(bd, 1e-6)
        ac = float(wc/(wc+wd)); ad = float(wd/(wc+wd))
        bcal = ac*cal_scores[best_classical] + ad*dcal
        btar = ac*target_scores[best_classical] + ad*dtar
        bth, bst = choose_threshold(cal["label"].to_numpy(), bcal)
        cal_scores["brier_fusion"] = bcal; target_scores["brier_fusion"] = btar
        thresholds["brier_fusion"] = bth; cal_stats["brier_fusion"] = bst

        # Source-calibration stacker. Base models were fit only on fit rows.
        # Selection uses OOF stacker probabilities inside the source calibration
        # partition, avoiding an in-sample stacking advantage.
        Zcal = np.column_stack([cal_scores[best_classical], dcal, np.abs(cal_scores[best_classical]-dcal)])
        Ztar = np.column_stack([target_scores[best_classical], dtar, np.abs(target_scores[best_classical]-dtar)])
        ycal_np = cal["label"].to_numpy(int)
        nfold = min(3, int(np.bincount(ycal_np).min()))
        scal = np.full(len(ycal_np), np.nan, dtype=float)
        skf = StratifiedKFold(n_splits=max(2, nfold), shuffle=True, random_state=cfg.seed + 7300 + didx)
        for str_idx, sva_idx in skf.split(Zcal, ycal_np):
            sm = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed+didx)),
            ])
            sm.fit(Zcal[str_idx], ycal_np[str_idx])
            scal[sva_idx] = sm.predict_proba(Zcal[sva_idx])[:,1]
        if not np.all(np.isfinite(scal)):
            raise RuntimeError("Source-only stacked fusion produced missing OOF calibration predictions")
        sth, sst = choose_threshold(ycal_np, scal)
        stack = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed+didx)),
        ])
        stack.fit(Zcal, ycal_np)
        star = stack.predict_proba(Ztar)[:,1]
        cal_scores["stacked_fusion"] = scal; target_scores["stacked_fusion"] = star
        thresholds["stacked_fusion"] = sth; cal_stats["stacked_fusion"] = sst
        import joblib
        joblib.dump(stack, cfg.out_root / "models" / f"{direction}_source_stacker.joblib")

        # Honest selection: source-calibration performance only, never target performance.
        candidate_order = list(cal_scores.keys())
        selected = max(candidate_order, key=lambda n: (cal_stats[n]["cal_f1"], cal_stats[n]["cal_mcc"], -abs(thresholds[n]-0.5)))
        selection_audit.append({
            "direction": direction, "selected_method": selected, "selected_threshold": thresholds[selected],
            "source_cal_f1": cal_stats[selected]["cal_f1"], "source_cal_mcc": cal_stats[selected]["cal_mcc"],
            "target_labels_used_for_selection": False,
            "best_classical_for_fusion": best_classical,
            "brier_fusion_classical_weight": ac, "brier_fusion_deberta_weight": ad,
        })

        for name in candidate_order:
            score = target_scores[name]
            th = thresholds[name]
            pred = (score >= th).astype(int)
            met = binary_metrics(tar["label"].to_numpy(), pred, score)
            met.update({
                "experiment": f"corrective_{direction}_{name}", "direction": direction,
                "method": name, "selected_by_source_calibration": bool(name == selected),
                "threshold_source_only": th, "source_cal_f1": cal_stats[name]["cal_f1"],
                "source_cal_mcc": cal_stats[name]["cal_mcc"],
                "n_source_fit": int(len(fit)), "n_source_cal": int(len(cal)), "n_target": int(len(tar)),
                "target_labels_used_for_model_selection": False,
            })
            if name == "brier_fusion":
                met["fusion_classical_weight"] = ac; met["fusion_deberta_weight"] = ad
            else:
                met["fusion_classical_weight"] = float("nan"); met["fusion_deberta_weight"] = float("nan")
            all_results.append(met)
            all_preds.append(pd.DataFrame({
                "direction": direction, "method": name, "selected_by_source_calibration": name == selected,
                "threshold": th, "y_true": tar["label"].to_numpy(), "y_score": score, "y_pred": pred,
            }))

    results = pd.DataFrame(all_results)
    save_csv(results, cfg.out_root / "results" / "human_llm_corrective_transfer.csv")
    save_csv(pd.concat(all_preds, ignore_index=True), cfg.out_root / "predictions" / "human_llm_corrective_transfer_predictions.csv")
    save_csv(pd.DataFrame(selection_audit), cfg.out_root / "audit" / "human_llm_corrective_source_selection.csv")
    write_llm_latex(results, pd.DataFrame(selection_audit), cfg.out_root / "tables" / "human_llm_corrective_transfer.tex")
    draw_llm_figure(results, cfg.out_root / "figures" / "human_llm_corrective_transfer.png")


def write_llm_latex(results: pd.DataFrame, selection: pd.DataFrame, path: Path) -> None:
    # Paper table shows the source-selected method per direction plus core candidate methods.
    order = ["word_svm", "char_svm", "hybrid_svm", "deberta_ensemble", "brier_fusion", "stacked_fusion"]
    d = results[results["method"].isin(order)].copy()
    lines = [
        r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
        r"\caption{Corrective Human$\leftrightarrow$LLM authoring-domain transfer. Model and threshold selection use only the source-domain calibration partition; target-domain labels are used only for final evaluation.}",
        r"\label{tab:llm_corrective_transfer}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}", r"\toprule",
        r"Train $\rightarrow$ Test & Method & Precision & Specificity & F1 & MCC & Brier & FPR & FNR \\", r"\midrule",
    ]
    for direction in ["human_to_llm", "llm_to_human"]:
        sub = d[d["direction"] == direction]
        for _, r in sub.iterrows():
            meth = str(r["method"]).replace("_", " ")
            if bool(r.get("selected_by_source_calibration", False)):
                meth = r"\textbf{" + meth + "}"
            arrow = "Human $\\rightarrow$ LLM" if direction == "human_to_llm" else "LLM $\\rightarrow$ Human"
            lines.append(
                f"{arrow} & {meth} & {r['precision']:.4f} & {r['specificity']:.4f} & {r['f1']:.4f} & {r['mcc']:.4f} & {r['brier']:.4f} & {r['fpr']:.4f} & {r['fnr']:.4f} \\\\"
            )
        if direction == "human_to_llm":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{0.4ex}",
              r"\begin{minipage}{0.98\textwidth}\footnotesize",
              r"\textit{Selection rule:} bold entries are selected using source-domain calibration F1/MCC only; no target-domain label influences model choice, threshold choice, or fusion weights.",
              r"\end{minipage}", r"\end{table*}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_llm_figure(results: pd.DataFrame, out: Path) -> None:
    selected = results[results["selected_by_source_calibration"] == True].copy()
    if selected.empty:
        return
    x = np.arange(len(selected)); w = 0.32
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x-w/2, selected["precision"].to_numpy(float), width=w, label="Precision")
    ax.bar(x+w/2, selected["specificity"].to_numpy(float), width=w, label="Specificity")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_to_", " → ").title() for d in selected["direction"]])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Source-selected authoring-domain transfer")
    ax.grid(axis="y", alpha=0.2, linestyle="--"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)


# -----------------------------------------------------------------------------
# Classical graph encoder
# -----------------------------------------------------------------------------

class TrustGraphEncoder:
    def __init__(self, sigma: float = 0.25, use_edges: bool = True):
        self.sigma = float(sigma); self.use_edges = bool(use_edges)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        scores = X[:, :4]; masks = X[:, 4:8]
        out=[]
        for s, m in zip(scores, masks):
            A = np.zeros((4,4), dtype=float)
            for i in range(4):
                for j in range(4):
                    if m[i] <= 0 or m[j] <= 0:
                        continue
                    if i == j:
                        A[i,j] = 1.0
                    elif self.use_edges:
                        A[i,j] = math.exp(-((s[i]-s[j])**2)/(2*self.sigma**2))
            deg = A.sum(axis=1, keepdims=True) + 1e-12
            g = (A/deg) @ np.nan_to_num(s, nan=0.5)
            obs = np.flatnonzero(m > 0.5)
            if len(obs) > 1 and self.use_edges:
                sub = A[np.ix_(obs, obs)]
                off = sub[~np.eye(len(obs), dtype=bool)]
                mean_agreement = float(np.mean(off)) if len(off) else 1.0
            else:
                mean_agreement = 1.0
            diag = np.array([float(A.sum()), mean_agreement, float(m.sum()),
                             float(np.max(s[obs])) if len(obs) else 0.5,
                             float(np.min(s[obs])) if len(obs) else 0.5,
                             float(np.std(s[obs])) if len(obs) else 0.0])
            out.append(np.concatenate([g, s, m, diag]))
        return np.asarray(out, dtype=float)


# -----------------------------------------------------------------------------
# One parameterized quantum circuit for graph + reliability experiments
# -----------------------------------------------------------------------------

class ParameterizedRCQTG:
    """Single implementation used in every corrective quantum cell.

    Parameters changed by the factorial:
      reliability: uniform [1,1,1,1] or fixed training-estimated values
      phase_coeff: scalar multiplying the local RZ reliability term

    Held fixed:
      score-angle mapping, layers, RX re-upload coefficient 0.25,
      pair strength lambda, pair equation, readout, classifier, data split.
    """
    def __init__(self, reliability: Sequence[float], phase_coeff: float,
                 layers: int = 2, pair_strength: float = 0.35, pair_enabled: bool = True):
        self.reliability = np.asarray(reliability, dtype=float)
        self.phase_coeff = float(phase_coeff)
        self.layers = int(layers)
        self.pair_strength = float(pair_strength)
        self.pair_enabled = bool(pair_enabled)
        if self.reliability.shape != (4,):
            raise ValueError("reliability must contain URL, email, SMS, QR values")
        if not QISKIT_AVAILABLE:
            raise RuntimeError(f"Qiskit is required: {QISKIT_ERROR}")

    @staticmethod
    def score_angle(p: float) -> float:
        return float(2.0*np.arcsin(np.sqrt(np.clip(p,0,1))))

    @staticmethod
    def _single(axis: str, i: int, n: int = 4):
        label=["I"]*n; label[n-1-i]=axis
        return SparsePauliOp.from_list([("".join(label),1.0)])

    @staticmethod
    def _pair(axis: str, i: int, j: int, n: int = 4):
        label=["I"]*n; label[n-1-i]=axis; label[n-1-j]=axis
        return SparsePauliOp.from_list([("".join(label),1.0)])

    def build_circuit(self, row: np.ndarray) -> QuantumCircuit:
        scores=np.clip(np.asarray(row[:4],float),0,1)
        masks=np.clip(np.asarray(row[4:8],float),0,1)
        rel=self.reliability
        qc=QuantumCircuit(4)
        for layer in range(max(1,self.layers)):
            layer_scale=1.0+0.10*layer
            for q in range(4):
                if masks[q] > 0.5:
                    theta=self.score_angle(scores[q])*math.sqrt(max(rel[q],1e-8))*layer_scale
                    # Reviewer-facing correction: base phase coefficient is explicit and independent.
                    phase=np.pi*(scores[q]-0.5)*self.phase_coeff*rel[q]
                    qc.ry(float(theta),q); qc.rz(float(phase),q)
            if self.pair_enabled:
                for i,j in EDGE_PAIRS:
                    if masks[i] > 0.5 and masks[j] > 0.5:
                        agreement=1.0-abs(scores[i]-scores[j])
                        signed=2.0*agreement-1.0
                        pair_rel=math.sqrt(rel[i]*rel[j])
                        phi=float(np.pi*self.pair_strength*pair_rel*signed)
                        qc.cx(i,j); qc.rz(phi,j); qc.cx(i,j)
            if layer+1 < max(1,self.layers):
                for q in range(4):
                    if masks[q] > 0.5:
                        qc.rx(float(0.25*self.score_angle(scores[q])*rel[q]),q)
        return qc

    def transform_qiskit(self, X: np.ndarray) -> np.ndarray:
        X=np.asarray(X,float)
        z=[self._single("Z",i) for i in range(4)]
        x=[self._single("X",i) for i in range(4)]
        zz=[self._pair("Z",i,j) for i,j in EDGE_PAIRS]
        xx=[self._pair("X",i,j) for i,j in EDGE_PAIRS]
        out=np.zeros((len(X),20),dtype=np.float32)
        for r,row in enumerate(tqdm(X,desc=f"Qiskit phase={self.phase_coeff:g}, pair={self.pair_enabled}")):
            st=Statevector.from_instruction(self.build_circuit(row))
            out[r,0:4]=[float(np.real(st.expectation_value(op))) for op in z]
            out[r,4:8]=[float(np.real(st.expectation_value(op))) for op in x]
            out[r,8:14]=[float(np.real(st.expectation_value(op))) for op in zz]
            out[r,14:20]=[float(np.real(st.expectation_value(op))) for op in xx]
        return out

    def transform_disjoint_exact(self, X: np.ndarray) -> np.ndarray:
        """Exact fast path for one-active-modality rows. Pair gates are mathematically inactive."""
        X=np.asarray(X,float); rel=self.reliability
        out=np.zeros((len(X),20),dtype=np.float32)
        for r,row in enumerate(X):
            scores=np.clip(row[:4],0,1); masks=np.clip(row[4:8],0,1)
            active=np.flatnonzero(masks>0.5)
            if len(active)!=1:
                raise ValueError("DISJOINT exact path requires exactly one active modality per row")
            q=int(active[0]); p=float(scores[q]); rq=float(rel[q])
            base=self.score_angle(p)
            state=np.array([1+0j,0+0j],dtype=np.complex128)
            for layer in range(max(1,self.layers)):
                theta=base*math.sqrt(max(rq,1e-8))*(1+0.10*layer)
                phase=np.pi*(p-0.5)*self.phase_coeff*rq
                state=_ry(theta)@state; state=_rz(phase)@state
                if layer+1<max(1,self.layers):
                    state=_rx(0.25*base*rq)@state
            a,b=state
            zq=float(abs(a)**2-abs(b)**2)
            xq=float(2*np.real(np.conjugate(a)*b))
            zs=np.ones(4,float); xs=np.zeros(4,float); zs[q]=zq; xs[q]=xq
            out[r,0:4]=zs; out[r,4:8]=xs
            out[r,8:14]=[zs[i]*zs[j] for i,j in EDGE_PAIRS]
            out[r,14:20]=[xs[i]*xs[j] for i,j in EDGE_PAIRS]
        return out


def _rx(theta: float) -> np.ndarray:
    c=math.cos(theta/2); s=math.sin(theta/2)
    return np.array([[c,-1j*s],[-1j*s,c]],complex)

def _ry(theta: float) -> np.ndarray:
    c=math.cos(theta/2); s=math.sin(theta/2)
    return np.array([[c,-s],[s,c]],complex)

def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j*theta),0j],[0j,np.exp(0.5j*theta)]],complex)


def validate_disjoint_fast_path(mapper: ParameterizedRCQTG, X: np.ndarray, cfg: Config, tag: str) -> None:
    n=min(cfg.qiskit_validation_rows,len(X))
    if n<=0: return
    sample=np.asarray(X[:n],float)
    fast=mapper.transform_disjoint_exact(sample)
    q=mapper.transform_qiskit(sample)
    err=float(np.max(np.abs(fast-q)))
    save_json({"tag":tag,"rows":n,"max_abs_error":err,"tolerance":1e-7,"equivalent":bool(err<=1e-7)},
              cfg.out_root/"audit"/f"{tag}_fast_vs_qiskit.json")
    if err>1e-7:
        raise RuntimeError(f"Fast/Qiskit equivalence failed for {tag}: {err}")


def save_quantum_circuit_artifacts(mapper: ParameterizedRCQTG, row: np.ndarray, cfg: Config,
                                   base_name: str, *, empirical: bool, note: str = "") -> Dict[str, Any]:
    """Save one exact Qiskit circuit in paper-ready and audit formats.

    Files written:
      figures/quantum_circuits/<base_name>.png/.jpg/.pdf/.svg
      audit/quantum_circuits/<base_name>.txt/.qasm/.json

    ``empirical=False`` is used only for clearly labelled all-channel design
    templates that expose the pair-capable topology. Such templates are never
    used to compute reported performance.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError(f"Qiskit is required to save circuit artifacts: {QISKIT_ERROR}")
    row=np.asarray(row,dtype=float)
    if row.shape[0] < 8:
        raise ValueError("Circuit row must contain four scores followed by four masks")
    qc=mapper.build_circuit(row)
    fig_dir=cfg.out_root/"figures"/"quantum_circuits"
    audit_dir=cfg.out_root/"audit"/"quantum_circuits"
    fig_dir.mkdir(parents=True,exist_ok=True); audit_dir.mkdir(parents=True,exist_ok=True)

    # Exact text and QASM audit representations.
    try:
        txt=str(qc.draw(output="text",fold=120))
    except Exception:
        txt=str(qc)
    (audit_dir/f"{base_name}.txt").write_text(txt,encoding="utf-8")
    qasm_saved=False; qasm_error=None
    try:
        from qiskit import qasm3
        (audit_dir/f"{base_name}.qasm").write_text(qasm3.dumps(qc),encoding="utf-8")
        qasm_saved=True
    except Exception as exc:
        qasm_error=str(exc)

    # Publication rendering. If the native drawer is unavailable, render the
    # exact Qiskit text circuit instead of silently skipping the figure.
    render_method="qiskit_mpl"; native_error=None
    try:
        fig=qc.draw(output="mpl",fold=120)
        if fig is None or not hasattr(fig,"savefig"):
            raise RuntimeError("Qiskit matplotlib drawer returned no Figure")
    except Exception as exc:
        native_error=str(exc); render_method="matplotlib_text_fallback"
        lines=txt.splitlines() or [txt]
        fig,ax=plt.subplots(figsize=(min(34,max(12,max(len(x) for x in lines)*0.105)),
                                     min(24,max(4,len(lines)*0.34+1.4))))
        ax.axis("off")
        ax.text(0.01,0.99,txt,transform=ax.transAxes,va="top",ha="left",
                family="DejaVu Sans Mono",fontsize=8.2)
        ax.set_title(base_name.replace("_"," "),loc="left",fontsize=11,fontweight="bold")
        fig.tight_layout()

    saved=[]
    for ext in ["png","jpg","pdf","svg"]:
        out=fig_dir/f"{base_name}.{ext}"
        kwargs={"bbox_inches":"tight"}
        if ext=="png": kwargs["dpi"]=600
        elif ext=="jpg": kwargs["dpi"]=450
        fig.savefig(out,**kwargs); saved.append(str(out))
    plt.close(fig)

    masks=np.clip(row[4:8],0,1); scores=np.clip(row[:4],0,1)
    meta={
        "base_name":base_name,
        "empirical_row":bool(empirical),
        "design_template_only":bool(not empirical),
        "note":note,
        "scores":scores.tolist(),
        "masks":masks.tolist(),
        "active_modalities":[MODALITIES[i] for i in np.flatnonzero(masks>0.5)],
        "active_modality_count":int(np.sum(masks>0.5)),
        "reliability":mapper.reliability.tolist(),
        "phase_coefficient":float(mapper.phase_coeff),
        "pair_strength":float(mapper.pair_strength),
        "pair_enabled":bool(mapper.pair_enabled),
        "layers":int(mapper.layers),
        "depth":int(qc.depth()),
        "size":int(qc.size()),
        "operations":{str(k):int(v) for k,v in qc.count_ops().items()},
        "execution":"Qiskit Statevector ideal simulation",
        "qpu_execution":False,
        "render_method":render_method,
        "native_qiskit_mpl_error":native_error,
        "qasm_saved":qasm_saved,
        "qasm_error":qasm_error,
        "figure_files":saved,
    }
    save_json(meta,audit_dir/f"{base_name}.json")
    return meta


def fit_quantum_classifier(Ftr: np.ndarray, ytr: np.ndarray, seed: int):
    model=Pipeline([
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed)),
    ])
    model.fit(Ftr,ytr)
    return model


def evidence_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"score_{m}" for m in MODALITIES]+[f"mask_{m}" for m in MODALITIES]].to_numpy(float)


def load_reliability(cfg: Config, seed: int) -> np.ndarray:
    p=cfg.v8_root/"audit"/f"DISJOINT_RC_QTG_reliability_seed{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing training-derived reliability: {p}")
    d=pd.read_csv(p); vals=[]
    for m in MODALITIES:
        q=d[d["modality"].astype(str).str.lower()==m]
        if q.empty: raise ValueError(f"Reliability for {m} missing in {p}")
        vals.append(float(q.iloc[0]["reliability"]))
    return np.asarray(vals,float)


# -----------------------------------------------------------------------------
# Corrective multimodal graph validation
# -----------------------------------------------------------------------------

def resolve_incident_evidence(cfg: Config) -> Path:
    candidates=[]
    if cfg.incident_evidence is not None:
        candidates.append(cfg.incident_evidence)
    candidates += [
        cfg.v8_root/"predictions"/"INCIDENT_evidence_table.csv",
        cfg.v8_root/"predictions"/f"INCIDENT_nested_evidence_seed{cfg.seed}.csv",
    ]
    for p in candidates:
        if p and p.exists(): return p
    raise FileNotFoundError(
        "No genuine incident-aligned evidence table was found. This corrective experiment will NOT synthetically join unrelated URL/email/SMS/QR rows. "
        "Run the v8 pipeline with a verified incident manifest, then pass --incident_evidence to the resulting INCIDENT_evidence_table.csv."
    )


def run_multimodal_graph_validation(cfg: Config) -> None:
    print("\n=== Corrective experiment B: genuine multimodal graph arbitration ===")
    p=resolve_incident_evidence(cfg)
    df=pd.read_csv(p)
    required={"label",*[f"score_{m}" for m in MODALITIES],*[f"mask_{m}" for m in MODALITIES]}
    miss=required.difference(df.columns)
    if miss: raise ValueError(f"Incident evidence missing columns: {sorted(miss)}")
    if "incident_id" not in df.columns:
        df["incident_id"]=[f"incident_{i}" for i in range(len(df))]
    counts=df[[f"mask_{m}" for m in MODALITIES]].sum(axis=1)
    multi=df[counts>=2].copy().reset_index(drop=True)
    if len(multi)<30 or multi["label"].nunique()<2:
        raise RuntimeError(f"Need >=30 genuine multi-source rows containing both classes; found {len(multi)} rows.")
    X=evidence_matrix(multi); y=multi["label"].to_numpy(int)
    tr,te=train_test_split(np.arange(len(y)),test_size=cfg.test_size,stratify=y,random_state=cfg.seed)
    Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
    pair_rate=float(np.mean(np.sum(Xte[:,4:8]>0.5,axis=1)>=2))
    save_json({
        "evidence_file":str(p),"rows_total":len(df),"genuine_multimodal_rows":len(multi),
        "train_rows":len(tr),"test_rows":len(te),"pair_activation_rate_test":pair_rate,
        "synthetic_cross_dataset_joining":False,"seed":cfg.seed,
    },cfg.out_root/"audit"/"multimodal_graph_protocol.json")

    rows=[]; pred_frames=[]
    def add(name,score,pred):
        m=binary_metrics(yte,pred,score); m.update({"experiment":name,"seed":cfg.seed,"n_train":len(tr),"n_test":len(te),"pair_activation_rate":pair_rate}); rows.append(m)
        pred_frames.append(pd.DataFrame({"experiment":name,"incident_id":multi.iloc[te]["incident_id"].astype(str).to_numpy(),"y_true":yte,"y_score":score,"y_pred":pred}))

    # Fixed masked mean.
    denom=Xte[:,4:8].sum(axis=1).clip(min=1); avg=(Xte[:,:4]*Xte[:,4:8]).sum(axis=1)/denom
    add("MaskedAverage",avg,(avg>=0.5).astype(int))

    late=Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=cfg.seed))])
    late.fit(Xtr,ytr); s=late.predict_proba(Xte)[:,1]; add("LateFusion",s,(s>=0.5).astype(int))

    for use_edges,name in [(False,"TGA_SelfOnly"),(True,"TGA_FullGraph")]:
        enc=TrustGraphEncoder(sigma=0.25,use_edges=use_edges)
        Ftr=enc.transform(Xtr); Fte=enc.transform(Xte)
        model=fit_quantum_classifier(Ftr,ytr,cfg.seed)
        s=model.predict_proba(Fte)[:,1]; add(name,s,(s>=0.5).astype(int))

    # Quantum local/pair on the same genuine multimodal split. Uniform reliability keeps this a pair-structure test.
    rel=np.ones(4,float); phase=0.5
    qlocal=ParameterizedRCQTG(rel,phase,cfg.quantum_layers,cfg.pair_strength,pair_enabled=False)
    qpair=ParameterizedRCQTG(rel,phase,cfg.quantum_layers,cfg.pair_strength,pair_enabled=True)

    # Save the exact quantum circuits used by this corrective experiment.
    # The empirical pair example is a real incident-aligned test row with >=2
    # active modalities. An additional all-channel template is saved only to
    # expose the complete four-qubit pair-capable topology.
    active_test_counts=np.sum(Xte[:,4:8]>0.5,axis=1)
    empirical_idx=int(np.argmax(active_test_counts))
    empirical_row=Xte[empirical_idx]
    circuit_manifest=[]
    circuit_manifest.append(save_quantum_circuit_artifacts(
        qlocal,empirical_row,cfg,
        f"multimodal_QTG_local_empirical_seed{cfg.seed}",empirical=True,
        note="Real incident-aligned multimodal test row used in the graph-validation experiment."))
    circuit_manifest.append(save_quantum_circuit_artifacts(
        qpair,empirical_row,cfg,
        f"multimodal_QTG_pair_empirical_seed{cfg.seed}",empirical=True,
        note="Real incident-aligned multimodal test row; pair gates activate only for observed modality pairs."))
    template_row=np.array([0.82,0.74,0.66,0.91,1,1,1,1],dtype=float)
    circuit_manifest.append(save_quantum_circuit_artifacts(
        qpair,template_row,cfg,
        f"multimodal_QTG_pair_all_channel_design_seed{cfg.seed}",empirical=False,
        note="All-channel design template for topology visualization only; never used for reported metrics."))
    save_json({"seed":cfg.seed,"circuits":circuit_manifest},
              cfg.out_root/"audit"/"quantum_circuits"/f"multimodal_circuit_manifest_seed{cfg.seed}.json")

    Fltr=qlocal.transform_qiskit(Xtr); Flte=qlocal.transform_qiskit(Xte)
    Fptr=qpair.transform_qiskit(Xtr); Fpte=qpair.transform_qiskit(Xte)
    ml=fit_quantum_classifier(Fltr,ytr,cfg.seed); mp=fit_quantum_classifier(Fptr,ytr,cfg.seed)
    sl=ml.predict_proba(Flte)[:,1]; sp=mp.predict_proba(Fpte)[:,1]
    add("QTG_Local",sl,(sl>=0.5).astype(int)); add("QTG_Pair",sp,(sp>=0.5).astype(int))

    feature_delta=np.linalg.norm(Fpte-Flte,axis=1)
    save_json({
        "test_rows":len(te),"mean_l2_local_pair_feature_difference":float(np.mean(feature_delta)),
        "max_l2_local_pair_feature_difference":float(np.max(feature_delta)),
        "nonzero_feature_difference_fraction":float(np.mean(feature_delta>1e-10)),
        "pair_gate_mechanism_active":bool(np.any(feature_delta>1e-10)),
    },cfg.out_root/"audit"/"multimodal_qtg_pair_mechanism.json")

    res=pd.DataFrame(rows); preds=pd.concat(pred_frames,ignore_index=True)
    save_csv(res,cfg.out_root/"results"/"multimodal_graph_validation.csv")
    save_csv(preds,cfg.out_root/"predictions"/"multimodal_graph_validation_predictions.csv")

    # Paired inference on the two key structural comparisons.
    stats=[]
    for a,b in [("TGA_FullGraph","TGA_SelfOnly"),("QTG_Pair","QTG_Local")]:
        pa=preds[preds["experiment"]==a].sort_values("incident_id")
        pb=preds[preds["experiment"]==b].sort_values("incident_id")
        st=paired_bootstrap_f1(pa["y_true"].to_numpy(int),pa["y_pred"].to_numpy(int),pb["y_pred"].to_numpy(int),cfg.bootstrap_samples,cfg.seed+len(a))
        st.update({"proposed":a,"comparator":b,"n_test":len(pa)})
        stats.append(st)
    save_csv(pd.DataFrame(stats),cfg.out_root/"results"/"multimodal_graph_paired_tests.csv")
    write_multimodal_latex(res,pd.DataFrame(stats),cfg.out_root/"tables"/"multimodal_graph_validation.tex")
    draw_multimodal_figure(res,cfg.out_root/"figures"/"multimodal_graph_validation.png")


def write_multimodal_latex(res: pd.DataFrame, stats: pd.DataFrame, path: Path) -> None:
    order=["MaskedAverage","LateFusion","TGA_SelfOnly","TGA_FullGraph","QTG_Local","QTG_Pair"]
    d=res.set_index("experiment").reindex(order).reset_index()
    lines=[r"\begin{table*}[!t]",r"\centering",r"\scriptsize",
           r"\caption{Genuine multi-source graph-arbitration validation on incident-aligned evidence. Every evaluated row contains at least two observed modalities; all methods use the same train/test split.}",
           r"\label{tab:multimodal_graph_validation}",r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{lcccccc}",r"\toprule",
           r"Method & F1 & MCC & ROC-AUC & PR-AUC & Brier & Pair activation \\",r"\midrule"]
    for _,r in d.iterrows():
        lines.append(f"{r['experiment'].replace('_',' ')} & {r['f1']:.4f} & {r['mcc']:.4f} & {r['roc_auc']:.4f} & {r['pr_auc']:.4f} & {r['brier']:.4f} & {r['pair_activation_rate']:.3f} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}}",r"\vspace{0.5ex}",
              r"\begin{minipage}{0.98\textwidth}\footnotesize",
              r"The table is valid for a multimodal-interaction claim only because the input evidence is incident-aligned and every retained row contains at least two observed modalities. No unrelated source rows are synthetically paired.",
              r"\end{minipage}",r"\end{table*}"]
    path.write_text("\n".join(lines),encoding="utf-8")


def draw_multimodal_figure(res: pd.DataFrame, out: Path) -> None:
    d=res.copy(); x=np.arange(len(d)); w=0.25
    fig,ax=plt.subplots(figsize=(10.8,5.2))
    for j,m in enumerate(["f1","roc_auc","pr_auc"]):
        ax.bar(x+(j-1)*w,d[m].to_numpy(float),width=w,label=m.upper().replace("_","-"))
    ax.set_xticks(x); ax.set_xticklabels(d["experiment"].str.replace("_"," "),rotation=20,ha="right")
    ax.set_ylim(0,1.03); ax.set_ylabel("Score"); ax.set_title("Incident-aligned multimodal arbitration")
    ax.grid(axis="y",alpha=0.2,linestyle="--"); ax.legend(frameon=False,ncol=3)
    fig.tight_layout(); fig.savefig(out,dpi=600,bbox_inches="tight"); plt.close(fig)


# -----------------------------------------------------------------------------
# Reviewer-requested reliability x phase factorial
# -----------------------------------------------------------------------------

def load_disjoint_seed(cfg: Config) -> Tuple[pd.DataFrame,pd.DataFrame]:
    p=cfg.v8_root/"predictions"/f"DISJOINT_nested_evidence_seed{cfg.seed}.csv"
    if not p.exists(): raise FileNotFoundError(f"Missing v8 nested evidence: {p}")
    d=pd.read_csv(p); d["outer_role"]=d["outer_role"].astype(str).str.lower()
    tr=d[d["outer_role"]=="train"].reset_index(drop=True); te=d[d["outer_role"]=="test"].reset_index(drop=True)
    if len(tr)==0 or len(te)==0: raise RuntimeError("Nested evidence lacks train/test rows")
    return tr,te


def run_reliability_factorial(cfg: Config) -> None:
    print("\n=== Corrective experiment C: reliability x phase factorial ===")
    tr,te=load_disjoint_seed(cfg)
    Xtr=evidence_matrix(tr); Xte=evidence_matrix(te); ytr=tr["label"].to_numpy(int); yte=te["label"].to_numpy(int)
    estimated=load_reliability(cfg,cfg.seed); uniform=np.ones(4,float)
    active_counts=np.sum(Xte[:,4:8]>0.5,axis=1)
    if not np.all(active_counts==1):
        raise RuntimeError("Reviewer factorial expects the cached strict DISJOINT evidence with one active modality per row")

    rows=[]; preds=[]; factorial_circuit_manifest=[]
    empirical_row=Xte[0].copy()
    design_row=np.array([0.82,0.74,0.66,0.91,1,1,1,1],dtype=float)
    for phase in cfg.phase_coeffs:
        for mode,rel in [("uniform",uniform),("estimated",estimated)]:
            tag=f"phase{phase:g}_{mode}"
            mapper=ParameterizedRCQTG(rel,phase,cfg.quantum_layers,cfg.pair_strength,pair_enabled=True)

            # Save both the actual DISJOINT circuit and a clearly labelled
            # all-channel design template for every factorial cell. The latter
            # is visualization only and is never used for performance.
            phase_tag=(f"{phase:g}").replace("-","m").replace(".","p")
            factorial_circuit_manifest.append(save_quantum_circuit_artifacts(
                mapper,empirical_row,cfg,
                f"factorial_phase{phase_tag}_{mode}_empirical_seed{cfg.seed}",empirical=True,
                note="Actual strict-DISJOINT test-row circuit for this reliability x phase cell; one modality is active."))
            factorial_circuit_manifest.append(save_quantum_circuit_artifacts(
                mapper,design_row,cfg,
                f"factorial_phase{phase_tag}_{mode}_all_channel_design_seed{cfg.seed}",empirical=False,
                note="All-channel topology template for this exact factorial parameterization; never used for reported metrics."))

            if cfg.full_qiskit_factorial:
                Ftr=mapper.transform_qiskit(Xtr); Fte=mapper.transform_qiskit(Xte); engine="qiskit_statevector"
            else:
                validate_disjoint_fast_path(mapper,Xte,cfg,f"factorial_{tag}")
                Ftr=mapper.transform_disjoint_exact(Xtr); Fte=mapper.transform_disjoint_exact(Xte); engine="exact_disjoint_validated_against_qiskit"
            model=fit_quantum_classifier(Ftr,ytr,cfg.seed)
            score=model.predict_proba(Fte)[:,1]; pred=(score>=0.5).astype(int)
            m=binary_metrics(yte,pred,score); m.update({
                "experiment":tag,"seed":cfg.seed,"phase_coefficient":phase,"reliability_mode":mode,
                "pair_strength":cfg.pair_strength,"feature_dim":20,"post_quantum_reliability_features":False,
                "execution":engine,"n_train":len(ytr),"n_test":len(yte),
            }); rows.append(m)
            preds.append(pd.DataFrame({"experiment":tag,"incident_id":te["incident_id"].astype(str).to_numpy(),"y_true":yte,"y_score":score,"y_pred":pred}))

    res=pd.DataFrame(rows); pr=pd.concat(preds,ignore_index=True)
    save_csv(res,cfg.out_root/"results"/f"reliability_phase_factorial_seed{cfg.seed}.csv")
    save_csv(pr,cfg.out_root/"predictions"/f"reliability_phase_factorial_predictions_seed{cfg.seed}.csv")

    # Direct factorial contrasts.
    contrasts=[]
    for phase in cfg.phase_coeffs:
        a=res[(res.phase_coefficient==phase)&(res.reliability_mode=="estimated")].iloc[0]
        b=res[(res.phase_coefficient==phase)&(res.reliability_mode=="uniform")].iloc[0]
        contrasts.append({"contrast":f"reliability_effect_at_phase_{phase:g}","delta_f1":a.f1-b.f1,"delta_mcc":a.mcc-b.mcc,"delta_brier":a.brier-b.brier})
    p0,p1=cfg.phase_coeffs
    for mode in ["uniform","estimated"]:
        a=res[(res.phase_coefficient==p1)&(res.reliability_mode==mode)].iloc[0]
        b=res[(res.phase_coefficient==p0)&(res.reliability_mode==mode)].iloc[0]
        contrasts.append({"contrast":f"phase_effect_{p1:g}_minus_{p0:g}_under_{mode}","delta_f1":a.f1-b.f1,"delta_mcc":a.mcc-b.mcc,"delta_brier":a.brier-b.brier})
    cdf=pd.DataFrame(contrasts)
    save_csv(cdf,cfg.out_root/"results"/f"reliability_phase_factorial_contrasts_seed{cfg.seed}.csv")
    save_json({
        "seed":cfg.seed,"phase_coefficients":list(cfg.phase_coeffs),"reliability_uniform":uniform.tolist(),
        "reliability_estimated":estimated.tolist(),"pair_strength_fixed":cfg.pair_strength,
        "quantum_layers_fixed":cfg.quantum_layers,"feature_dim_fixed":20,
        "post_quantum_features_used":False,"specialist_evidence_identical_across_cells":True,
        "classifier_identical_across_cells":True,
        "quantum_circuit_artifacts_saved":True,
        "circuit_manifest":str(cfg.out_root/"audit"/"quantum_circuits"/f"reliability_factorial_circuit_manifest_seed{cfg.seed}.json"),
    },cfg.out_root/"audit"/f"reliability_phase_factorial_design_seed{cfg.seed}.json")
    save_json({"seed":cfg.seed,"circuits":factorial_circuit_manifest},
              cfg.out_root/"audit"/"quantum_circuits"/f"reliability_factorial_circuit_manifest_seed{cfg.seed}.json")
    write_factorial_latex(res,cdf,cfg.out_root/"tables"/f"reliability_phase_factorial_seed{cfg.seed}.tex")
    draw_factorial_figure(res,cfg.out_root/"figures"/f"reliability_phase_factorial_seed{cfg.seed}.png")


def write_factorial_latex(res: pd.DataFrame, contrasts: pd.DataFrame, path: Path) -> None:
    d=res.sort_values(["phase_coefficient","reliability_mode"])
    lines=[r"\begin{table}[!t]",r"\centering",r"\scriptsize",
           r"\caption{Corrective $2\times2$ reliability-factorial experiment using one parameterized RC-QTG circuit. The local phase coefficient and reliability source are varied while specialist evidence, pair strength, readout, classifier, and data split remain fixed. All cells use only the same 20 quantum observables.}",
           r"\label{tab:reliability_phase_factorial}",r"\begin{tabular}{cccccc}",r"\toprule",
           r"Phase coeff. & Reliability & F1 & MCC & ROC-AUC & Brier \\",r"\midrule"]
    for _,r in d.iterrows():
        lines.append(f"{r['phase_coefficient']:.2f} & {str(r['reliability_mode']).capitalize()} & {r['f1']:.4f} & {r['mcc']:.4f} & {r['roc_auc']:.4f} & {r['brier']:.4f} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\vspace{0.5ex}",
              r"\begin{minipage}{0.96\columnwidth}\footnotesize",
              r"Uniform reliability sets all four modality reliabilities to one. Estimated reliability uses the training-only Brier-skill values saved by the strict nested pipeline. No post-quantum reliability summary is included, preventing a second reliability pathway from confounding this circuit-level comparison.",
              r"\end{minipage}",r"\end{table}"]
    path.write_text("\n".join(lines),encoding="utf-8")


def draw_factorial_figure(res: pd.DataFrame, out: Path) -> None:
    phases=sorted(res["phase_coefficient"].unique()); x=np.arange(len(phases)); w=0.34
    fig,ax=plt.subplots(figsize=(6.8,4.6))
    for j,mode in enumerate(["uniform","estimated"]):
        vals=[float(res[(res.phase_coefficient==p)&(res.reliability_mode==mode)]["f1"].iloc[0]) for p in phases]
        ax.bar(x+(j-0.5)*w,vals,width=w,label=mode.capitalize())
    ax.set_xticks(x); ax.set_xticklabels([f"{p:g}" for p in phases]); ax.set_xlabel("Local phase coefficient"); ax.set_ylabel("F1")
    ax.set_ylim(max(0,float(res.f1.min())-0.03),min(1.0,float(res.f1.max())+0.02)); ax.set_title("Reliability × phase factorial")
    ax.grid(axis="y",alpha=0.2,linestyle="--"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out,dpi=600,bbox_inches="tight"); plt.close(fig)


def write_master_tex(cfg: Config) -> None:
    parts=[]
    for p in [
        cfg.out_root/"tables"/"human_llm_corrective_transfer.tex",
        cfg.out_root/"tables"/"multimodal_graph_validation.tex",
        cfg.out_root/"tables"/f"reliability_phase_factorial_seed{cfg.seed}.tex",
    ]:
        if p.exists():
            parts.append(f"% ===== {p.name} =====\n"+p.read_text(encoding="utf-8"))
    (cfg.out_root/"tables"/"QTrustAgentX_corrective_tables_all.tex").write_text("\n\n".join(parts),encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(description="Run QTrustAgent-X corrective experiments without overwriting v8/v9 results.")
    ap.add_argument("--data_root",type=Path,required=True)
    ap.add_argument("--v8_root",type=Path,required=True)
    ap.add_argument("--out_root",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--run",choices=["all","llm","multimodal","factorial"],default="all")
    ap.add_argument("--incident_evidence",type=Path,default=None)
    ap.add_argument("--test_size",type=float,default=0.20)
    ap.add_argument("--bootstrap_samples",type=int,default=2000)

    ap.add_argument("--llm_model",type=str,default="microsoft/deberta-v3-base")
    ap.add_argument("--llm_ensemble",type=int,default=3)
    ap.add_argument("--llm_epochs",type=int,default=6)
    ap.add_argument("--llm_patience",type=int,default=2)
    ap.add_argument("--llm_batch_size",type=int,default=8)
    ap.add_argument("--llm_grad_accum",type=int,default=4)
    ap.add_argument("--llm_max_length",type=int,default=384)
    ap.add_argument("--llm_lr",type=float,default=1e-5)
    ap.add_argument("--llm_cal_fraction",type=float,default=0.20)
    ap.add_argument("--allow_cpu_llm",action="store_true")

    ap.add_argument("--quantum_layers",type=int,default=2)
    ap.add_argument("--pair_strength",type=float,default=0.35)
    ap.add_argument("--phase_coeffs",type=float,nargs=2,default=[0.5,1.0])
    ap.add_argument("--full_qiskit_factorial",action="store_true")
    ap.add_argument("--qiskit_validation_rows",type=int,default=16)
    args=ap.parse_args()

    cfg=Config(
        data_root=args.data_root,v8_root=args.v8_root,out_root=args.out_root,seed=args.seed,
        run=args.run,incident_evidence=args.incident_evidence,test_size=args.test_size,
        bootstrap_samples=args.bootstrap_samples,llm_model=args.llm_model,llm_ensemble=args.llm_ensemble,
        llm_epochs=args.llm_epochs,llm_patience=args.llm_patience,llm_batch_size=args.llm_batch_size,
        llm_grad_accum=args.llm_grad_accum,llm_max_length=args.llm_max_length,llm_lr=args.llm_lr,
        llm_cal_fraction=args.llm_cal_fraction,allow_cpu_llm=args.allow_cpu_llm,
        quantum_layers=args.quantum_layers,pair_strength=args.pair_strength,
        phase_coeffs=tuple(args.phase_coeffs),full_qiskit_factorial=args.full_qiskit_factorial,
        qiskit_validation_rows=args.qiskit_validation_rows,
    )
    ensure_dirs(cfg); set_seed(cfg.seed)
    save_json({k:str(v) if isinstance(v,Path) else v for k,v in cfg.__dict__.items()},cfg.out_root/"audit"/"corrective_config.json")

    started=time.perf_counter(); failures=[]
    if cfg.run in ["all","llm"]:
        try: run_llm_transfer(cfg)
        except Exception as exc:
            failures.append({"experiment":"llm","error":str(exc)})
            if cfg.run=="llm": raise
    if cfg.run in ["all","multimodal"]:
        try: run_multimodal_graph_validation(cfg)
        except Exception as exc:
            failures.append({"experiment":"multimodal","error":str(exc)})
            if cfg.run=="multimodal": raise
    if cfg.run in ["all","factorial"]:
        try: run_reliability_factorial(cfg)
        except Exception as exc:
            failures.append({"experiment":"factorial","error":str(exc)})
            if cfg.run=="factorial": raise

    write_master_tex(cfg)
    save_json({
        "runtime_seconds":round(time.perf_counter()-started,3),"failures":failures,
        "scientific_boundaries":{
            "llm":"target labels are evaluation-only and never used for source-domain model/threshold/fusion selection",
            "multimodal":"multimodal claims require verified incident-aligned evidence with at least two observed modalities per row",
            "factorial":"one parameterized circuit; only phase coefficient and reliability source vary; post-quantum reliability features excluded",
        }
    },cfg.out_root/"audit"/"CORRECTIVE_EXPERIMENT_AUDIT.json")
    print(f"\nCompleted corrective experiments. Outputs: {cfg.out_root}")
    if failures:
        print("Some requested branches could not run; see audit/CORRECTIVE_EXPERIMENT_AUDIT.json")


if __name__=="__main__":
    main()
