"""
QTrustAgent-X v9: Incremental RC-QTG Component Ablations
=======================================================

Purpose
-------
Run ONLY the experiments missing from the v8 paper audit, while reusing the
strict nested DISJOINT evidence, results, and predictions already produced by
QTrustAgent-X v8. This script does NOT retrain URL/email/SMS/QR specialists,
does NOT rerun DeBERTa, and does NOT rerun V1-V8.

New experiments
---------------
V0   Source-prior / mask-only logistic control.
V9A  QTG-pair + reliability-scaled quantum circuit only:
     use the RC-QTG reliability-scaled circuit but classify only the 20 quantum
     observables; no appended reliability statistics.
V9B  QTG-pair + post-quantum reliability features only:
     retain the uncalibrated V7 QTG-pair circuit and append the same four
     reliability statistics used by V8.
V9C  QTG-pair + active-score skip only:
     retain the V7 QTG-pair circuit and append h1 only. Under DISJOINT, h1 is
     exactly the active specialist probability.

Existing V7/V8 predictions are loaded from v8 outputs for matched comparison;
they are never refit by this script.

Outputs
-------
- new fitted model bundles (joblib)
- per-seed metrics CSV
- per-row prediction CSV
- paired V8-vs-new tests
- modality-wise metrics and modality-macro F1
- feature variance/rank/redundancy audit
- publication figures
- only NEW quantum-circuit figures for V9A (no re-rendering of V6/V7/V8)
- consolidated v9 summary CSV and LaTeX table

Example
-------
python qtrustagentx_v9_incremental_ablation.py ^
  --v8_root D:/other/QTrustAgentX/QTrustAgentX_Results_v8 ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Results_v9 ^
  --repeats 5 ^
  --seed 42

Requirements
------------
pip install numpy pandas scipy scikit-learn matplotlib joblib qiskit pylatexenc

Scientific boundary
-------------------
The DISJOINT benchmark contains one active modality per row. Pair gates are
therefore inactive on empirical rows. V9B and V9C use the V7 circuit definition
for matched representation control; V9A uses the reliability-scaled V8 circuit
but removes the appended classical reliability statistics. These experiments
separate the coupled V8 design choices without claiming quantum advantage.
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

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp
    QISKIT_AVAILABLE = True
    QISKIT_IMPORT_ERROR = ""
except Exception as exc:
    QuantumCircuit = None
    Statevector = None
    SparsePauliOp = None
    QISKIT_AVAILABLE = False
    QISKIT_IMPORT_ERROR = str(exc)


MODALITIES = ["url", "email", "sms", "qr"]
EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
NEW_METHODS = [
    "V0_SourcePrior_MaskOnly",
    "V9A_RC_QTG_ReliabilityCircuitOnly",
    "V9B_QTG_Pair_PostQuantumFeaturesOnly",
    "V9C_QTG_Pair_ActiveScoreSkipOnly",
]
CACHED_REFERENCE_SUFFIXES = {
    "V4_TGA_CACHED": "V4_TGA",
    "V5_RFF_CACHED": "V5_RFF_Control",
    "V7_QTG_Pair_CACHED": "V7_QTG_Entangled",
    "V8_RC_QTG_CACHED": "V8_Proposed_RC_QTG",
}


@dataclass
class Config:
    v8_root: Path
    out_root: Path
    seed: int = 42
    repeats: int = 5
    threshold: float = 0.50
    quantum_layers: int = 2
    quantum_pair_strength: float = 0.35
    reliability_power: float = 1.0
    bootstrap_samples: int = 2000
    resume: bool = True
    save_feature_matrices: bool = True
    save_quantum_circuit_figures: bool = True
    full_qiskit_features: bool = False
    qiskit_validation_rows: int = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: Config) -> None:
    for sub in [
        "models", "results", "predictions", "tables", "figures", "audit",
        "audit/quantum_circuits", "cache/v9_quantum_features", "logs", "config",
    ]:
        (cfg.out_root / sub).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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


def ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, float("nan"), float("nan")
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    try:
        from scipy.stats import t as student_t
        critical = float(student_t.ppf(0.975, df=len(arr) - 1))
    except Exception:
        critical = 2.776 if len(arr) == 5 else 1.96
    return mean, mean - critical * se, mean + critical * se


def arbitration_features(evidence: pd.DataFrame) -> np.ndarray:
    cols = [f"score_{m}" for m in MODALITIES] + [f"mask_{m}" for m in MODALITIES]
    return evidence[cols].to_numpy(dtype=float)


def load_nested_evidence(cfg: Config, seed: int) -> pd.DataFrame:
    candidates = [
        cfg.v8_root / "predictions" / f"DISJOINT_nested_evidence_seed{seed}.csv",
        cfg.v8_root / "predictions" / f"DISJOINT_evidence_table_seed{seed}.csv",
        cfg.v8_root / "predictions" / "DISJOINT_evidence_table.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            required = {
                "incident_id", "label", "outer_role",
                *[f"score_{m}" for m in MODALITIES],
                *[f"mask_{m}" for m in MODALITIES],
            }
            missing = required.difference(df.columns)
            if missing:
                continue
            return df
    raise FileNotFoundError(
        f"Could not find strict nested evidence for seed {seed}. Expected "
        f"{cfg.v8_root / 'predictions' / f'DISJOINT_nested_evidence_seed{seed}.csv'}"
    )


def load_reliability(cfg: Config, seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    p = cfg.v8_root / "audit" / f"DISJOINT_RC_QTG_reliability_seed{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing cached v8 reliability file: {p}")
    df = pd.read_csv(p)
    if "modality" not in df.columns or "reliability" not in df.columns:
        raise ValueError(f"Unexpected reliability schema: {p}")
    rel = []
    for mod in MODALITIES:
        row = df[df["modality"].astype(str).str.lower() == mod]
        if row.empty:
            raise ValueError(f"Reliability for modality {mod} missing in {p}")
        rel.append(float(row.iloc[0]["reliability"]))
    return np.asarray(rel, dtype=float), df


def load_cached_predictions(cfg: Config, seed: int) -> pd.DataFrame:
    p = cfg.v8_root / "predictions" / f"DISJOINT_arbitration_predictions_seed{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing v8 prediction file: {p}")
    return pd.read_csv(p)


def load_cached_metrics(cfg: Config, seed: int) -> pd.DataFrame:
    p = cfg.v8_root / "results" / f"DISJOINT_arbitration_seed{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing v8 metrics file: {p}")
    return pd.read_csv(p)


class QTGFeatureMap:
    """Exact four-qubit Qiskit feature map used by the v8 V7 control."""

    def __init__(self, layers: int = 2, pair_strength: float = 0.35, variant: str = "entangled"):
        self.layers = int(layers)
        self.pair_strength = float(pair_strength)
        self.variant = str(variant)

    @staticmethod
    def _score_angle(p: float) -> float:
        return float(2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0))))

    @staticmethod
    def _pauli_single(axis: str, i: int, n: int = 4):
        label = ["I"] * n
        label[n - 1 - i] = axis
        return SparsePauliOp.from_list([("".join(label), 1.0)])

    @staticmethod
    def _pauli_pair(axis: str, i: int, j: int, n: int = 4):
        label = ["I"] * n
        label[n - 1 - i] = axis
        label[n - 1 - j] = axis
        return SparsePauliOp.from_list([("".join(label), 1.0)])

    def _build_circuit(self, row: np.ndarray) -> QuantumCircuit:
        scores = np.clip(np.asarray(row[:4], dtype=float), 0.0, 1.0)
        masks = np.clip(np.asarray(row[4:8], dtype=float), 0.0, 1.0)
        qc = QuantumCircuit(4)
        entangled = self.variant not in {"none", "no_entangle", "local"}
        for layer in range(max(1, self.layers)):
            layer_scale = 1.0 + 0.10 * layer
            for q in range(4):
                if masks[q] > 0.5:
                    theta = self._score_angle(scores[q]) * layer_scale
                    qc.ry(float(theta), q)
                    qc.rz(float(np.pi * (scores[q] - 0.5) * 0.5), q)
            if entangled:
                for i, j in EDGE_PAIRS:
                    if masks[i] > 0.5 and masks[j] > 0.5:
                        agreement = 1.0 - abs(scores[i] - scores[j])
                        phi = float(np.pi * self.pair_strength * (2.0 * agreement - 1.0))
                        qc.cx(i, j)
                        qc.rz(phi, j)
                        qc.cx(i, j)
            if layer + 1 < max(1, self.layers):
                for q in range(4):
                    if masks[q] > 0.5:
                        qc.rx(float(0.25 * self._score_angle(scores[q])), q)
        return qc

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not QISKIT_AVAILABLE:
            raise RuntimeError(f"Qiskit is unavailable: {QISKIT_IMPORT_ERROR}")
        X = np.asarray(X, dtype=float)
        z_ops = [self._pauli_single("Z", i) for i in range(4)]
        x_ops = [self._pauli_single("X", i) for i in range(4)]
        zz_ops = [self._pauli_pair("Z", i, j) for i, j in EDGE_PAIRS]
        xx_ops = [self._pauli_pair("X", i, j) for i, j in EDGE_PAIRS]
        out = np.zeros((len(X), 20), dtype=np.float32)
        for r, row in enumerate(X):
            state = Statevector.from_instruction(self._build_circuit(row))
            out[r, 0:4] = [float(np.real(state.expectation_value(op))) for op in z_ops]
            out[r, 4:8] = [float(np.real(state.expectation_value(op))) for op in x_ops]
            out[r, 8:14] = [float(np.real(state.expectation_value(op))) for op in zz_ops]
            out[r, 14:20] = [float(np.real(state.expectation_value(op))) for op in xx_ops]
        return out


class ReliabilityScaledQTGFeatureMap(QTGFeatureMap):
    """V8 reliability-scaled Qiskit circuit, returning only 20 quantum observables."""

    def __init__(self, reliability: Sequence[float], layers: int = 2, pair_strength: float = 0.35):
        super().__init__(layers=layers, pair_strength=pair_strength, variant="entangled")
        self.reliability = np.asarray(reliability, dtype=float)
        if self.reliability.shape != (4,):
            raise ValueError("Reliability vector must have four values: URL, email, SMS, QR.")

    def _build_circuit(self, row: np.ndarray) -> QuantumCircuit:
        scores = np.clip(np.asarray(row[:4], dtype=float), 0.0, 1.0)
        masks = np.clip(np.asarray(row[4:8], dtype=float), 0.0, 1.0)
        rel = np.asarray(self.reliability, dtype=float)
        qc = QuantumCircuit(4)
        for layer in range(max(1, self.layers)):
            layer_scale = 1.0 + 0.10 * layer
            for q in range(4):
                if masks[q] > 0.5:
                    theta = self._score_angle(scores[q]) * math.sqrt(max(rel[q], 1e-8)) * layer_scale
                    qc.ry(float(theta), q)
                    qc.rz(float(np.pi * (scores[q] - 0.5) * rel[q]), q)
            for i, j in EDGE_PAIRS:
                if masks[i] > 0.5 and masks[j] > 0.5:
                    pair_rel = math.sqrt(rel[i] * rel[j])
                    agreement = 1.0 - abs(scores[i] - scores[j])
                    signed_agreement = 2.0 * agreement - 1.0
                    phi = float(np.pi * self.pair_strength * pair_rel * signed_agreement)
                    qc.cx(i, j)
                    qc.rz(phi, j)
                    qc.cx(i, j)
            if layer + 1 < max(1, self.layers):
                for q in range(4):
                    if masks[q] > 0.5:
                        qc.rx(float(0.25 * self._score_angle(scores[q]) * rel[q]), q)
        return qc




def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0); s = math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0); s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-0.5j * theta), 0.0j],
        [0.0j, np.exp(0.5j * theta)],
    ], dtype=np.complex128)


def disjoint_exact_quantum_features(X: np.ndarray, layers: int, reliability: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact fast-path for the reported one-active-modality DISJOINT protocol.

    No pair gate can activate when exactly one mask is 1. The four-qubit state is
    therefore a tensor product of one evolved single-qubit state and three |0>
    states. This function evaluates the same gate sequence algebraically and
    reconstructs the same 20 Z/X/ZZ/XX observables. It changes no model or data.
    A small Qiskit equivalence audit is written for every seed before these
    features are used.
    """
    X = np.asarray(X, dtype=float)
    rel = None if reliability is None else np.asarray(reliability, dtype=float)
    out = np.zeros((len(X), 20), dtype=np.float32)
    pair_index = {pair: k for k, pair in enumerate(EDGE_PAIRS)}
    for r, row in enumerate(X):
        scores = np.clip(row[:4], 0.0, 1.0)
        masks = np.clip(row[4:8], 0.0, 1.0)
        active = np.flatnonzero(masks > 0.5)
        if len(active) != 1:
            raise ValueError(
                "Fast DISJOINT feature path requires exactly one active modality per row. "
                f"Row {r} has {len(active)} active modalities. Use --full_qiskit_features."
            )
        q = int(active[0])
        p = float(scores[q])
        base_theta = float(2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0))))
        state = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        for layer in range(max(1, int(layers))):
            layer_scale = 1.0 + 0.10 * layer
            if rel is None:
                theta = base_theta * layer_scale
                phase = float(np.pi * (p - 0.5) * 0.5)
                rx_angle = float(0.25 * base_theta)
            else:
                rq = float(rel[q])
                theta = base_theta * math.sqrt(max(rq, 1e-8)) * layer_scale
                phase = float(np.pi * (p - 0.5) * rq)
                rx_angle = float(0.25 * base_theta * rq)
            state = _ry(theta) @ state
            state = _rz(phase) @ state
            if layer + 1 < max(1, int(layers)):
                state = _rx(rx_angle) @ state
        a, b = state
        z = float((abs(a) ** 2 - abs(b) ** 2).real)
        x = float((2.0 * np.real(np.conjugate(a) * b)).real)
        z_single = np.ones(4, dtype=float)
        x_single = np.zeros(4, dtype=float)
        z_single[q] = z
        x_single[q] = x
        out[r, 0:4] = z_single
        out[r, 4:8] = x_single
        zz = []
        xx = []
        for i, j in EDGE_PAIRS:
            zz.append(float(z_single[i] * z_single[j]))
            xx.append(float(x_single[i] * x_single[j]))
        out[r, 8:14] = zz
        out[r, 14:20] = xx
    return out


def validate_fast_quantum_equivalence(X: np.ndarray, q7_mapper: QTGFeatureMap,
                                      q8_mapper: ReliabilityScaledQTGFeatureMap,
                                      cfg: Config, seed: int) -> None:
    n = min(int(cfg.qiskit_validation_rows), len(X))
    if n <= 0:
        return
    sample = np.asarray(X[:n], dtype=float)
    fast7 = disjoint_exact_quantum_features(sample, cfg.quantum_layers, reliability=None)
    fast8 = disjoint_exact_quantum_features(sample, cfg.quantum_layers, reliability=q8_mapper.reliability)
    q7 = q7_mapper.transform(sample)
    q8 = q8_mapper.transform(sample)
    err7 = float(np.max(np.abs(fast7 - q7)))
    err8 = float(np.max(np.abs(fast8 - q8)))
    audit = {
        "seed": seed,
        "validation_rows": n,
        "max_abs_error_fast_vs_qiskit_v7": err7,
        "max_abs_error_fast_vs_qiskit_v9a": err8,
        "tolerance": 1e-7,
        "v7_equivalent": bool(err7 <= 1e-7),
        "v9a_equivalent": bool(err8 <= 1e-7),
        "purpose": "validate exact DISJOINT single-active-qubit fast path against Qiskit Statevector",
    }
    save_json(audit, cfg.out_root / "audit" / f"DISJOINT_v9_fast_quantum_equivalence_seed{seed}.json")
    if err7 > 1e-7 or err8 > 1e-7:
        raise RuntimeError(
            f"Fast quantum feature equivalence check failed for seed {seed}: "
            f"V7 error={err7:.3e}, V9A error={err8:.3e}. Use --full_qiskit_features."
        )

def reliability_summaries(X: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Return h1..h4 exactly as in v8 RC-QTG."""
    X = np.asarray(X, dtype=float)
    rel = np.asarray(reliability, dtype=float)
    scores = np.clip(X[:, :4], 0.0, 1.0)
    masks = np.clip(X[:, 4:8], 0.0, 1.0)
    out = np.zeros((len(X), 4), dtype=np.float32)
    for r, (s, m) in enumerate(zip(scores, masks)):
        active_rel = m * rel
        denom = max(float(np.sum(active_rel)), 1e-8)
        out[r, 0] = float(np.sum(s * active_rel) / denom) if np.sum(m) > 0 else 0.5
        pair_vals = [
            (1.0 - abs(s[i] - s[j])) * math.sqrt(rel[i] * rel[j])
            for i, j in EDGE_PAIRS if m[i] > 0.5 and m[j] > 0.5
        ]
        out[r, 1] = float(np.mean(pair_vals)) if pair_vals else 0.0
        out[r, 2] = float(np.sum(m) / 4.0)
        out[r, 3] = float(np.sum(active_rel) / 4.0)
    return out


def cached_transform(path: Path, transform_fn, X: np.ndarray, resume: bool = True) -> np.ndarray:
    if resume and path.exists():
        arr = np.load(path)
        if len(arr) == len(X):
            print(f"[cache] {path.name}")
            return arr
    t0 = time.perf_counter()
    print(f"[compute] {path.name}: {len(X):,} rows")
    arr = np.asarray(transform_fn(X), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"[done] {path.name}: {time.perf_counter() - t0:.1f}s")
    return arr


def fit_lr(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    model.fit(Xtr, ytr)
    return model


def _paired_randomization_f1(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                              n_iter: int, seed: int) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    obs = float(f1_score(y, pred_a, zero_division=0) - f1_score(y, pred_b, zero_division=0))
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = np.empty(n_iter, dtype=float)
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot[b] = f1_score(y[idx], pred_a[idx], zero_division=0) - f1_score(y[idx], pred_b[idx], zero_division=0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    extreme = 0
    for _ in range(n_iter):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, pred_b, pred_a)
        pb = np.where(swap, pred_a, pred_b)
        d = f1_score(y, pa, zero_division=0) - f1_score(y, pb, zero_division=0)
        if abs(d) >= abs(obs) - 1e-15:
            extreme += 1
    return {
        "f1_difference": obs,
        "bootstrap_ci_low": float(lo),
        "bootstrap_ci_high": float(hi),
        "paired_randomization_p": float((extreme + 1) / (n_iter + 1)),
    }


def feature_diagnostics(name: str, X: np.ndarray) -> Tuple[Dict[str, Any], pd.DataFrame]:
    X = np.asarray(X, dtype=float)
    variance = np.var(X, axis=0)
    nonconstant = variance > 1e-12
    Xc = X - np.mean(X, axis=0, keepdims=True)
    try:
        rank = int(np.linalg.matrix_rank(Xc, tol=1e-10))
    except Exception:
        rank = -1
    dup_pairs = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            if np.allclose(X[:, i], X[:, j], rtol=1e-7, atol=1e-9):
                dup_pairs.append((i, j))
    summary = {
        "representation": name,
        "n_rows": int(X.shape[0]),
        "n_columns": int(X.shape[1]),
        "nonzero_variance_columns": int(np.sum(nonconstant)),
        "constant_columns": int(np.sum(~nonconstant)),
        "centered_numerical_rank": rank,
        "exact_or_near_duplicate_column_pairs": len(dup_pairs),
        "duplicate_pairs": dup_pairs,
    }
    detail = pd.DataFrame({
        "representation": name,
        "feature_index": np.arange(X.shape[1]),
        "variance": variance,
        "is_constant": (~nonconstant).astype(int),
    })
    return summary, detail


def save_new_circuit_figure(mapper: ReliabilityScaledQTGFeatureMap, row: np.ndarray,
                            cfg: Config, seed: int) -> None:
    if not cfg.save_quantum_circuit_figures:
        return
    qc = mapper._build_circuit(np.asarray(row, dtype=float))
    qdir = cfg.out_root / "audit" / "quantum_circuits"
    qdir.mkdir(parents=True, exist_ok=True)
    base = qdir / f"DISJOINT_V9A_reliability_circuit_only_seed{seed}"
    (base.with_suffix(".txt")).write_text(str(qc.draw(output="text", fold=120)), encoding="utf-8")
    try:
        from qiskit import qasm3
        base.with_suffix(".qasm").write_text(qasm3.dumps(qc), encoding="utf-8")
    except Exception as exc:
        save_json({"qasm_error": str(exc)}, base.with_name(base.name + "_qasm_error.json"))
    status: Dict[str, Any] = {
        "seed": seed,
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "num_qubits": int(qc.num_qubits),
        "operations": {str(k): int(v) for k, v in qc.count_ops().items()},
        "empirical_masks": [float(x) for x in row[4:8]],
        "empirical_scores": [float(x) for x in row[:4]],
        "execution": "Qiskit Statevector ideal simulation",
        "new_circuit_only": True,
    }
    try:
        fig = qc.draw(output="mpl", fold=120)
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        status["render_method"] = "qiskit_mpl"
    except Exception as exc:
        status["render_method"] = "text_fallback"
        status["mpl_error"] = str(exc)
        txt = str(qc.draw(output="text", fold=120))
        fig, ax = plt.subplots(figsize=(15, max(4.0, 0.3 * len(txt.splitlines()) + 1.5)))
        ax.axis("off")
        ax.text(0.01, 0.99, txt, va="top", ha="left", family="monospace", fontsize=8)
        fig.tight_layout()
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    save_json(status, base.with_suffix(".json"))


def attach_cached_reference_predictions(cfg: Config, seed: int, evidence_test: pd.DataFrame,
                                        pred_out: List[pd.DataFrame]) -> None:
    cached = load_cached_predictions(cfg, seed)
    for out_name, suffix in CACHED_REFERENCE_SUFFIXES.items():
        sub = cached[cached["experiment"].astype(str).str.endswith(suffix)].copy()
        if sub.empty:
            continue
        keep = ["incident_id", "y_true", "y_score", "y_pred"]
        sub = sub[keep].merge(
            evidence_test[["incident_id", "observed_modality"]],
            on="incident_id", how="left", validate="one_to_one",
        )
        sub.insert(0, "experiment", out_name)
        sub.insert(1, "seed", seed)
        pred_out.append(sub)


def cached_reference_metric_rows(cfg: Config, seed: int) -> List[Dict[str, Any]]:
    cached = load_cached_metrics(cfg, seed)
    rows: List[Dict[str, Any]] = []
    for out_name, suffix in CACHED_REFERENCE_SUFFIXES.items():
        sub = cached[cached["experiment"].astype(str).str.endswith(suffix)]
        if sub.empty:
            continue
        r = sub.iloc[0].to_dict()
        r["experiment"] = out_name
        r["source"] = "cached_v8"
        rows.append(r)
    return rows


def run_seed(cfg: Config, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    result_path = cfg.out_root / "results" / f"DISJOINT_v9_component_ablation_seed{seed}.csv"
    pred_path = cfg.out_root / "predictions" / f"DISJOINT_v9_component_ablation_predictions_seed{seed}.csv"
    mod_path = cfg.out_root / "results" / f"DISJOINT_v9_modality_metrics_seed{seed}.csv"
    paired_path = cfg.out_root / "results" / f"DISJOINT_v9_paired_tests_seed{seed}.csv"
    if cfg.resume and result_path.exists() and pred_path.exists() and mod_path.exists() and paired_path.exists():
        print(f"[seed {seed}] v9 results cached")
        return pd.read_csv(result_path), pd.read_csv(pred_path), pd.read_csv(mod_path), pd.read_csv(paired_path)

    evidence = load_nested_evidence(cfg, seed).copy()
    evidence["outer_role"] = evidence["outer_role"].astype(str).str.lower()
    tr_df = evidence[evidence["outer_role"] == "train"].reset_index(drop=True)
    te_df = evidence[evidence["outer_role"] == "test"].reset_index(drop=True)
    if len(tr_df) == 0 or len(te_df) == 0:
        raise RuntimeError(f"Seed {seed}: nested evidence lacks outer train/test rows.")
    if "observed_modality" not in evidence.columns:
        def infer_mod(row):
            active = [m for m in MODALITIES if float(row[f"mask_{m}"]) > 0.5]
            return active[0] if active else "none"
        tr_df["observed_modality"] = tr_df.apply(infer_mod, axis=1)
        te_df["observed_modality"] = te_df.apply(infer_mod, axis=1)

    Xtr = arbitration_features(tr_df)
    Xte = arbitration_features(te_df)
    ytr = tr_df["label"].to_numpy(dtype=int)
    yte = te_df["label"].to_numpy(dtype=int)
    rel, rel_df = load_reliability(cfg, seed)
    save_csv(rel_df, cfg.out_root / "audit" / f"DISJOINT_v9_reliability_reused_seed{seed}.csv")

    # DISJOINT should have exactly one active source per row.
    active_counts = np.sum(Xte[:, 4:8] > 0.5, axis=1)
    save_json({
        "seed": seed,
        "n_train": int(len(tr_df)),
        "n_test": int(len(te_df)),
        "test_rows_exactly_one_active_modality": int(np.sum(active_counts == 1)),
        "test_rows_with_pair_activation": int(np.sum(active_counts >= 2)),
        "pair_interaction_activation_rate": float(np.mean(active_counts >= 2)),
        "reliability": {m: float(v) for m, v in zip(MODALITIES, rel)},
    }, cfg.out_root / "audit" / f"DISJOINT_v9_protocol_seed{seed}.json")

    # Compute only the two quantum bases needed by the new ablations.
    cache_dir = cfg.out_root / "cache" / "v9_quantum_features"
    q7_mapper = QTGFeatureMap(layers=cfg.quantum_layers, pair_strength=cfg.quantum_pair_strength, variant="entangled")
    q8_mapper = ReliabilityScaledQTGFeatureMap(rel, layers=cfg.quantum_layers, pair_strength=cfg.quantum_pair_strength)

    if cfg.full_qiskit_features:
        q7_fn = q7_mapper.transform
        q8_fn = q8_mapper.transform
        cache_tag = "qiskit"
    else:
        validate_fast_quantum_equivalence(Xte, q7_mapper, q8_mapper, cfg, seed)
        q7_fn = lambda xx: disjoint_exact_quantum_features(xx, cfg.quantum_layers, reliability=None)
        q8_fn = lambda xx: disjoint_exact_quantum_features(xx, cfg.quantum_layers, reliability=rel)
        cache_tag = "exact_disjoint"
    q7_tr = cached_transform(cache_dir / f"seed{seed}_{cache_tag}_q7_train.npy", q7_fn, Xtr, cfg.resume)
    q7_te = cached_transform(cache_dir / f"seed{seed}_{cache_tag}_q7_test.npy", q7_fn, Xte, cfg.resume)
    q8_tr = cached_transform(cache_dir / f"seed{seed}_{cache_tag}_q8_relscaled20_train.npy", q8_fn, Xtr, cfg.resume)
    q8_te = cached_transform(cache_dir / f"seed{seed}_{cache_tag}_q8_relscaled20_test.npy", q8_fn, Xte, cfg.resume)

    htr = reliability_summaries(Xtr, rel)
    hte = reliability_summaries(Xte, rel)

    feature_sets = {
        "V0_SourcePrior_MaskOnly": (Xtr[:, 4:8].astype(np.float32), Xte[:, 4:8].astype(np.float32)),
        "V9A_RC_QTG_ReliabilityCircuitOnly": (q8_tr, q8_te),
        "V9B_QTG_Pair_PostQuantumFeaturesOnly": (np.hstack([q7_tr, htr]), np.hstack([q7_te, hte])),
        "V9C_QTG_Pair_ActiveScoreSkipOnly": (np.hstack([q7_tr, htr[:, [0]]]), np.hstack([q7_te, hte[:, [0]]])),
    }

    rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    feature_audit_summary: List[Dict[str, Any]] = []
    feature_audit_details: List[pd.DataFrame] = []

    for exp, (Ftr, Fte) in feature_sets.items():
        t0 = time.perf_counter()
        model = fit_lr(Ftr, ytr, seed)
        fit_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        score = model.predict_proba(Fte)[:, 1]
        pred_sec = time.perf_counter() - t0
        pred = (score >= cfg.threshold).astype(int)
        met = binary_metrics(yte, pred, score)
        met.update({
            "experiment": exp, "seed": seed,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "train_time_sec": float(fit_sec), "predict_time_sec": float(pred_sec),
            "feature_dim": int(Ftr.shape[1]), "source": "v9_new",
            "pair_interaction_activation_rate": float(np.mean(active_counts >= 2)),
        })
        rows.append(met)
        pred_frames.append(pd.DataFrame({
            "experiment": exp, "seed": seed,
            "incident_id": te_df["incident_id"].astype(str).to_numpy(),
            "observed_modality": te_df["observed_modality"].astype(str).to_numpy(),
            "y_true": yte, "y_score": score, "y_pred": pred,
        }))
        joblib.dump({
            "classifier": model,
            "feature_definition": exp,
            "reliability": rel,
            "quantum_layers": cfg.quantum_layers,
            "pair_strength": cfg.quantum_pair_strength,
        }, cfg.out_root / "models" / f"DISJOINT_{exp}_seed{seed}.joblib")

        diag, detail = feature_diagnostics(exp, Ftr)
        diag["seed"] = seed
        feature_audit_summary.append(diag)
        detail.insert(0, "seed", seed)
        feature_audit_details.append(detail)

    # Add diagnostic for the complete V8 representation without fitting V8.
    full_v8_train = np.hstack([q8_tr, htr])
    diag, detail = feature_diagnostics("V8_RC_QTG_FullRepresentation_NoRefit", full_v8_train)
    diag["seed"] = seed
    feature_audit_summary.append(diag)
    detail.insert(0, "seed", seed)
    feature_audit_details.append(detail)

    save_csv(pd.DataFrame(feature_audit_summary), cfg.out_root / "audit" / f"DISJOINT_v9_feature_rank_summary_seed{seed}.csv")
    save_csv(pd.concat(feature_audit_details, ignore_index=True), cfg.out_root / "audit" / f"DISJOINT_v9_feature_variance_seed{seed}.csv")

    # New circuit figure only for V9A. V9B/V9C reuse the V7 circuit and are not re-rendered.
    empirical_idx = 0
    save_new_circuit_figure(q8_mapper, Xte[empirical_idx], cfg, seed)

    # Include cached V4/V5/V7/V8 metrics/predictions for comparison only.
    rows.extend(cached_reference_metric_rows(cfg, seed))
    attach_cached_reference_predictions(cfg, seed, te_df, pred_frames)

    out = pd.DataFrame(rows)
    pred_all = pd.concat(pred_frames, ignore_index=True)
    save_csv(out, result_path)
    save_csv(pred_all, pred_path)

    # Paired tests: cached V8 vs each new ablation on identical rows.
    paired_rows: List[Dict[str, Any]] = []
    v8 = pred_all[pred_all["experiment"] == "V8_RC_QTG_CACHED"].copy()
    if not v8.empty:
        v8 = v8.sort_values("incident_id").reset_index(drop=True)
        for exp in NEW_METHODS:
            new = pred_all[pred_all["experiment"] == exp].copy().sort_values("incident_id").reset_index(drop=True)
            if len(new) != len(v8) or not np.array_equal(new["incident_id"].to_numpy(), v8["incident_id"].to_numpy()):
                continue
            stat = _paired_randomization_f1(
                v8["y_true"].to_numpy(int),
                v8["y_pred"].to_numpy(int),
                new["y_pred"].to_numpy(int),
                n_iter=cfg.bootstrap_samples,
                seed=seed + sum(ord(c) for c in exp),
            )
            stat.update({
                "seed": seed,
                "proposed": "V8_RC_QTG_CACHED",
                "baseline_or_ablation": exp,
                "direction": "V8_minus_comparator",
                "n_test": int(len(v8)),
            })
            paired_rows.append(stat)
    paired_df = pd.DataFrame(paired_rows)
    save_csv(paired_df, paired_path)

    # Modality-wise metrics and modality-macro summary.
    mod_rows: List[Dict[str, Any]] = []
    for exp, gexp in pred_all.groupby("experiment"):
        per_f1 = []
        for mod in MODALITIES:
            gm = gexp[gexp["observed_modality"].astype(str).str.lower() == mod]
            if gm.empty:
                continue
            mm = binary_metrics(gm["y_true"].to_numpy(int), gm["y_pred"].to_numpy(int), gm["y_score"].to_numpy(float))
            mm.update({"seed": seed, "experiment": exp, "modality": mod, "n": int(len(gm)), "is_macro_row": 0})
            mod_rows.append(mm)
            per_f1.append(mm["f1"])
        if per_f1:
            mod_rows.append({
                "seed": seed, "experiment": exp, "modality": "MACRO",
                "n": int(len(gexp)), "f1": float(np.mean(per_f1)), "is_macro_row": 1,
            })
    modality_df = pd.DataFrame(mod_rows)
    save_csv(modality_df, mod_path)

    draw_seed_figure(out, cfg.out_root / "figures" / f"DISJOINT_v9_component_ablation_seed{seed}.png", seed)
    return out, pred_all, modality_df, paired_df


def draw_seed_figure(metrics: pd.DataFrame, out_path: Path, seed: int) -> None:
    wanted_order = NEW_METHODS + ["V7_QTG_Pair_CACHED", "V8_RC_QTG_CACHED"]
    d = metrics[metrics["experiment"].isin(wanted_order)].copy()
    if d.empty:
        return
    d["order"] = d["experiment"].map({k: i for i, k in enumerate(wanted_order)})
    d = d.sort_values("order")
    x = np.arange(len(d))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for j, metric in enumerate(["f1", "roc_auc", "pr_auc"]):
        ax.bar(x + (j - 1) * width, d[metric].astype(float), width=width, label=metric.upper().replace("ROC_AUC", "ROC-AUC").replace("PR_AUC", "PR-AUC"))
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(v) for v in d["experiment"]], rotation=20, ha="right")
    ymin = max(0.0, float(np.nanmin(d[["f1", "roc_auc", "pr_auc"]].to_numpy())) - 0.05)
    ax.set_ylim(ymin, 1.01)
    ax.set_ylabel("Score")
    ax.set_title(f"RC-QTG component ablation, seed {seed}")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def short_name(exp: str) -> str:
    mapping = {
        "V0_SourcePrior_MaskOnly": "Mask-only prior",
        "V9A_RC_QTG_ReliabilityCircuitOnly": "Rel.-circuit only",
        "V9B_QTG_Pair_PostQuantumFeaturesOnly": "Post-features only",
        "V9C_QTG_Pair_ActiveScoreSkipOnly": "Active-score skip",
        "V4_TGA_CACHED": "TGA (v8)",
        "V5_RFF_CACHED": "RFF (v8)",
        "V7_QTG_Pair_CACHED": "QTG pair (v8)",
        "V8_RC_QTG_CACHED": "RC-QTG (v8)",
    }
    return mapping.get(str(exp), str(exp).replace("_", " "))


def aggregate(cfg: Config) -> None:
    result_files = sorted((cfg.out_root / "results").glob("DISJOINT_v9_component_ablation_seed*.csv"))
    if not result_files:
        return
    all_metrics = pd.concat([pd.read_csv(p) for p in result_files], ignore_index=True)
    save_csv(all_metrics, cfg.out_root / "tables" / "DISJOINT_v9_component_ablation_all_repeats.csv")

    summary_rows = []
    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier", "mcc", "fpr", "fnr", "train_time_sec", "predict_time_sec"]
    for exp, g in all_metrics.groupby("experiment"):
        row: Dict[str, Any] = {"experiment": exp, "n_runs": int(len(g))}
        for m in metric_cols:
            if m in g.columns:
                mean, lo, hi = ci95(g[m])
                row[m] = mean; row[f"{m}_ci_low"] = lo; row[f"{m}_ci_high"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    save_csv(summary, cfg.out_root / "tables" / "DISJOINT_v9_component_ablation_summary_ci.csv")

    paired_files = sorted((cfg.out_root / "results").glob("DISJOINT_v9_paired_tests_seed*.csv"))
    if paired_files:
        paired = pd.concat([pd.read_csv(p) for p in paired_files], ignore_index=True)
        save_csv(paired, cfg.out_root / "tables" / "DISJOINT_v9_paired_tests_all.csv")
        ps = []
        for comp, g in paired.groupby("baseline_or_ablation"):
            mean, lo, hi = ci95(g["f1_difference"])
            ps.append({
                "comparator": comp,
                "mean_delta_f1_V8_minus_comparator": mean,
                "ci_low": lo,
                "ci_high": hi,
                "positive_runs": int(np.sum(g["f1_difference"] > 0)),
                "negative_runs": int(np.sum(g["f1_difference"] < 0)),
                "ties": int(np.sum(g["f1_difference"] == 0)),
                "n_runs": int(len(g)),
            })
        save_csv(pd.DataFrame(ps), cfg.out_root / "tables" / "DISJOINT_v9_paired_summary.csv")

    mod_files = sorted((cfg.out_root / "results").glob("DISJOINT_v9_modality_metrics_seed*.csv"))
    if mod_files:
        mod_all = pd.concat([pd.read_csv(p) for p in mod_files], ignore_index=True)
        save_csv(mod_all, cfg.out_root / "tables" / "DISJOINT_v9_modality_metrics_all.csv")
        macro = mod_all[mod_all["modality"] == "MACRO"].copy()
        macro_rows = []
        for exp, g in macro.groupby("experiment"):
            mean, lo, hi = ci95(g["f1"])
            macro_rows.append({
                "experiment": exp, "macro_f1": mean,
                "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
                "n_runs": int(len(g)),
            })
        macro_summary = pd.DataFrame(macro_rows)
        save_csv(macro_summary, cfg.out_root / "tables" / "DISJOINT_v9_modality_macro_f1_summary.csv")
    else:
        macro_summary = pd.DataFrame()

    feat_files = sorted((cfg.out_root / "audit").glob("DISJOINT_v9_feature_rank_summary_seed*.csv"))
    if feat_files:
        feat = pd.concat([pd.read_csv(p) for p in feat_files], ignore_index=True)
        save_csv(feat, cfg.out_root / "tables" / "DISJOINT_v9_feature_rank_all_seeds.csv")

    draw_aggregate_figure(summary, cfg.out_root / "figures" / "DISJOINT_v9_component_ablation_summary.png")
    if not macro_summary.empty:
        draw_macro_figure(macro_summary, cfg.out_root / "figures" / "DISJOINT_v9_modality_macro_f1.png")
    write_latex_table(summary, cfg.out_root / "tables" / "DISJOINT_v9_component_ablation_table.tex")


def draw_aggregate_figure(summary: pd.DataFrame, out_path: Path) -> None:
    wanted = NEW_METHODS + ["V4_TGA_CACHED", "V5_RFF_CACHED", "V7_QTG_Pair_CACHED", "V8_RC_QTG_CACHED"]
    d = summary[summary["experiment"].isin(wanted)].copy()
    if d.empty:
        return
    d["order"] = d["experiment"].map({k: i for i, k in enumerate(wanted)})
    d = d.sort_values("order")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(14.2, 6.2))
    for metric, marker in [("f1", "o"), ("roc_auc", "s"), ("pr_auc", "^")]:
        ax.plot(x, d[metric].astype(float), marker=marker, markersize=7.5, linewidth=2.2, label=metric.upper().replace("ROC_AUC", "ROC-AUC").replace("PR_AUC", "PR-AUC"))
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(v) for v in d["experiment"]], rotation=22, ha="right")
    vals = d[["f1", "roc_auc", "pr_auc"]].to_numpy(float)
    ax.set_ylim(max(0.0, float(np.nanmin(vals)) - 0.05), 1.01)
    ax.set_ylabel("Mean score across five outer seeds")
    ax.set_title("RC-QTG component ablation with cached v8 references")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_macro_figure(macro: pd.DataFrame, out_path: Path) -> None:
    d = macro.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    y = d["macro_f1"].to_numpy(float)
    lo = d["macro_f1_ci_low"].to_numpy(float)
    hi = d["macro_f1_ci_high"].to_numpy(float)
    yerr = np.vstack([y - lo, hi - y])
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(v) for v in d["experiment"]], rotation=22, ha="right")
    ax.set_ylim(max(0.0, float(np.nanmin(y)) - 0.08), 1.01)
    ax.set_ylabel("Modality-macro F1")
    ax.set_title("Equal-weight performance across URL, email, SMS, and QR")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    wanted = NEW_METHODS + ["V7_QTG_Pair_CACHED", "V8_RC_QTG_CACHED"]
    d = summary[summary["experiment"].isin(wanted)].copy()
    d["order"] = d["experiment"].map({k: i for i, k in enumerate(wanted)})
    d = d.sort_values("order")
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{RC-QTG component ablation across five strict nested outer repetitions. V7 and V8 are loaded from the completed v8 experiment and are not refit.}",
        r"\label{tab:rcqtg_component_ablation_v9}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & F1 & ROC-AUC & PR-AUC & Brier & MCC \\",
        r"\midrule",
    ]
    for _, r in d.iterrows():
        def f(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return "--"
        lines.append(
            f"{short_name(r['experiment'])} & {f(r.get('f1'))} & {f(r.get('roc_auc'))} & {f(r.get('pr_auc'))} & {f(r.get('brier'))} & {f(r.get('mcc'))} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_report(cfg: Config) -> None:
    summary_path = cfg.out_root / "tables" / "DISJOINT_v9_component_ablation_summary_ci.csv"
    macro_path = cfg.out_root / "tables" / "DISJOINT_v9_modality_macro_f1_summary.csv"
    paired_path = cfg.out_root / "tables" / "DISJOINT_v9_paired_summary.csv"
    report: Dict[str, Any] = {
        "v8_root": str(cfg.v8_root),
        "v9_root": str(cfg.out_root),
        "previous_V1_to_V8_retrained": False,
        "specialists_retrained": False,
        "deberta_retrained": False,
        "new_quantum_feature_maps_executed": [
            "V7 QTG-pair observables reconstructed by an exact DISJOINT fast path and validated against Qiskit; no V7 refit",
            "V8 reliability-scaled circuit definition executed for V9A; exact fast path is validated against Qiskit unless --full_qiskit_features is used",
        ],
        "new_quantum_circuit_figures_rendered": "V9A only",
        "component_ablation_summary": str(summary_path),
        "modality_macro_summary": str(macro_path),
        "paired_summary": str(paired_path),
        "claim_boundary": "DISJOINT has one active modality per row; pair gates remain inactive on empirical rows.",
    }
    save_json(report, cfg.out_root / "audit" / "V9_INCREMENTAL_EXPERIMENT_AUDIT.json")
    (cfg.out_root / "audit" / "V9_INCREMENTAL_EXPERIMENT_AUDIT.txt").write_text(
        "\n".join([
            "QTrustAgent-X v9 incremental experiment audit",
            "",
            "Previous V1-V8 retrained: NO",
            "Specialists retrained: NO",
            "DeBERTa retrained: NO",
            "New V9A quantum circuit figures: YES",
            "V6/V7/V8 circuit figures regenerated: NO",
            "",
            "New experiments:",
            "- V0 mask-only source-prior control",
            "- V9A reliability-scaled quantum circuit only",
            "- V9B V7 pair circuit + four post-quantum reliability features",
            "- V9C V7 pair circuit + active-score skip only",
            "",
            "Also produced modality-macro F1 and feature-rank diagnostics.",
        ]),
        encoding="utf-8",
    )




def preflight_v8_cache(cfg: Config) -> None:
    missing: List[str] = []
    for seed in [cfg.seed + i for i in range(cfg.repeats)]:
        required = [
            cfg.v8_root / "predictions" / f"DISJOINT_nested_evidence_seed{seed}.csv",
            cfg.v8_root / "audit" / f"DISJOINT_RC_QTG_reliability_seed{seed}.csv",
            cfg.v8_root / "predictions" / f"DISJOINT_arbitration_predictions_seed{seed}.csv",
            cfg.v8_root / "results" / f"DISJOINT_arbitration_seed{seed}.csv",
        ]
        for path in required:
            if not path.exists():
                missing.append(str(path))
    report = {
        "v8_root": str(cfg.v8_root),
        "required_seed_range": [cfg.seed, cfg.seed + cfg.repeats - 1],
        "missing_files": missing,
        "ready": not missing,
    }
    save_json(report, cfg.out_root / "audit" / "V9_PREFLIGHT_V8_CACHE.json")
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "V9 needs the saved strict-nested v8 evidence/reliability/prediction files. "
            "The following files are missing:\n  - " + joined +
            "\nDo not rerun V1-V8 unless these cache files were deleted."
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Run only missing QTrustAgent-X v9 component ablations using cached v8 outputs.")
    parser.add_argument("--v8_root", type=Path, required=True, help="Completed QTrustAgentX_Results_v8 directory.")
    parser.add_argument("--out_root", type=Path, required=True, help="New QTrustAgentX_Results_v9 directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--quantum_layers", type=int, default=2)
    parser.add_argument("--quantum_pair_strength", type=float, default=0.35)
    parser.add_argument("--reliability_power", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--no_circuit_figures", action="store_true")
    parser.add_argument("--full_qiskit_features", action="store_true", help="Force Qiskit Statevector evaluation for every row. Default uses an exact DISJOINT single-qubit fast path validated against Qiskit on each seed.")
    parser.add_argument("--qiskit_validation_rows", type=int, default=12)
    args = parser.parse_args()

    cfg = Config(
        v8_root=args.v8_root,
        out_root=args.out_root,
        seed=args.seed,
        repeats=args.repeats,
        threshold=args.threshold,
        quantum_layers=args.quantum_layers,
        quantum_pair_strength=args.quantum_pair_strength,
        reliability_power=args.reliability_power,
        bootstrap_samples=args.bootstrap_samples,
        resume=not args.no_resume,
        save_quantum_circuit_figures=not args.no_circuit_figures,
        full_qiskit_features=args.full_qiskit_features,
        qiskit_validation_rows=args.qiskit_validation_rows,
    )
    ensure_dirs(cfg)
    save_json({k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()}, cfg.out_root / "config" / "v9_run_config.json")

    preflight_v8_cache(cfg)

    if not QISKIT_AVAILABLE:
        raise RuntimeError(f"Qiskit is required for V9A-V9C quantum feature extraction: {QISKIT_IMPORT_ERROR}")

    start = time.perf_counter()
    for seed in [cfg.seed + i for i in range(cfg.repeats)]:
        print(f"\n===== V9 incremental seed {seed} =====")
        run_seed(cfg, seed)
    aggregate(cfg)
    write_final_report(cfg)
    save_json({"runtime_seconds": round(time.perf_counter() - start, 3)}, cfg.out_root / "logs" / "v9_runtime.json")
    print(f"\nCompleted v9 incremental experiments. Outputs: {cfg.out_root}")
    print("No specialist, DeBERTa, or previous V1-V8 refitting was performed.")


if __name__ == "__main__":
    main()
