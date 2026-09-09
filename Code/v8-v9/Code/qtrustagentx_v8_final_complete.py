"""
QTrustAgent-X v8: Publication-Grade Quantum Evidence Arbitration
====================================================

This script is the publication-grade experimental pipeline for QTrustAgent-X.
It fixes the remaining evaluation and implementation issues without inventing results.

What this version fixes
-----------------------
1. Genuine incident-level fusion is supported when an incident manifest is provided.
2. If no incident manifest is provided, the script clearly labels results as disjoint
   evidence-level ensemble results and does not call them true multi-channel fusion.
3. Majority voting is implemented as actual vote aggregation.
4. Trust graph reasoning is implemented as a real four-node modality graph:
   URL, Email, SMS, and QR.
5. Quantum arbitration uses four-qubit Qiskit circuits executed with ideal Statevector simulation.
   The circuit encodes the four modality scores and masks, applies optional
   graph-conditioned entanglement/data re-uploading, and returns measured
   expectation features for a class-balanced logistic-regression arbiter.
6. DISJOINT arbitration uses a strict outer holdout: inner-OOF specialist scores
   train the arbiter, while specialist models fit only on outer-train rows score outer-test rows.
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
python qtrustagentx_v8_publication_quantum.py ^
  --data_root D:/other/QTrustAgentX/Dataset_Reorganized ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Results_v8 ^
  --mode all ^
  --qr_limit 20000 ^
  --repeats 5

python qtrustagentx_v8_publication_quantum.py ^
  --data_root D:/other/QTrustAgentX/Dataset_Reorganized ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Results_v8 ^
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
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

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
from sklearn.kernel_approximation import RBFSampler
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
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp
    QISKIT_AVAILABLE = True
    QISKIT_IMPORT_ERROR = ""
except Exception as e:
    QuantumCircuit = None
    Statevector = None
    SparsePauliOp = None
    QISKIT_AVAILABLE = False
    QISKIT_IMPORT_ERROR = str(e)


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
    trust_sigma: float = 0.25
    threshold: float = 0.50
    quantum_layers: int = 2
    reliability_power: float = 1.0
    quantum_pair_strength: float = 0.35
    llm_oof_folds: int = 5
    llm_max_rows: int = 0
    llm_epochs: int = 3
    # Physical GPU microbatch. Effective batch = llm_batch_size * llm_grad_accum.
    # Publication default: 32 x 4 = 128. This gives enough optimizer updates
    # for the small email corpus while fitting an RTX 4070 Ti SUPER comfortably.
    llm_batch_size: int = 32
    llm_grad_accum: int = 4
    llm_max_length: int = 256
    llm_lr: float = 2e-5
    llm_num_workers: int = 0
    allow_cpu_llm: bool = False
    resume_cache: bool = True
    sklearn_jobs: int = 8
    llm_model: str = "microsoft/deberta-v3-small"
    rff_dim: int = 512
    rff_gamma: float = 0.5
    bootstrap_samples: int = 2000
    domain_transfer_epochs: int = 3
    domain_transfer_val_fraction: float = 0.20
    save_quantum_circuit_figures: bool = True
    strict_nested_disjoint: bool = True
    artifacts_only: bool = False
    # Loss is always evaluated in FP32 to avoid Half/Float dtype mismatches.


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: Config) -> None:
    for sub in [
        "models", "results", "figures", "reports", "predictions", "explanations",
        "splits", "tables", "logs", "config", "robustness", "faithfulness",
        "deployment", "audit", "llm", "cache",
        "audit/quantum_circuits", "figures/quantum_circuits"
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
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
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = float("nan")
        try:
            out["brier"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
        except Exception:
            out["brier"] = float("nan")
    else:
        out.update({"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")})
    return out


def ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Mean and two-sided 95% Student-t CI across independent repeats.

    With only five repeated outer splits, a normal 1.96 multiplier is too
    optimistic. Student-t intervals correctly account for the small number of
    repeats.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, float("nan"), float("nan")
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    try:
        from scipy.stats import t as student_t
        critical = float(student_t.ppf(0.975, df=len(arr) - 1))
    except Exception:
        # Conservative fallback for the default five-run protocol (df=4).
        critical = 2.776 if len(arr) == 5 else 1.96
    return mean, float(mean - critical * se), float(mean + critical * se)


class QiskitQuantumFeatureMap(BaseEstimator, TransformerMixin):
    """Four-qubit Qiskit feature map with matched no-entanglement control.

    Each qubit represents URL, email, SMS, or QR. Available scores are encoded
    with amplitude-compatible RY rotations and a score-dependent phase. Missing
    modalities remain in |0>. In the entangled variants, observed modality pairs
    receive an RZZ-equivalent CX-RZ-CX interaction whose angle encodes agreement.

    The returned feature vector contains Z and X expectations for each qubit and
    ZZ and XX expectations for all six modality pairs (20 quantum observables).
    X/XX observables are included intentionally: a diagonal RZZ phase commutes
    with Z/ZZ measurements, so Z-only readout would make the pair phase largely
    invisible. This design therefore makes the entangling operation empirically
    testable against the otherwise identical ``no_entangle`` circuit.

    Execution uses Qiskit's exact Statevector simulator. This is quantum-circuit
    simulation, not IBM QPU execution and not evidence of quantum advantage.
    """
    EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def __init__(self, variant: str = "entangled", layers: int = 2,
                 seed: int = 42, pair_strength: float = 0.35):
        self.variant = variant
        self.layers = layers
        self.seed = seed
        self.pair_strength = pair_strength

    def fit(self, X: Any, y: Optional[Any] = None) -> "QiskitQuantumFeatureMap":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 8:
            raise ValueError("Qiskit arbitration expects four scores followed by four masks.")
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for quantum variants. Install qiskit and qiskit-aer. "
                + QISKIT_IMPORT_ERROR
            )
        self.n_qubits_ = 4
        self.edge_pairs_ = list(self.EDGE_PAIRS)
        return self

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

    @staticmethod
    def _score_angle(p: float) -> float:
        # Probability-compatible angle: sin^2(theta/2) = p.
        return float(2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0))))

    def _build_circuit(self, row: np.ndarray) -> QuantumCircuit:
        scores = np.clip(np.asarray(row[:4], dtype=float), 0.0, 1.0)
        masks = np.clip(np.asarray(row[4:8], dtype=float), 0.0, 1.0)
        qc = QuantumCircuit(4)
        entangled = self.variant not in {"none", "no_entangle", "local"}
        for layer in range(max(1, int(self.layers))):
            layer_scale = 1.0 + 0.10 * layer
            for q in range(4):
                if masks[q] > 0.5:
                    theta = self._score_angle(scores[q]) * layer_scale
                    qc.ry(theta, q)
                    qc.rz(float(np.pi * (scores[q] - 0.5) * 0.5), q)
            if entangled:
                for i, j in self.edge_pairs_:
                    if masks[i] > 0.5 and masks[j] > 0.5:
                        agreement = 1.0 - abs(scores[i] - scores[j])
                        phi = float(np.pi * self.pair_strength * (2.0 * agreement - 1.0))
                        # RZZ(phi), decomposed for broad Qiskit compatibility.
                        qc.cx(i, j)
                        qc.rz(phi, j)
                        qc.cx(i, j)
            # Data re-uploading after pair phases makes interaction-dependent
            # changes visible in local X/Z statistics in deeper circuits.
            if layer + 1 < max(1, int(self.layers)):
                for q in range(4):
                    if masks[q] > 0.5:
                        qc.rx(float(0.25 * self._score_angle(scores[q])), q)
        return qc

    def transform(self, X: Any) -> np.ndarray:
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is unavailable. Install qiskit and qiskit-aer.")
        X = np.asarray(X, dtype=float)
        z_ops = [self._pauli_single("Z", i) for i in range(4)]
        x_ops = [self._pauli_single("X", i) for i in range(4)]
        zz_ops = [self._pauli_pair("Z", i, j) for i, j in self.EDGE_PAIRS]
        xx_ops = [self._pauli_pair("X", i, j) for i, j in self.EDGE_PAIRS]
        out = np.zeros((len(X), 20), dtype=float)
        for r, row in enumerate(X):
            state = Statevector.from_instruction(self._build_circuit(row))
            out[r, 0:4] = [float(np.real(state.expectation_value(op))) for op in z_ops]
            out[r, 4:8] = [float(np.real(state.expectation_value(op))) for op in x_ops]
            out[r, 8:14] = [float(np.real(state.expectation_value(op))) for op in zz_ops]
            out[r, 14:20] = [float(np.real(state.expectation_value(op))) for op in xx_ops]
        return out


class QiskitQuantumKernelFeatures(BaseEstimator, TransformerMixin):
    """Small landmark quantum-kernel embedding using Qiskit statevectors.

    The kernel uses a fixed set of training landmarks. Each feature is the
    squared state overlap between an observation circuit and one landmark
    circuit. Landmark count is deliberately small to keep simulation practical.
    """
    def __init__(self, n_landmarks: int = 16, seed: int = 42):
        self.n_landmarks = n_landmarks
        self.seed = seed

    def fit(self, X: Any, y: Optional[Any] = None) -> "QiskitQuantumKernelFeatures":
        X = np.asarray(X, dtype=float)
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for the quantum-kernel variant.")
        rng = np.random.default_rng(self.seed)
        n = min(self.n_landmarks, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        self.landmarks_ = X[idx]
        self._map = QiskitQuantumFeatureMap(variant="entangled", layers=1, seed=self.seed).fit(X)
        self.landmark_states_ = [Statevector.from_instruction(self._map._build_circuit(x)) for x in self.landmarks_]
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), len(self.landmark_states_)), dtype=float)
        for r, row in enumerate(X):
            state = Statevector.from_instruction(self._map._build_circuit(row))
            for j, ref in enumerate(self.landmark_states_):
                out[r, j] = float(abs(np.vdot(ref.data, state.data)) ** 2)
        return out


class ReliabilityCalibratedQuantumFusion(BaseEstimator, TransformerMixin):
    """Reliability-calibrated mask-gated quantum evidence fusion (RC-QTG).

    Reliability is learned only from the arbiter training partition. For each
    modality, the reliability score is a Brier skill score relative to the
    constant-prevalence predictor on the same active training rows. The score is
    clipped to [0.05, 1] and therefore cannot explode as inverse-Brier weights do.

    Available modalities are encoded on four qubits. Pair phases are activated
    only when both modalities are observed; their signed angle distinguishes
    agreement from conflict and is scaled by pair reliability. The readout uses
    Z, X, ZZ and XX observables so the pair phase is measurable. Four transparent
    classical reliability summaries are appended to the 20 quantum observables.
    """
    EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def __init__(self, layers=2, pair_strength=0.35, reliability_power=1.0, seed=42):
        self.layers = layers
        self.pair_strength = pair_strength
        self.reliability_power = reliability_power
        self.seed = seed

    def fit(self, X, y=None):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for RC-QTG.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 8:
            raise ValueError("RC-QTG expects four scores followed by four masks.")
        y = np.asarray(y if y is not None else np.zeros(len(X)), dtype=int)
        scores = np.clip(X[:, :4], 0.0, 1.0)
        masks = np.clip(X[:, 4:8], 0.0, 1.0)
        rel = []
        details = []
        for k in range(4):
            active = masks[:, k] > 0.5
            if active.sum() < 4 or len(np.unique(y[active])) < 2:
                skill = 0.50
                model_brier = float("nan")
                baseline_brier = float("nan")
            else:
                yy = y[active]
                pp = scores[active, k]
                model_brier = float(brier_score_loss(yy, pp))
                prevalence = float(np.mean(yy))
                baseline = np.full(len(yy), prevalence, dtype=float)
                baseline_brier = float(brier_score_loss(yy, baseline))
                skill = 1.0 - model_brier / max(baseline_brier, 1e-8)
            reliability = float(np.clip(skill, 0.05, 1.0)) ** float(self.reliability_power)
            rel.append(reliability)
            details.append({
                "modality": MODALITIES[k], "active_train_rows": int(active.sum()),
                "brier": model_brier, "baseline_brier": baseline_brier,
                "brier_skill": float(skill), "reliability": reliability,
            })
        self.reliability_ = np.asarray(rel, dtype=float)
        self.reliability_details_ = details
        self.edge_pairs_ = list(self.EDGE_PAIRS)
        return self

    @staticmethod
    def _pauli_single(axis, i, n=4):
        label = ["I"] * n
        label[n - 1 - i] = axis
        return SparsePauliOp.from_list([("".join(label), 1.0)])

    @staticmethod
    def _pauli_pair(axis, i, j, n=4):
        label = ["I"] * n
        label[n - 1 - i] = axis
        label[n - 1 - j] = axis
        return SparsePauliOp.from_list([("".join(label), 1.0)])

    @staticmethod
    def _score_angle(p):
        return float(2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0))))

    def _build_circuit(self, row):
        scores = np.clip(np.asarray(row[:4], dtype=float), 0.0, 1.0)
        masks = np.clip(np.asarray(row[4:8], dtype=float), 0.0, 1.0)
        rel = np.asarray(self.reliability_, dtype=float)
        qc = QuantumCircuit(4)
        for layer in range(max(1, int(self.layers))):
            layer_scale = 1.0 + 0.10 * layer
            for q in range(4):
                if masks[q] > 0.5:
                    theta = self._score_angle(scores[q]) * math.sqrt(max(rel[q], 1e-8)) * layer_scale
                    qc.ry(float(theta), q)
                    qc.rz(float(np.pi * (scores[q] - 0.5) * rel[q]), q)
            for i, j in self.edge_pairs_:
                if masks[i] > 0.5 and masks[j] > 0.5:
                    pair_rel = math.sqrt(rel[i] * rel[j])
                    agreement = 1.0 - abs(scores[i] - scores[j])
                    signed_agreement = 2.0 * agreement - 1.0
                    phi = float(np.pi * self.pair_strength * pair_rel * signed_agreement)
                    qc.cx(i, j)
                    qc.rz(phi, j)
                    qc.cx(i, j)
            if layer + 1 < max(1, int(self.layers)):
                for q in range(4):
                    if masks[q] > 0.5:
                        qc.rx(float(0.25 * self._score_angle(scores[q]) * rel[q]), q)
        return qc

    def transform(self, X):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is unavailable.")
        X = np.asarray(X, dtype=float)
        z_ops = [self._pauli_single("Z", i) for i in range(4)]
        x_ops = [self._pauli_single("X", i) for i in range(4)]
        zz_ops = [self._pauli_pair("Z", i, j) for i, j in self.EDGE_PAIRS]
        xx_ops = [self._pauli_pair("X", i, j) for i, j in self.EDGE_PAIRS]
        out = np.zeros((len(X), 24), dtype=float)
        for r, row in enumerate(X):
            state = Statevector.from_instruction(self._build_circuit(row))
            out[r, 0:4] = [float(np.real(state.expectation_value(op))) for op in z_ops]
            out[r, 4:8] = [float(np.real(state.expectation_value(op))) for op in x_ops]
            out[r, 8:14] = [float(np.real(state.expectation_value(op))) for op in zz_ops]
            out[r, 14:20] = [float(np.real(state.expectation_value(op))) for op in xx_ops]
            s = np.clip(row[:4], 0, 1)
            m = np.clip(row[4:8], 0, 1)
            active_rel = m * self.reliability_
            denom = max(float(np.sum(active_rel)), 1e-8)
            out[r, 20] = float(np.sum(s * active_rel) / denom) if np.sum(m) > 0 else 0.5
            pair_vals = [
                (1.0 - abs(s[i] - s[j])) * math.sqrt(self.reliability_[i] * self.reliability_[j])
                for i, j in self.edge_pairs_ if m[i] > 0.5 and m[j] > 0.5
            ]
            out[r, 21] = float(np.mean(pair_vals)) if pair_vals else 0.0
            out[r, 22] = float(np.sum(m) / 4.0)
            out[r, 23] = float(np.sum(active_rel) / 4.0)
        return out


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


def build_url_model(kind: str = "extra", quantum: bool = False, seed: int = 42, n_jobs: int = 8) -> Pipeline:
    # Specialist model is implementation-aligned: 89 supplied URL attributes -> standardization -> ExtraTrees-600.
    steps: List[Tuple[str, Any]] = [("scale", StandardScaler())]
    if quantum:
        raise ValueError("Quantum processing belongs to the arbitration stage, not the URL specialist.")
    clf = ExtraTreesClassifier(n_estimators=600, n_jobs=n_jobs, random_state=seed, class_weight="balanced") if kind == "extra" else RandomForestClassifier(n_estimators=500, n_jobs=n_jobs, random_state=seed, class_weight="balanced")
    return Pipeline(steps + [("clf", clf)])


def build_text_model(max_features: int, svd_components: Optional[int] = None, quantum: bool = False, seed: int = 42) -> Pipeline:
    # Specialist model is implementation-aligned: TF-IDF 1-2 grams -> SVD-256 -> standardization -> calibrated linear SVM.
    steps: List[Tuple[str, Any]] = [
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True))
    ]
    if svd_components:
        steps += [("svd", TruncatedSVD(n_components=svd_components, n_iter=3, n_oversamples=8, random_state=seed)), ("scale", StandardScaler())]
    if quantum:
        raise ValueError("Quantum processing belongs to the arbitration stage, not the text specialist.")
    clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3)
    return Pipeline(steps + [("clf", clf)])


def build_qr_model(seed: int = 42, n_jobs: int = 8) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", ExtraTreesClassifier(n_estimators=500, n_jobs=n_jobs, random_state=seed, class_weight="balanced"))
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


def run_dataset_integrity_audits(cfg: Config) -> None:
    """Exact/normalized duplicate audits for the non-QR source tables."""
    audit_rows = []

    # URL exact feature-vector duplicates and conflicting labels.
    try:
        X, y, meta = load_url(cfg)
        h = pd.util.hash_pandas_object(pd.DataFrame(X).reset_index(drop=True), index=False).astype(str)
        tmp = pd.DataFrame({"hash": h, "label": y.reset_index(drop=True)})
        dup = int(tmp.duplicated("hash").sum())
        conflict = int((tmp.groupby("hash")["label"].nunique() > 1).sum())
        audit_rows.append({"source": "url", "rows": len(tmp), "exact_or_feature_duplicates": dup, "conflicting_label_hashes": conflict})
    except Exception as exc:
        audit_rows.append({"source": "url", "error": str(exc)})

    # Text audits normalize only whitespace/case. They do not remove rows; they
    # quantify exact-content leakage risk for reporting and split design.
    for source, loader in [("email", load_email), ("sms", load_sms)]:
        try:
            X, y, meta = loader(cfg)
            norm = X.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
            hh = norm.map(lambda z: hashlib.sha256(z.encode("utf-8", errors="ignore")).hexdigest())
            tmp = pd.DataFrame({"hash": hh, "label": y.reset_index(drop=True)})
            if source == "email" and "domain" in meta.columns:
                tmp["domain"] = meta["domain"].reset_index(drop=True).astype(str)
                cross_domain = int(sum(g["domain"].nunique() > 1 for _, g in tmp.groupby("hash")))
            else:
                cross_domain = 0
            dup = int(tmp.duplicated("hash").sum())
            conflict = int((tmp.groupby("hash")["label"].nunique() > 1).sum())
            audit_rows.append({
                "source": source, "rows": len(tmp), "normalized_exact_duplicates": dup,
                "conflicting_label_hashes": conflict, "cross_domain_duplicate_hashes": cross_domain,
            })
            if dup:
                dups = tmp[tmp.duplicated("hash", keep=False)].sort_values("hash")
                save_csv(dups, cfg.out_root / "audit" / f"{source}_normalized_duplicate_hashes.csv")
        except Exception as exc:
            audit_rows.append({"source": source, "error": str(exc)})

    save_csv(pd.DataFrame(audit_rows), cfg.out_root / "audit" / "source_integrity_audit.csv")


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


def oof_scores(X: Any, y: pd.Series, model: Any, cfg: Config,
               precomputed_features: Optional[np.ndarray] = None,
               seed: int = 42, cache_key: Optional[str] = None) -> Tuple[np.ndarray, Any]:
    """Generate OOF scores with resumable disk caching.

    Specialist OOF fits are CPU-heavy (especially TF-IDF/SVD and ExtraTrees).
    Each fold is cached immediately, so an interrupted run resumes instead of
    recomputing completed folds. The cache is keyed by modality and seed.
    """
    X_use = precomputed_features if precomputed_features is not None else X
    scores = np.full(len(y), np.nan, dtype=float)
    cache_dir = cfg.out_root / "cache" / "specialists"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key or "specialist"
    score_cache = cache_dir / f"{key}_oof_seed{seed}.npy"
    done_cache = cache_dir / f"{key}_oof_done_seed{seed}.npy"

    if cfg.resume_cache and score_cache.exists() and done_cache.exists():
        try:
            cached = np.load(score_cache)
            done = np.load(done_cache).astype(bool)
            if len(cached) == len(y) and len(done) == len(y):
                scores[:] = cached
            else:
                done = np.zeros(len(y), dtype=bool)
        except Exception:
            done = np.zeros(len(y), dtype=bool)
    else:
        done = np.zeros(len(y), dtype=bool)

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        if cfg.resume_cache and np.all(done[va]) and np.all(np.isfinite(scores[va])):
            print(f"[{key}] OOF fold {fold}/{cfg.cv_folds}: cached")
            continue
        print(f"[{key}] OOF fold {fold}/{cfg.cv_folds}: fitting {len(tr):,} train / {len(va):,} valid rows")
        t0 = time.perf_counter()
        m = clone(model)
        Xtr = X_use[tr] if isinstance(X_use, np.ndarray) else X_use.iloc[tr]
        Xva = X_use[va] if isinstance(X_use, np.ndarray) else X_use.iloc[va]
        m.fit(Xtr, y.iloc[tr])
        scores[va] = get_score(m, Xva)
        done[va] = True
        np.save(score_cache, scores)
        np.save(done_cache, done.astype(np.uint8))
        print(f"[{key}] OOF fold {fold} completed in {time.perf_counter()-t0:.1f}s")
        del m

    if not np.all(np.isfinite(scores)):
        missing = int(np.sum(~np.isfinite(scores)))
        raise RuntimeError(f"{key}: {missing} OOF predictions are missing after cross-fitting.")

    final_path = cfg.out_root / "models" / f"{key}_specialist_seed{seed}.joblib"
    if cfg.resume_cache and final_path.exists():
        try:
            final_model = joblib.load(final_path)
            print(f"[{key}] final specialist model: cached")
            return scores, final_model
        except Exception:
            pass

    print(f"[{key}] fitting final specialist model on all {len(y):,} rows")
    final_model = clone(model)
    final_model.fit(X_use, y)
    joblib.dump(final_model, final_path)
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


def crossfit_email_dual_fusion(meta_email: pd.DataFrame, cfg: Config, seed: int) -> pd.DataFrame:
    """Cross-fitted classical/DeBERTa email fusion without second-level leakage.

    For each fold, reliability weights are estimated only from the complementary
    training folds and then applied to the held-out fold. The returned fused score
    is therefore an OOF meta-feature. A separate full-data weight estimate is saved
    only as a deployment reference and is never used to score the OOF rows.
    """
    df = meta_email.copy()
    if "score_email" not in df.columns or "score_email_llm" not in df.columns:
        return df
    y = df["label"].to_numpy(int)
    classical = np.clip(df["score_email"].to_numpy(float), 0.0, 1.0)
    llm = np.clip(df["score_email_llm"].fillna(0.5).to_numpy(float), 0.0, 1.0)
    n_splits = min(int(cfg.cv_folds), int(np.bincount(y).min()))
    if n_splits < 2:
        df["score_email_fused"] = classical
        df["pred_email_fused"] = (classical >= cfg.threshold).astype(int)
        return df
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 7103)
    fused = np.full(len(df), np.nan, dtype=float)
    audit_rows = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        bc = float(brier_score_loss(y[tr], classical[tr]))
        bl = float(brier_score_loss(y[tr], llm[tr]))
        wc = 1.0 / max(bc, 1e-6)
        wl = 1.0 / max(bl, 1e-6)
        alpha_c = wc / (wc + wl)
        alpha_l = wl / (wc + wl)
        fused[va] = alpha_c * classical[va] + alpha_l * llm[va]
        audit_rows.append({
            "fold": fold, "n_train": len(tr), "n_valid": len(va),
            "classical_brier_train": bc, "llm_brier_train": bl,
            "classical_weight": alpha_c, "llm_weight": alpha_l,
        })
    if not np.all(np.isfinite(fused)):
        raise RuntimeError("Cross-fitted email fusion produced missing predictions.")
    df["score_email_fused"] = fused
    df["pred_email_fused"] = (fused >= cfg.threshold).astype(int)
    # Keep the primary arbitration email channel on the classical specialist.
    # The dual score is reported as a secondary email experiment so the main
    # arbiter can be evaluated with a strict nested outer holdout efficiently.

    bc_full = float(brier_score_loss(y, classical))
    bl_full = float(brier_score_loss(y, llm))
    wc = 1.0 / max(bc_full, 1e-6)
    wl = 1.0 / max(bl_full, 1e-6)
    save_csv(pd.DataFrame(audit_rows), cfg.out_root / "llm" / f"email_dual_fusion_crossfit_seed{seed}.csv")
    save_json({
        "protocol": "cross_fitted_meta_fusion",
        "oof_rows_use_only_complementary_fold_weights": True,
        "full_data_weights_are_reference_only": True,
        "classical_brier_full_oof": bc_full,
        "llm_brier_full_oof": bl_full,
        "classical_weight_reference": float(wc / (wc + wl)),
        "llm_weight_reference": float(wl / (wc + wl)),
    }, cfg.out_root / "llm" / f"email_dual_specialist_fusion_seed{seed}.json")
    return df


def build_specialist_oof(cfg: Config, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, pd.DataFrame]]:
    data: Dict[str, Tuple[Any, pd.Series, Any, Optional[np.ndarray], pd.DataFrame]] = {}

    X_url, y_url, meta_url = load_url(cfg)
    data["url"] = (X_url, y_url, build_url_model("extra", quantum=False, seed=seed, n_jobs=cfg.sklearn_jobs), None, meta_url)

    X_email, y_email, meta_email = load_email(cfg)
    data["email"] = (X_email, y_email, build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=seed), None, meta_email)

    X_sms, y_sms, meta_sms = load_sms(cfg)
    data["sms"] = (X_sms, y_sms, build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=seed), None, meta_sms)

    X_qr, y_qr, meta_qr = load_qr(cfg)
    qr_cache = cfg.out_root / "cache_qr_features.npy"
    X_qrf = qr_features(X_qr.tolist(), cache_path=qr_cache)
    data["qr"] = (X_qr, y_qr, build_qr_model(seed=seed, n_jobs=cfg.sklearn_jobs), X_qrf, meta_qr)

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
        scores, fitted = oof_scores(X, y, base_model, cfg, precomputed_features=feats, seed=seed, cache_key=mod)
        models_dict[mod] = fitted
        meta = meta.copy()
        meta[f"score_{mod}"] = scores
        meta[f"pred_{mod}"] = (scores >= cfg.threshold).astype(int)
        meta_dict[mod] = meta
        joblib.dump(fitted, cfg.out_root / "models" / f"{mod}_specialist_seed{seed}.joblib")
        metrics = binary_metrics(y.values, meta[f"pred_{mod}"].values, scores)
        metrics.update({"experiment": f"{mod.upper()}_OOF_Specialist", "seed": seed, "n": len(y)})
        specialist_rows.append(metrics)

    # Add the transformer as a second email specialist only after its own OOF
    # predictions are complete. The final email score remains one modality score.
    try:
        llm_oof = run_llm_email_oof(cfg)
        if llm_oof is not None and "email_row_id" in meta_dict["email"].columns:
            meta_dict["email"] = meta_dict["email"].merge(llm_oof, on="email_row_id", how="left")
            meta_dict["email"]["score_email_llm"] = meta_dict["email"]["score_email_llm"].fillna(0.5)
            # Meta-level fusion is itself cross-fitted. No arbiter row receives a
            # fusion weight estimated using that row's label.
            meta_dict["email"] = crossfit_email_dual_fusion(meta_dict["email"], cfg, seed)
            models_dict["email_llm"] = llm_oof
    except Exception as e:
        save_json({"status":"unavailable", "reason":str(e)}, cfg.out_root / "llm" / f"email_dual_specialist_fusion_seed{seed}.json")

    spec = pd.DataFrame(specialist_rows)
    save_csv(spec, cfg.out_root / "results" / f"specialist_oof_results_seed{seed}.csv")
    return spec, models_dict, meta_dict


def make_nested_disjoint_evidence(cfg: Config, seed: int) -> pd.DataFrame:
    """Strict two-level disjoint benchmark with an untouched outer holdout.

    The outer split is performed on raw rows before specialist fitting. Inner OOF
    scores for outer-train rows are generated using only outer-train data. Each
    final specialist is then refit on all outer-train rows and scores the untouched
    outer-test rows. The arbiter trains on inner-OOF rows and is evaluated only on
    those untouched specialist predictions. This removes the meta-level leakage of
    generating all OOF scores first and splitting the arbiter afterwards.

    DeBERTa is evaluated separately as a text-specialist/domain-transfer experiment;
    the primary nested arbitration uses the classical email specialist so that the
    outer holdout remains computationally tractable and unambiguous.
    """
    rows: List[Dict[str, Any]] = []
    nested_metrics: List[Dict[str, Any]] = []

    def add_rows(mod: str, meta: pd.DataFrame, indices: np.ndarray, scores: np.ndarray, role: str):
        id_col = f"{mod}_row_id" if f"{mod}_row_id" in meta.columns else "row_id"
        for local_pos, source_idx in enumerate(indices):
            r = meta.iloc[int(source_idx)]
            row = {
                "incident_id": f"{mod}_{int(r[id_col])}",
                "label": int(r["label"]),
                "fusion_valid": 0,
                "protocol": "strict_nested_disjoint_evidence",
                "observed_modality": mod,
                "outer_role": role,
                "outer_seed": seed,
                "source_row_index": int(source_idx),
                "score_email_llm": 0.5,
            }
            for m in MODALITIES:
                row[f"score_{m}"] = float(scores[local_pos]) if m == mod else 0.5
                row[f"mask_{m}"] = 1 if m == mod else 0
            rows.append(row)

    # Load once per modality and produce inner OOF + untouched outer-test scores.
    loaders = {
        "url": lambda: (*load_url(cfg), None),
        "email": lambda: (*load_email(cfg), None),
        "sms": lambda: (*load_sms(cfg), None),
        "qr": lambda: (*load_qr(cfg), "qr"),
    }
    for mod in MODALITIES:
        loaded = loaders[mod]()
        X_raw, y, meta, feature_kind = loaded
        y = y.reset_index(drop=True)
        meta = meta.reset_index(drop=True)
        all_idx = np.arange(len(y))
        dev_idx, test_idx = train_test_split(
            all_idx, test_size=cfg.test_size, stratify=y.to_numpy(), random_state=seed + 100 * (MODALITIES.index(mod) + 1)
        )
        dev_idx = np.asarray(dev_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        if mod == "url":
            base_model = build_url_model(seed=seed, n_jobs=cfg.sklearn_jobs)
            X_use = X_raw
            pre_dev = pre_test = None
        elif mod == "email":
            base_model = build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=seed)
            X_use = X_raw
            pre_dev = pre_test = None
        elif mod == "sms":
            base_model = build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=seed)
            X_use = X_raw
            pre_dev = pre_test = None
        else:
            base_model = build_qr_model(seed=seed, n_jobs=cfg.sklearn_jobs)
            qr_cache = cfg.out_root / "cache_qr_features.npy"
            all_features = qr_features(X_raw.tolist(), cache_path=qr_cache)
            X_use = X_raw
            pre_dev = all_features[dev_idx]
            pre_test = all_features[test_idx]

        X_dev = X_use.iloc[dev_idx].reset_index(drop=True) if hasattr(X_use, "iloc") else np.asarray(X_use)[dev_idx]
        y_dev = y.iloc[dev_idx].reset_index(drop=True)
        inner_scores, fitted = oof_scores(
            X_dev, y_dev, base_model, cfg,
            precomputed_features=pre_dev,
            seed=seed,
            cache_key=f"nested_{mod}_outer{seed}",
        )
        if mod == "qr":
            test_scores = get_score(fitted, pre_test)
        else:
            X_test = X_use.iloc[test_idx] if hasattr(X_use, "iloc") else np.asarray(X_use)[test_idx]
            test_scores = get_score(fitted, X_test)

        inner_scores = np.asarray(inner_scores, dtype=float)
        test_scores = np.asarray(test_scores, dtype=float)
        add_rows(mod, meta, dev_idx, inner_scores, "train")
        add_rows(mod, meta, test_idx, test_scores, "test")

        train_metrics = binary_metrics(y_dev.to_numpy(), (inner_scores >= cfg.threshold).astype(int), inner_scores)
        train_metrics.update({"modality": mod, "outer_seed": seed, "split": "outer_train_inner_oof", "n": int(len(dev_idx))})
        test_y = y.iloc[test_idx].to_numpy()
        test_metrics = binary_metrics(test_y, (test_scores >= cfg.threshold).astype(int), test_scores)
        test_metrics.update({"modality": mod, "outer_seed": seed, "split": "outer_test_untouched", "n": int(len(test_idx))})
        nested_metrics.extend([train_metrics, test_metrics])

        audit = {
            "modality": mod, "outer_seed": seed,
            "n_total": int(len(y)), "n_outer_train": int(len(dev_idx)), "n_outer_test": int(len(test_idx)),
            "outer_test_never_used_for_specialist_fit": True,
            "arbiter_train_specialist_scores": "inner OOF on outer-train only",
            "arbiter_test_specialist_scores": "final specialist fit on outer-train only",
        }
        save_json(audit, cfg.out_root / "audit" / f"nested_{mod}_outer_seed{seed}.json")

    evidence = pd.DataFrame(rows)
    save_csv(pd.DataFrame(nested_metrics), cfg.out_root / "results" / f"DISJOINT_nested_specialists_seed{seed}.csv")
    # Keep a deterministic order but do not mix train/test membership.
    evidence = evidence.sort_values(["outer_role", "observed_modality", "incident_id"]).reset_index(drop=True)
    return evidence


def run_nested_disjoint_pipeline(cfg: Config) -> None:
    """Run repeated strict nested outer-holdout experiments for disjoint evidence."""
    seeds = [cfg.seed + i for i in range(cfg.repeats)]
    for seed in seeds:
        set_seed(seed)
        evidence_path = cfg.out_root / "predictions" / f"DISJOINT_nested_evidence_seed{seed}.csv"
        if cfg.resume_cache and evidence_path.exists():
            try:
                evidence = pd.read_csv(evidence_path)
                required = {"outer_role", "label", *[f"score_{m}" for m in MODALITIES], *[f"mask_{m}" for m in MODALITIES]}
                if not required.issubset(evidence.columns):
                    raise ValueError("nested evidence cache schema mismatch")
                print(f"[DISJOINT seed {seed}] strict nested evidence: cached")
            except Exception:
                evidence = make_nested_disjoint_evidence(cfg, seed)
                save_csv(evidence, evidence_path)
        else:
            evidence = make_nested_disjoint_evidence(cfg, seed)
            save_csv(evidence, evidence_path)
        train_arbiters_once(evidence, cfg, seed, "DISJOINT")
    aggregate_repeats(cfg, "DISJOINT")
    nested_files = sorted((cfg.out_root / "results").glob("DISJOINT_nested_specialists_seed*.csv"))
    if nested_files:
        ns = pd.concat([pd.read_csv(x) for x in nested_files], ignore_index=True)
        save_csv(ns, cfg.out_root / "tables" / "DISJOINT_nested_specialists_all.csv")
        summary_rows = []
        for (modality, split), g in ns.groupby(["modality", "split"]):
            row = {"modality": modality, "split": split, "n_runs": int(len(g))}
            for metric in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier", "mcc"]:
                mean, lo, hi = ci95(g[metric])
                row[metric] = mean; row[f"{metric}_ci_low"] = lo; row[f"{metric}_ci_high"] = hi
            summary_rows.append(row)
        save_csv(pd.DataFrame(summary_rows), cfg.out_root / "tables" / "DISJOINT_nested_specialists_summary_ci.csv")
    save_json({
        "protocol": "strict_nested_outer_holdout",
        "repeats": cfg.repeats,
        "specialist_outer_test_used_for_training": False,
        "arbiter_outer_test_used_for_training": False,
        "claim": "evidence-level arbitration only; no incident fusion claim",
    }, cfg.out_root / "audit" / "DISJOINT_strict_nested_protocol.json")


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
            row["score_email_llm"] = float(r.get("score_email_llm", 0.5)) if mod == "email" else 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def make_manifest_evidence(cfg: Config, meta_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build incident evidence with strict reference and label auditing.

    Non-empty manifest references must resolve to a source row. Source labels must
    agree with the incident label. Reusing the same source row in multiple incident
    records is rejected because it can place identical evidence in both train and
    test incident splits. Blank references remain valid missing modalities.
    """
    if cfg.incident_manifest is None:
        raise RuntimeError("incident_manifest is required for incident-level fusion.")
    manifest = safe_read_csv(cfg.incident_manifest)
    if "incident_id" not in manifest.columns:
        manifest["incident_id"] = np.arange(len(manifest)).astype(str)
    if "label" not in manifest.columns:
        raise ValueError("incident_manifest must contain a label column.")
    if manifest["incident_id"].astype(str).duplicated().any():
        dup = int(manifest["incident_id"].astype(str).duplicated().sum())
        raise ValueError(f"incident_manifest contains {dup} duplicate incident_id values.")

    y = label_to_binary(manifest["label"])
    lookups: Dict[str, pd.DataFrame] = {}
    id_columns: Dict[str, str] = {}
    for mod, meta in meta_dict.items():
        id_col = f"{mod}_row_id" if f"{mod}_row_id" in meta.columns else "row_id"
        id_columns[mod] = id_col
        lookups[mod] = meta.set_index(id_col, drop=False)

    # QR lookups tolerate path formatting differences, but ambiguous basenames
    # are not accepted.
    qr_meta = meta_dict["qr"].copy()
    qr_meta["_norm_path"] = qr_meta["qr_path"].astype(str).map(lambda x: os.path.normcase(os.path.normpath(x)))
    qr_exact = {k: i for i, k in zip(qr_meta.index, qr_meta["_norm_path"])}
    basename_groups: Dict[str, List[int]] = {}
    for i, pth in zip(qr_meta.index, qr_meta["qr_path"].astype(str)):
        basename_groups.setdefault(os.path.normcase(Path(pth).name), []).append(i)

    invalid_refs: List[Dict[str, Any]] = []
    label_mismatches: List[Dict[str, Any]] = []
    used_refs: Dict[str, List[str]] = {m: [] for m in MODALITIES}
    rows: List[Dict[str, Any]] = []

    for i, r in manifest.iterrows():
        inc_id = str(r["incident_id"])
        inc_label = int(y.iloc[i])
        row: Dict[str, Any] = {
            "incident_id": inc_id, "label": inc_label,
            "fusion_valid": 0, "protocol": "incident_level_fusion",
            "score_email_llm": 0.5,
        }
        observed = 0
        for mod in MODALITIES:
            score = 0.5; mask = 0; source_label = None; source_ref = None
            if mod == "qr":
                raw = r.get("qr_path") if "qr_path" in manifest.columns else None
                if pd.notna(raw) and str(raw).strip():
                    raw_s = str(raw).strip()
                    norm = os.path.normcase(os.path.normpath(raw_s))
                    match_idx = None
                    exact_matches = qr_meta.index[qr_meta["_norm_path"] == norm].tolist()
                    if len(exact_matches) == 1:
                        match_idx = exact_matches[0]
                    else:
                        b = os.path.normcase(Path(raw_s).name)
                        candidates = basename_groups.get(b, [])
                        if len(candidates) == 1:
                            match_idx = candidates[0]
                    if match_idx is None:
                        invalid_refs.append({"incident_id": inc_id, "modality": mod, "reference": raw_s})
                    else:
                        rr = qr_meta.loc[match_idx]
                        score = float(rr["score_qr"]); mask = 1
                        source_label = int(rr["label"]); source_ref = os.path.normcase(os.path.normpath(str(rr["qr_path"])))
            else:
                col = f"{mod}_row_id"
                raw = r.get(col) if col in manifest.columns else None
                if pd.notna(raw) and str(raw).strip():
                    try:
                        rid = int(float(raw))
                    except Exception:
                        invalid_refs.append({"incident_id": inc_id, "modality": mod, "reference": str(raw)})
                        rid = None
                    if rid is not None:
                        if rid not in lookups[mod].index:
                            invalid_refs.append({"incident_id": inc_id, "modality": mod, "reference": rid})
                        else:
                            rr = lookups[mod].loc[rid]
                            if isinstance(rr, pd.DataFrame):
                                raise ValueError(f"Source {mod} row id {rid} is duplicated in the source metadata.")
                            score = float(rr[f"score_{mod}"]); mask = 1
                            source_label = int(rr["label"]); source_ref = str(rid)
                            if mod == "email" and "score_email_llm" in rr.index:
                                row["score_email_llm"] = float(rr.get("score_email_llm", 0.5))
            if mask:
                used_refs[mod].append(str(source_ref))
                if source_label is not None and source_label != inc_label:
                    label_mismatches.append({
                        "incident_id": inc_id, "modality": mod, "source_reference": source_ref,
                        "incident_label": inc_label, "source_label": int(source_label),
                    })
            row[f"score_{mod}"] = float(score)
            row[f"mask_{mod}"] = int(mask)
            observed += int(mask)
        row["n_observed_channels"] = observed
        row["fusion_valid"] = int(observed >= 2)
        rows.append(row)

    duplicate_ref_rows = []
    for mod, refs in used_refs.items():
        vc = pd.Series(refs, dtype=str).value_counts()
        for ref, count in vc[vc > 1].items():
            duplicate_ref_rows.append({"modality": mod, "source_reference": ref, "incident_uses": int(count)})

    audit = {
        "manifest_rows": int(len(manifest)),
        "invalid_nonblank_references": int(len(invalid_refs)),
        "label_mismatches": int(len(label_mismatches)),
        "duplicated_source_references_across_incidents": int(len(duplicate_ref_rows)),
    }
    save_json(audit, cfg.out_root / "audit" / "incident_manifest_integrity.json")
    if invalid_refs:
        save_csv(pd.DataFrame(invalid_refs), cfg.out_root / "audit" / "incident_manifest_invalid_references.csv")
    if label_mismatches:
        save_csv(pd.DataFrame(label_mismatches), cfg.out_root / "audit" / "incident_manifest_label_mismatches.csv")
    if duplicate_ref_rows:
        save_csv(pd.DataFrame(duplicate_ref_rows), cfg.out_root / "audit" / "incident_manifest_duplicate_source_refs.csv")

    problems = []
    if invalid_refs:
        problems.append(f"{len(invalid_refs)} nonblank references do not resolve")
    if label_mismatches:
        problems.append(f"{len(label_mismatches)} source/incident label mismatches")
    if duplicate_ref_rows:
        problems.append(f"{len(duplicate_ref_rows)} source references are reused across incidents")
    if problems:
        raise RuntimeError("Incident manifest failed integrity audit: " + "; ".join(problems) + ". See audit/incident_manifest_*.csv")

    evidence = pd.DataFrame(rows)
    channel_counts = evidence[[f"mask_{m}" for m in MODALITIES]].sum(axis=1)
    save_json({
        "observed_channel_counts": channel_counts.value_counts().sort_index().to_dict(),
        "rows_with_two_or_more_channels": int(np.sum(channel_counts >= 2)),
        "multichannel_fraction": float(np.mean(channel_counts >= 2)),
    }, cfg.out_root / "audit" / "incident_manifest_channel_counts.json")
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


def _paired_bootstrap_f1(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                         n_boot: int, seed: int) -> Dict[str, float]:
    """Paired uncertainty analysis for F1(A)-F1(B) on identical rows.

    The percentile bootstrap supplies a confidence interval. A separate paired
    randomization test swaps A/B predictions independently per row under the
    null of exchangeability and provides the inferential p-value.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    observed = float(f1_score(y_true, pred_a, zero_division=0) - f1_score(y_true, pred_b, zero_division=0))
    if len(y_true) < 2 or n_boot <= 0:
        return {
            "f1_difference": observed, "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"), "paired_randomization_p": float("nan"),
        }
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_boot = int(n_boot)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = f1_score(y_true[idx], pred_a[idx], zero_division=0) - f1_score(y_true[idx], pred_b[idx], zero_division=0)
    lo, hi = np.quantile(diffs, [0.025, 0.975])

    # Paired randomization/permutation test. The +1 correction prevents a zero
    # Monte-Carlo p-value and is standard for finite randomization samples.
    extreme = 0
    for _ in range(n_boot):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, pred_b, pred_a)
        pb = np.where(swap, pred_a, pred_b)
        d = f1_score(y_true, pa, zero_division=0) - f1_score(y_true, pb, zero_division=0)
        if abs(d) >= abs(observed) - 1e-15:
            extreme += 1
    p_perm = float((extreme + 1) / (n_boot + 1))
    return {
        "f1_difference": observed,
        "bootstrap_ci_low": float(lo), "bootstrap_ci_high": float(hi),
        "paired_randomization_p": p_perm,
    }


def run_quantum_mechanism_audit(cfg: Config) -> None:
    """Verify that the entangling branch changes measured circuit features.

    This is a label-free circuit-mechanism audit, not a phishing-performance
    experiment and not a substitute for genuine incident-aligned data. It uses a
    deterministic two-modality score grid with URL and email masks active and
    compares otherwise matched local and entangled Qiskit feature maps.
    """
    if not QISKIT_AVAILABLE:
        return
    out_csv = cfg.out_root / "audit" / "quantum_pair_mechanism_grid.csv"
    out_json = cfg.out_root / "audit" / "quantum_pair_mechanism_summary.json"
    out_fig = cfg.out_root / "figures" / "quantum_pair_mechanism_audit.png"
    if cfg.resume_cache and out_csv.exists() and out_json.exists() and out_fig.exists():
        return
    grid = np.linspace(0.05, 0.95, 13)
    rows = []
    X = []
    for a in grid:
        for b in grid:
            X.append([a, b, 0.5, 0.5, 1, 1, 0, 0])
            rows.append({"score_url": float(a), "score_email": float(b)})
    X = np.asarray(X, dtype=float)
    local = QiskitQuantumFeatureMap(
        variant="no_entangle", layers=cfg.quantum_layers, seed=cfg.seed,
        pair_strength=cfg.quantum_pair_strength,
    ).fit(X)
    ent = QiskitQuantumFeatureMap(
        variant="entangled", layers=cfg.quantum_layers, seed=cfg.seed,
        pair_strength=cfg.quantum_pair_strength,
    ).fit(X)
    fl = local.transform(X)
    fe = ent.transform(X)
    delta = np.linalg.norm(fe - fl, axis=1)
    max_abs = np.max(np.abs(fe - fl), axis=1)
    for r, d, m in zip(rows, delta, max_abs):
        r["l2_feature_difference"] = float(d)
        r["max_abs_feature_difference"] = float(m)
    df = pd.DataFrame(rows)
    save_csv(df, out_csv)
    save_json({
        "purpose": "label-free mechanism audit; not a classification benchmark",
        "active_modalities": ["url", "email"],
        "grid_points": int(len(df)),
        "mean_l2_feature_difference": float(np.mean(delta)),
        "max_l2_feature_difference": float(np.max(delta)),
        "nonzero_difference_fraction": float(np.mean(delta > 1e-10)),
        "qpu_execution": False,
        "execution": "Qiskit Statevector ideal circuit simulation",
    }, out_json)

    mat = df.pivot(index="score_email", columns="score_url", values="l2_feature_difference").to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 6.1))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis",
                   extent=[grid.min(), grid.max(), grid.min(), grid.max()])
    ax.set_xlabel("URL specialist score")
    ax.set_ylabel("Email specialist score")
    ax.set_title("Entangling-circuit feature effect: local vs pair-gated QTG")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("L2 difference in 20 measured observables")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _save_quantum_circuit_visual(qc: Any, base_name: str, cfg: Config) -> Dict[str, Any]:
    """Persist a circuit in publication and audit formats.

    Qiskit's matplotlib drawer can fail when optional drawing packages such as
    pylatexenc are absent. This helper first requests the native Qiskit diagram.
    If that fails, it renders Qiskit's text circuit through matplotlib so PNG and
    JPEG artifacts are still guaranteed. The fallback is explicitly recorded in
    the accompanying status JSON and does not alter the circuit itself.
    """
    qdir = cfg.out_root / "audit" / "quantum_circuits"
    fdir = cfg.out_root / "figures" / "quantum_circuits"
    qdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    # Always save a text representation. This is dependency-light and useful for
    # exact visual audit even when the native matplotlib drawer is unavailable.
    try:
        text_drawing = str(qc.draw(output="text", fold=120))
    except Exception:
        text_drawing = str(qc)
    (qdir / f"{base_name}.txt").write_text(text_drawing, encoding="utf-8")

    native_error = None
    render_method = "qiskit_mpl"
    fig = None
    try:
        fig = qc.draw(output="mpl", fold=120)
        if fig is None or not hasattr(fig, "savefig"):
            raise RuntimeError("Qiskit matplotlib drawer returned no Figure object.")
    except Exception as exc:
        native_error = str(exc)
        render_method = "matplotlib_text_fallback"
        lines = text_drawing.splitlines() or [text_drawing]
        max_chars = max([len(x) for x in lines] + [80])
        width = min(34.0, max(12.0, max_chars * 0.105))
        height = min(24.0, max(4.0, len(lines) * 0.34 + 1.4))
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_facecolor("#F7FAFC")
        ax.set_facecolor("#F7FAFC")
        ax.axis("off")
        ax.text(
            0.01, 0.99, text_drawing,
            transform=ax.transAxes, va="top", ha="left",
            family="DejaVu Sans Mono", fontsize=8.2, color="#17202A",
        )
        ax.set_title(
            "Qiskit quantum circuit (text-render fallback)",
            loc="left", fontsize=12, fontweight="bold", color="#4B0082", pad=10,
        )
        fig.tight_layout()

    saved = []
    # Save both in audit/quantum_circuits and figures/quantum_circuits so the
    # circuit is easy to find for reproducibility and paper preparation.
    targets = [qdir, fdir]
    for directory in targets:
        for ext in ["png", "jpg", "pdf", "svg"]:
            path = directory / f"{base_name}.{ext}"
            try:
                kwargs = {"bbox_inches": "tight"}
                if ext in {"png", "jpg"}:
                    kwargs["dpi"] = 600 if ext == "png" else 450
                fig.savefig(path, **kwargs)
                saved.append(str(path))
            except Exception as exc:
                save_json(
                    {"format": ext, "error": str(exc)},
                    qdir / f"{base_name}_{ext}_save_error.json",
                )
    plt.close(fig)

    status = {
        "render_method": render_method,
        "native_qiskit_mpl_error": native_error,
        "saved_files": saved,
        "png_saved": any(x.lower().endswith(".png") for x in saved),
        "jpg_saved": any(x.lower().endswith(".jpg") for x in saved),
        "pdf_saved": any(x.lower().endswith(".pdf") for x in saved),
        "svg_saved": any(x.lower().endswith(".svg") for x in saved),
    }
    save_json(status, qdir / f"{base_name}_figure_status.json")
    return status


def _save_quantum_circuit_examples(mapper: Any, examples: Sequence[Tuple[str, np.ndarray, bool]],
                                   cfg: Config, prefix: str, seed: int, tag: str) -> List[Dict[str, Any]]:
    """Save QASM, metadata, text, PNG, JPEG, PDF and SVG for circuit examples."""
    if not QISKIT_AVAILABLE or not hasattr(mapper, "_build_circuit"):
        return []
    qdir = cfg.out_root / "audit" / "quantum_circuits"
    qdir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for label, row, is_empirical_multichannel in examples:
        row = np.asarray(row, dtype=float)
        qc = mapper._build_circuit(row)
        base_name = f"{prefix}_{tag}_{label}_seed{seed}"
        meta = {
            "tag": tag,
            "prefix": prefix,
            "seed": int(seed),
            "example": label,
            "empirical_multichannel_row": bool(is_empirical_multichannel),
            "scores": [float(v) for v in row[:4]],
            "masks": [float(v) for v in row[4:8]],
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "num_qubits": int(qc.num_qubits),
            "operations": {str(k): int(v) for k, v in qc.count_ops().items()},
            "execution": "Qiskit Statevector ideal circuit simulation",
            "qpu_execution": False,
        }
        save_json(meta, qdir / f"{base_name}.json")
        try:
            from qiskit import qasm3
            (qdir / f"{base_name}.qasm").write_text(qasm3.dumps(qc), encoding="utf-8")
            meta["qasm_saved"] = True
        except Exception as exc:
            meta["qasm_saved"] = False
            meta["qasm_export_error"] = str(exc)
            save_json({"qasm_export_error": str(exc)}, qdir / f"{base_name}_qasm_error.json")

        if cfg.save_quantum_circuit_figures:
            fig_status = _save_quantum_circuit_visual(qc, base_name, cfg)
            meta.update(fig_status)
        save_json(meta, qdir / f"{base_name}.json")
        records.append(meta)
    return records


def save_quantum_circuit_audit(mapper: Any, evidence: pd.DataFrame, cfg: Config,
                               prefix: str, seed: int, tag: str) -> None:
    """Save one evaluated circuit plus clearly labelled circuit design templates."""
    if not QISKIT_AVAILABLE or not hasattr(mapper, "_build_circuit"):
        return
    X = arbitration_features(evidence)
    if len(X) == 0:
        return
    active_counts = X[:, 4:8].sum(axis=1)
    empirical_idx = int(np.flatnonzero(active_counts >= 2)[0]) if np.any(active_counts >= 2) else 0
    examples = [
        ("empirical", X[empirical_idx], bool(active_counts[empirical_idx] >= 2)),
        # This is a circuit-design visualization, not a synthetic performance row.
        ("design_template_all_channels", np.array([0.82, 0.74, 0.66, 0.91, 1, 1, 1, 1], dtype=float), False),
        ("design_template_two_channels", np.array([0.82, 0.31, 0.50, 0.50, 1, 1, 0, 0], dtype=float), False),
    ]
    records = _save_quantum_circuit_examples(mapper, examples, cfg, prefix, seed, tag)
    save_json(
        {"prefix": prefix, "seed": seed, "tag": tag, "n_examples": len(records), "examples": records},
        cfg.out_root / "audit" / "quantum_circuits" / f"{prefix}_{tag}_seed{seed}_manifest.json",
    )


def regenerate_quantum_artifacts_from_saved_models(cfg: Config, prefix: str) -> Dict[str, Any]:
    """Create missing circuit figures from cached V6/V7/V8 models without retraining.

    This function is the fast path for completed experiments. It never changes
    predictions or metrics. It only re-renders circuit artifacts from the exact
    saved mapper objects and regenerates aggregate result figures/tables.
    """
    seeds = [cfg.seed + i for i in range(cfg.repeats)]
    tags = {
        "V6_no_entangle": "V6_QTG_NoEntangle",
        "V7_entangled": "V7_QTG_Entangled",
        "V8_RC_QTG": "V8_Proposed_RC_QTG",
    }
    summary: Dict[str, Any] = {"prefix": prefix, "seeds": seeds, "rendered": [], "missing_models": [], "errors": []}
    examples = [
        ("cached_design_template_all_channels", np.array([0.82, 0.74, 0.66, 0.91, 1, 1, 1, 1], dtype=float), False),
        ("cached_design_template_two_channels", np.array([0.82, 0.31, 0.50, 0.50, 1, 1, 0, 0], dtype=float), False),
        ("cached_design_template_single_channel", np.array([0.82, 0.50, 0.50, 0.50, 1, 0, 0, 0], dtype=float), False),
    ]
    for seed in seeds:
        for tag, model_token in tags.items():
            path = cfg.out_root / "models" / f"{prefix}_{model_token}_seed{seed}.joblib"
            if not path.exists():
                summary["missing_models"].append(str(path))
                continue
            try:
                bundle = joblib.load(path)
                mapper = bundle.get("mapper") if isinstance(bundle, dict) else None
                if mapper is None or not hasattr(mapper, "_build_circuit"):
                    raise TypeError(f"Saved model does not contain a quantum mapper: {path.name}")
                records = _save_quantum_circuit_examples(mapper, examples, cfg, prefix, seed, tag)
                summary["rendered"].append({"seed": seed, "tag": tag, "model": str(path), "n_examples": len(records)})
            except Exception as exc:
                summary["errors"].append({"seed": seed, "tag": tag, "model": str(path), "error": str(exc)})
    qdir = cfg.out_root / "audit" / "quantum_circuits"
    png_count = len(list(qdir.glob(f"{prefix}_*.png")))
    jpg_count = len(list(qdir.glob(f"{prefix}_*.jpg")))
    summary["audit_png_count"] = png_count
    summary["audit_jpg_count"] = jpg_count
    summary["success"] = bool(png_count > 0 and jpg_count > 0 and not summary["errors"])
    save_json(summary, qdir / f"{prefix}_artifact_regeneration_summary.json")
    return summary


def regenerate_cached_figures(cfg: Config, prefix: str) -> None:
    """Regenerate paper figures from existing result CSVs without model fitting."""
    aggregate_repeats(cfg, prefix)
    for result_path in sorted((cfg.out_root / "results").glob(f"{prefix}_arbitration_seed*.csv")):
        try:
            seed_match = re.search(r"seed(\d+)", result_path.stem)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))
            df = pd.read_csv(result_path)
            draw_arbitration_figure(df, cfg.out_root / "figures" / f"{prefix}_arbitration_seed{seed}.png")
            pred_path = cfg.out_root / "predictions" / f"{prefix}_arbitration_predictions_seed{seed}.csv"
            if pred_path.exists():
                draw_primary_diagnostics(pd.read_csv(pred_path), df, cfg.out_root / "figures", prefix, seed)
            rob_path = cfg.out_root / "robustness" / f"{prefix}_robustness_seed{seed}.csv"
            if rob_path.exists():
                draw_robustness_figure(pd.read_csv(rob_path), cfg.out_root / "figures" / f"{prefix}_robustness_seed{seed}.png")
            faith_path = cfg.out_root / "faithfulness" / f"{prefix}_faithfulness_summary_seed{seed}.csv"
            if faith_path.exists():
                draw_faithfulness_figure(pd.read_csv(faith_path), cfg.out_root / "figures" / f"{prefix}_faithfulness_seed{seed}.png")
        except Exception as exc:
            save_json(
                {"prefix": prefix, "result_file": str(result_path), "error": str(exc)},
                cfg.out_root / "audit" / f"{prefix}_cached_figure_regeneration_error_{result_path.stem}.json",
            )


def write_experiment_completeness_audit(cfg: Config) -> Dict[str, Any]:
    """Write a machine-readable final checklist for the completed experiment."""
    required_common = [
        cfg.out_root / "results" / "human_llm_true_generalization.csv",
        cfg.out_root / "audit" / "qiskit_environment.json",
        cfg.out_root / "audit" / "quantum_pair_mechanism_summary.json",
        cfg.out_root / "figures" / "quantum_pair_mechanism_audit.png",
    ]
    required_disjoint = [
        cfg.out_root / "tables" / "DISJOINT_arbitration_summary_ci.csv",
        cfg.out_root / "tables" / "DISJOINT_paired_primary_summary.csv",
        cfg.out_root / "tables" / "DISJOINT_nested_specialists_summary_ci.csv",
        cfg.out_root / "audit" / "DISJOINT_strict_nested_protocol.json",
        cfg.out_root / "figures" / "DISJOINT_publication_f1_ci.png",
        cfg.out_root / "figures" / "DISJOINT_quantum_ablation.png",
    ]
    required = list(required_common)
    if cfg.mode in {"all", "disjoint"}:
        required.extend(required_disjoint)
    missing = [str(p) for p in required if not p.exists()]
    qdir = cfg.out_root / "audit" / "quantum_circuits"
    circuit_pngs = sorted(str(p) for p in qdir.glob("*.png"))
    circuit_jpgs = sorted(str(p) for p in qdir.glob("*.jpg"))
    if not circuit_pngs:
        missing.append("quantum circuit PNG figures")
    if not circuit_jpgs:
        missing.append("quantum circuit JPEG figures")

    scope_boundaries = [
        "Qiskit circuits are executed with ideal Statevector simulation, not on a quantum processor.",
    ]
    if cfg.mode in {"all", "disjoint"}:
        scope_boundaries.append(
            "DISJOINT evidence contains one observed modality per row; genuine same-incident cross-channel fusion and empirical entanglement activation are not claimed."
        )

    report = {
        "status": "COMPLETE_WITH_SCOPE_BOUNDARIES" if not missing else "INCOMPLETE_ARTIFACTS",
        "missing_required_artifacts": missing,
        "quantum_circuit_png_count": len(circuit_pngs),
        "quantum_circuit_jpg_count": len(circuit_jpgs),
        "scope_boundaries": scope_boundaries,
        "results_must_not_be_recomputed_for_artifact_regeneration": True,
    }
    save_json(report, cfg.out_root / "reports" / "EXPERIMENT_COMPLETENESS_AUDIT.json")
    lines = [
        f"STATUS: {report['status']}",
        f"Quantum circuit PNG files: {len(circuit_pngs)}",
        f"Quantum circuit JPEG files: {len(circuit_jpgs)}",
        "",
        "Missing required artifacts:",
        *([f"- {x}" for x in missing] if missing else ["- None"]),
        "",
        "Scientific claim boundaries:",
        *[f"- {x}" for x in scope_boundaries],
    ]
    (cfg.out_root / "reports" / "EXPERIMENT_COMPLETENESS_AUDIT.txt").write_text("\n".join(lines), encoding="utf-8")
    return report


def train_arbiters_once(evidence: pd.DataFrame, cfg: Config, seed: int, prefix: str) -> pd.DataFrame:
    """Evaluate eight matched arbitration variants on exactly the same held-out rows.

    V1  masked average.
    V2  true majority vote.
    V3  learned linear late fusion.
    V4  mask-aware Trust Graph Arbitration (TGA).
    V5  matched classical nonlinear RFF control.
    V6  Qiskit local/no-entanglement circuit.
    V7  Qiskit entangled circuit with the same encoding/readout as V6.
    V8  proposed reliability-calibrated mask-gated entangled circuit (RC-QTG).

    In disjoint mode V6/V7 are expected to be identical or nearly identical
    because no row contains an observed modality pair. That negative control is
    scientifically useful and prevents attributing single-channel gains to
    entanglement that was never activated.
    """
    y = evidence["label"].astype(int).to_numpy()
    X = arbitration_features(evidence)
    if len(evidence) < 10 or len(np.unique(y)) < 2:
        raise RuntimeError(f"{prefix}: arbitration requires >=10 rows and both classes.")
    idx = np.arange(len(evidence))
    if "outer_role" in evidence.columns:
        role = evidence["outer_role"].astype(str).str.lower().to_numpy()
        tr = np.flatnonzero(role == "train")
        te = np.flatnonzero(role == "test")
        if len(tr) == 0 or len(te) == 0:
            raise RuntimeError(f"{prefix}: outer_role is present but train/test rows are missing.")
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            raise RuntimeError(f"{prefix}: fixed outer train/test split must contain both classes.")
    else:
        tr, te = train_test_split(idx, test_size=cfg.test_size, stratify=y, random_state=seed)
    masks_te = X[te, 4:8]
    multichannel_te = int(np.sum(masks_te.sum(axis=1) >= 2))
    pair_active_rate = float(multichannel_te / max(len(te), 1))
    save_json({
        "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_rows_with_two_or_more_modalities": multichannel_te,
        "test_pair_interaction_activation_rate": pair_active_rate,
        "entanglement_claim_supported": bool(multichannel_te > 0),
    }, cfg.out_root / "audit" / f"{prefix}_quantum_interaction_audit_seed{seed}.json")

    result_path = cfg.out_root / "results" / f"{prefix}_arbitration_seed{seed}.csv"
    robustness_path = cfg.out_root / "robustness" / f"{prefix}_robustness_seed{seed}.csv"
    faithfulness_path = cfg.out_root / "faithfulness" / f"{prefix}_faithfulness_summary_seed{seed}.csv"
    model_paths = {
        "V3": cfg.out_root / "models" / f"{prefix}_V3_LearnedLateFusion_seed{seed}.joblib",
        "V4": cfg.out_root / "models" / f"{prefix}_V4_TGA_seed{seed}.joblib",
        "V5": cfg.out_root / "models" / f"{prefix}_V5_RFF_Control_seed{seed}.joblib",
        "V6": cfg.out_root / "models" / f"{prefix}_V6_QTG_NoEntangle_seed{seed}.joblib",
        "V7": cfg.out_root / "models" / f"{prefix}_V7_QTG_Entangled_seed{seed}.joblib",
        "V8": cfg.out_root / "models" / f"{prefix}_V8_Proposed_RC_QTG_seed{seed}.joblib",
    }

    # New V8 results must contain all eight experiments before cache reuse.
    if cfg.resume_cache and result_path.exists():
        try:
            cached = pd.read_csv(result_path)
            required = {f"{prefix}_V{i}_" for i in range(1, 9)}
            names = cached.get("experiment", pd.Series(dtype=str)).astype(str).tolist()
            complete = len(cached) >= 8 and all(any(n.startswith(r) for n in names) for r in required)
            if complete and robustness_path.exists() and faithfulness_path.exists():
                print(f"[{prefix} seed {seed}] V8 arbitration/robustness/faithfulness: cached")
                # Results are already complete, but drawing dependencies may have
                # prevented circuit PNG/JPEG generation in an earlier run. Rebuild
                # visual artifacts from the exact cached quantum mapper objects.
                regenerate_quantum_artifacts_from_saved_models(cfg, prefix)
                return cached
        except Exception:
            pass

    save_csv(pd.DataFrame({"train_index": tr}), cfg.out_root / "splits" / f"{prefix}_arbiter_train_seed{seed}.csv")
    save_csv(pd.DataFrame({"test_index": te}), cfg.out_root / "splits" / f"{prefix}_arbiter_test_seed{seed}.csv")

    rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    prediction_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def add_result(name, score, pred, train_time, predict_time):
        score = np.asarray(score, dtype=float)
        pred = np.asarray(pred, dtype=int)
        m = binary_metrics(y[te], pred, score)
        m.update({
            "experiment": name, "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
            "train_time_sec": float(train_time), "predict_time_sec": float(predict_time),
            "multichannel_test_rows": multichannel_te, "pair_interaction_activation_rate": pair_active_rate,
        })
        rows.append(m)
        prediction_cache[name] = (score, pred)
        pred_frames.append(pd.DataFrame({
            "experiment": name, "seed": seed,
            "incident_id": evidence.iloc[te]["incident_id"].astype(str).to_numpy(),
            "y_true": y[te], "y_score": score, "y_pred": pred,
        }))

    # V1: masked score average.
    t = time.perf_counter(); avg_score, avg_pred = average_scores(evidence.iloc[te])
    add_result(f"{prefix}_V1_AverageScore", avg_score, avg_pred, 0.0, time.perf_counter() - t)

    # V2: actual hard-vote aggregation.
    t = time.perf_counter(); mv_score, mv_pred = majority_vote_scores(evidence.iloc[te], cfg.threshold)
    add_result(f"{prefix}_V2_TrueMajorityVote", mv_score, mv_pred, 0.0, time.perf_counter() - t)

    # V3: linear late fusion.
    late = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    t = time.perf_counter(); late.fit(X[tr], y[tr]); train_time = time.perf_counter() - t
    t = time.perf_counter(); score = late.predict_proba(X[te])[:, 1]; pred_time = time.perf_counter() - t
    add_result(f"{prefix}_V3_LearnedLateFusion", score, score >= cfg.threshold, train_time, pred_time)
    joblib.dump(late, model_paths["V3"])

    # V4: explicit graph representation.
    graph = Pipeline([
        ("graph", TrustGraphEncoder(sigma=cfg.trust_sigma)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    t = time.perf_counter(); graph.fit(X[tr], y[tr]); train_time = time.perf_counter() - t
    t = time.perf_counter(); graph_score = graph.predict_proba(X[te])[:, 1]; pred_time = time.perf_counter() - t
    add_result(f"{prefix}_V4_TGA", graph_score, graph_score >= cfg.threshold, train_time, pred_time)
    joblib.dump(graph, model_paths["V4"])

    # V5: matched classical nonlinear control.
    rff = Pipeline([
        ("scale", StandardScaler()),
        ("rff", RBFSampler(gamma=cfg.rff_gamma, n_components=cfg.rff_dim, random_state=seed)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    t = time.perf_counter(); rff.fit(X[tr], y[tr]); train_time = time.perf_counter() - t
    t = time.perf_counter(); rff_score = rff.predict_proba(X[te])[:, 1]; pred_time = time.perf_counter() - t
    add_result(f"{prefix}_V5_RFF_Control", rff_score, rff_score >= cfg.threshold, train_time, pred_time)
    joblib.dump(rff, model_paths["V5"])

    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required for V6-V8. Install qiskit and qiskit-aer.")

    # V6: Qiskit local/no-entanglement negative control.
    qlocal = QiskitQuantumFeatureMap(
        variant="no_entangle", layers=cfg.quantum_layers, seed=seed,
        pair_strength=cfg.quantum_pair_strength,
    )
    t = time.perf_counter(); qlocal.fit(X[tr], y[tr]); qtr_local = qlocal.transform(X[tr]); qte_local = qlocal.transform(X[te])
    qlocal_clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))])
    qlocal_clf.fit(qtr_local, y[tr]); train_time = time.perf_counter() - t
    t = time.perf_counter(); qlocal_score = qlocal_clf.predict_proba(qte_local)[:, 1]; pred_time = time.perf_counter() - t
    add_result(f"{prefix}_V6_QTG_NoEntangle", qlocal_score, qlocal_score >= cfg.threshold, train_time, pred_time)
    joblib.dump({"mapper": qlocal, "classifier": qlocal_clf}, model_paths["V6"])

    # V7: identical encoding/readout plus pair interactions.
    qent = QiskitQuantumFeatureMap(
        variant="entangled", layers=cfg.quantum_layers, seed=seed,
        pair_strength=cfg.quantum_pair_strength,
    )
    t = time.perf_counter(); qent.fit(X[tr], y[tr]); qtr_ent = qent.transform(X[tr]); qte_ent = qent.transform(X[te])
    qent_clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))])
    qent_clf.fit(qtr_ent, y[tr]); train_time = time.perf_counter() - t
    t = time.perf_counter(); qent_score = qent_clf.predict_proba(qte_ent)[:, 1]; pred_time = time.perf_counter() - t
    add_result(f"{prefix}_V7_QTG_Entangled", qent_score, qent_score >= cfg.threshold, train_time, pred_time)
    joblib.dump({"mapper": qent, "classifier": qent_clf}, model_paths["V7"])

    # V8: proposed reliability-calibrated mask-gated quantum fusion.
    proposed = ReliabilityCalibratedQuantumFusion(
        layers=cfg.quantum_layers, pair_strength=cfg.quantum_pair_strength,
        reliability_power=cfg.reliability_power, seed=seed,
    )
    t = time.perf_counter(); proposed.fit(X[tr], y[tr]); ptr = proposed.transform(X[tr]); pte = proposed.transform(X[te])
    pclf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))])
    pclf.fit(ptr, y[tr]); train_time = time.perf_counter() - t
    t = time.perf_counter(); p_score = pclf.predict_proba(pte)[:, 1]; pred_time = time.perf_counter() - t
    add_result(f"{prefix}_V8_Proposed_RC_QTG", p_score, p_score >= cfg.threshold, train_time, pred_time)
    joblib.dump({"mapper": proposed, "classifier": pclf}, model_paths["V8"])
    save_csv(pd.DataFrame(proposed.reliability_details_), cfg.out_root / "audit" / f"{prefix}_RC_QTG_reliability_seed{seed}.csv")

    out = pd.DataFrame(rows)
    save_csv(out, result_path)
    pred_all = pd.concat(pred_frames, ignore_index=True)
    save_csv(pred_all, cfg.out_root / "predictions" / f"{prefix}_arbitration_predictions_seed{seed}.csv")

    # Paired primary-model comparisons on identical held-out rows.
    stats_rows = []
    proposed_name = f"{prefix}_V8_Proposed_RC_QTG"
    for baseline_name in [f"{prefix}_V4_TGA", f"{prefix}_V5_RFF_Control", f"{prefix}_V7_QTG_Entangled"]:
        _, p_prop = prediction_cache[proposed_name]
        _, p_base = prediction_cache[baseline_name]
        bs = _paired_bootstrap_f1(y[te], p_prop, p_base, cfg.bootstrap_samples, seed + sum((i + 1) * ord(ch) for i, ch in enumerate(baseline_name)) % 10000)
        bs.update({"seed": seed, "proposed": proposed_name, "baseline": baseline_name, "n_test": len(te)})
        stats_rows.append(bs)
    save_csv(pd.DataFrame(stats_rows), cfg.out_root / "results" / f"{prefix}_paired_primary_tests_seed{seed}.csv")

    # Audit exact circuits. In disjoint mode the empirical circuit truthfully
    # contains no pair gate; the separate design template is explicitly non-data.
    save_quantum_circuit_audit(qlocal, evidence.iloc[te].reset_index(drop=True), cfg, prefix, seed, "V6_no_entangle")
    save_quantum_circuit_audit(qent, evidence.iloc[te].reset_index(drop=True), cfg, prefix, seed, "V7_entangled")
    save_quantum_circuit_audit(proposed, evidence.iloc[te].reset_index(drop=True), cfg, prefix, seed, "V8_RC_QTG")

    comparison_models = {
        "RFF_Control": rff,
        "QTG_NoEntangle": (qlocal, qlocal_clf),
        "QTG_Entangled": (qent, qent_clf),
        "RC_QTG_Proposed": (proposed, pclf),
    }
    robustness_and_faithfulness(
        evidence.iloc[te].reset_index(drop=True), graph, comparison_models,
        cfg, prefix, seed,
    )
    draw_arbitration_figure(out, cfg.out_root / "figures" / f"{prefix}_arbitration_seed{seed}.png")
    draw_primary_diagnostics(pred_all, out, cfg.out_root / "figures", prefix, seed)
    return out


def _model_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for a fitted arbiter.

    Supported representations:
      * sklearn Pipeline/estimator with ``predict_proba``;
      * ``(mapper, classifier)`` tuple used in-memory by V5/V6;
      * ``{"mapper": ..., "classifier": ...}`` bundle saved with joblib.

    Keeping one scoring path prevents the V6 tuple/dictionary mismatch that
    previously caused ``ReliabilityCalibratedQuantumFusion is not subscriptable``.
    """
    X = np.asarray(X, dtype=float)
    if isinstance(model, dict):
        if "mapper" in model and "classifier" in model:
            mapper = model["mapper"]
            clf = model["classifier"]
            return np.asarray(clf.predict_proba(mapper.transform(X))[:, 1], dtype=float)
        raise TypeError(f"Unsupported saved model bundle keys: {sorted(model.keys())}")
    if isinstance(model, tuple) and len(model) == 2:
        mapper, clf = model
        return np.asarray(clf.predict_proba(mapper.transform(X))[:, 1], dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(get_score(model, X), dtype=float)


def robustness_and_faithfulness(
    test_evidence: pd.DataFrame,
    graph_model: Any,
    quantum_models: Dict[str, Any],
    cfg: Config,
    prefix: str,
    seed: int,
) -> None:
    """Evaluate trained arbiters under matched corruption and deletion tests.

    Corruption matrices are generated once per seed and reused for every
    arbiter, giving paired robustness comparisons. Faithfulness is evaluated
    on the proposed V6 model when available, otherwise V5.
    """
    y = test_evidence["label"].astype(int).to_numpy()
    base_X = arbitration_features(test_evidence)

    # Generate identical poisoned evidence for every model in this seed.
    poisoned: Dict[str, np.ndarray] = {}
    for rate in [0.05, 0.10, 0.20, 0.30]:
        rng = np.random.default_rng(seed * 1000 + int(rate * 100))
        Xp = base_X.copy()
        masks = Xp[:, 4:8] > 0.5
        for i in range(Xp.shape[0]):
            observed = np.flatnonzero(masks[i])
            if observed.size and rng.random() < rate:
                flip = int(rng.choice(observed))
                Xp[i, flip] = 1.0 - Xp[i, flip]
        poisoned[f"score_poison_{int(rate*100)}pct"] = Xp

    missing: Dict[str, np.ndarray] = {}
    for mod_i, mod in enumerate(MODALITIES):
        Xm = base_X.copy()
        Xm[:, mod_i] = 0.5
        Xm[:, 4 + mod_i] = 0.0
        missing[f"missing_{mod}"] = Xm

    model_items: List[Tuple[str, Any]] = [("TGA", graph_model)]
    model_items.extend(list(quantum_models.items()))
    rows: List[Dict[str, Any]] = []

    for model_name, model in model_items:
        base_score = _model_score(model, base_X)
        base_pred = (base_score >= cfg.threshold).astype(int)
        rows.append({"model": model_name, "condition": "unpoisoned",
                     **binary_metrics(y, base_pred, base_score)})

        for condition, Xp in poisoned.items():
            score = _model_score(model, Xp)
            pred = (score >= cfg.threshold).astype(int)
            rows.append({"model": model_name, "condition": condition,
                         **binary_metrics(y, pred, score)})

        for condition, Xm in missing.items():
            score = _model_score(model, Xm)
            pred = (score >= cfg.threshold).astype(int)
            rows.append({"model": model_name, "condition": condition,
                         **binary_metrics(y, pred, score)})

    rob = pd.DataFrame(rows)
    save_csv(rob, cfg.out_root / "robustness" / f"{prefix}_robustness_seed{seed}.csv")
    draw_robustness_figure(rob, cfg.out_root / "figures" / f"{prefix}_robustness_seed{seed}.png")

    # Use the proposed RC-QTG arbiter for perturbation diagnostics. The
    # entangled QTG control is retained as a fallback.
    primary_name = "RC_QTG_Proposed" if "RC_QTG_Proposed" in quantum_models else "QTG_Entangled"
    primary_q = quantum_models.get(primary_name)
    if primary_q is None:
        return

    n = len(base_X)
    base_score = _model_score(primary_q, base_X)
    deletion_delta = np.full((n, 4), np.nan, dtype=float)
    sufficiency_score = np.full((n, 4), np.nan, dtype=float)
    observed_mask = base_X[:, 4:8] > 0.5

    # Batch each modality deletion/retention rather than evaluating one circuit
    # per row. This is materially faster for the statevector simulator.
    for mod_i, mod in enumerate(MODALITIES):
        observed = observed_mask[:, mod_i]
        if not np.any(observed):
            continue

        Xdel = base_X.copy()
        Xdel[:, mod_i] = 0.5
        Xdel[:, 4 + mod_i] = 0.0
        dscore = _model_score(primary_q, Xdel)
        deletion_delta[observed, mod_i] = base_score[observed] - dscore[observed]

        Xkeep = np.zeros_like(base_X, dtype=float)
        Xkeep[:, :4] = 0.5
        Xkeep[:, 4:8] = 0.0
        Xkeep[:, mod_i] = base_X[:, mod_i]
        Xkeep[:, 4 + mod_i] = base_X[:, 4 + mod_i]
        kscore = _model_score(primary_q, Xkeep)
        sufficiency_score[observed, mod_i] = kscore[observed]

    # Positive deletion effects are normalized only across modalities that are
    # actually observed for that row. Negative effects remain visible in the
    # signed deletion columns but receive zero positive contribution mass.
    positive = np.where(np.isfinite(deletion_delta), np.maximum(deletion_delta, 0.0), 0.0)
    denom = positive.sum(axis=1, keepdims=True)
    contribution = np.divide(positive, denom, out=np.zeros_like(positive), where=denom > 1e-12)
    contribution[~observed_mask] = np.nan

    # Comprehensiveness: remove every observed modality simultaneously.
    Xnone = np.zeros_like(base_X, dtype=float)
    Xnone[:, :4] = 0.5
    Xnone[:, 4:8] = 0.0
    none_score = _model_score(primary_q, Xnone)
    comprehensiveness_delta = base_score - none_score

    expl = pd.DataFrame({
        "incident_id": test_evidence["incident_id"].astype(str).to_numpy(),
        "label": y,
        "model": primary_name,
        "base_score": base_score,
        "all_deleted_score": none_score,
        "comprehensiveness_delta": comprehensiveness_delta,
    })
    for mod_i, mod in enumerate(MODALITIES):
        expl[f"observed_{mod}"] = observed_mask[:, mod_i].astype(int)
        expl[f"deletion_delta_{mod}"] = deletion_delta[:, mod_i]
        expl[f"normalized_contribution_{mod}"] = contribution[:, mod_i]
        expl[f"sufficiency_score_{mod}"] = sufficiency_score[:, mod_i]

    save_csv(expl, cfg.out_root / "faithfulness" / f"{prefix}_faithfulness_seed{seed}.csv")

    summary_rows: List[Dict[str, Any]] = []
    for mod_i, mod in enumerate(MODALITIES):
        d = deletion_delta[:, mod_i]
        c = contribution[:, mod_i]
        sf = sufficiency_score[:, mod_i]
        valid = np.isfinite(d)
        if not np.any(valid):
            summary_rows.append({
                "modality": mod, "n_observed": 0,
                "mean_deletion_impact": float("nan"),
                "mean_signed_deletion_delta": float("nan"),
                "median_deletion_impact": float("nan"),
                "mean_contribution": float("nan"),
                "mean_sufficiency_score": float("nan"),
            })
            continue
        summary_rows.append({
            "modality": mod,
            "n_observed": int(valid.sum()),
            "mean_deletion_impact": float(np.nanmean(np.abs(d))),
            "mean_signed_deletion_delta": float(np.nanmean(d)),
            "median_deletion_impact": float(np.nanmedian(np.abs(d))),
            "mean_contribution": float(np.nanmean(c)),
            "mean_sufficiency_score": float(np.nanmean(sf)),
        })

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, cfg.out_root / "faithfulness" / f"{prefix}_faithfulness_summary_seed{seed}.csv")
    save_json({
        "model": primary_name,
        "mean_absolute_comprehensiveness": float(np.mean(np.abs(comprehensiveness_delta))),
        "mean_signed_comprehensiveness": float(np.mean(comprehensiveness_delta)),
        "n_test": int(n),
    }, cfg.out_root / "faithfulness" / f"{prefix}_faithfulness_overall_seed{seed}.json")
    draw_faithfulness_figure(summary_df, cfg.out_root / "figures" / f"{prefix}_faithfulness_seed{seed}.png")

def _short_method_name(x: str) -> str:
    x = str(x).replace("DISJOINT_", "").replace("INCIDENT_", "")
    replacements = {
        "V1_AverageScore": "Average",
        "V2_TrueMajorityVote": "Majority",
        "V3_LearnedLateFusion": "Late Fusion",
        "V4_TGA": "TGA",
        "V5_RFF_Control": "RFF",
        "V6_QTG_NoEntangle": "QTG local",
        "V7_QTG_Entangled": "QTG entangled",
        "V8_Proposed_RC_QTG": "RC-QTG",
    }
    for k, v in replacements.items():
        if x == k or x.endswith(k):
            return v
    return x.replace("_", " ")


def draw_arbitration_figure(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    methods = [_short_method_name(x) for x in df["experiment"]]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    palette = ["#2E86AB", "#F18F01", "#6A994E", "#C73E1D", "#7B2CBF", "#00A6A6"]
    x = np.arange(len(df))
    width = 0.12
    fig, ax = plt.subplots(figsize=(15, 6.4))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2.5) * width, df[m].astype(float).values, width,
               label=m.upper().replace("ROC_AUC", "ROC-AUC").replace("PR_AUC", "PR-AUC"),
               color=palette[i], alpha=0.92, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=22, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Score")
    ax.set_title("QTrustAgent-X arbitration comparison")
    ax.grid(axis="y", alpha=0.20, linestyle="--")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_robustness_figure(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    palette = ["#264653", "#E76F51", "#2A9D8F", "#E9C46A", "#7B2CBF", "#D62828"]
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    for i, model in enumerate(df["model"].unique()):
        sub = df[df["model"] == model]
        ax.plot(np.arange(len(sub)), sub["f1"].astype(float), marker="o", linewidth=2.1,
                markersize=5.5, label=model.replace("_", " "), color=palette[i % len(palette)])
    conditions = list(df[df["model"] == df["model"].iloc[0]]["condition"].astype(str))
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([c.replace("score_poison_", "poison ").replace("pct", "%").replace("missing_", "remove ") for c in conditions],
                       rotation=28, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("F1-score")
    ax.set_title("Robustness of fixed trained arbiters")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_faithfulness_figure(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - 0.22, df["mean_deletion_impact"], width=0.44, label="Deletion impact",
           color="#E76F51", edgecolor="white")
    ax.bar(x + 0.22, df["mean_contribution"], width=0.44, label="Normalized contribution",
           color="#2A9D8F", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["modality"].str.upper())
    ax.set_ylabel("Sensitivity score")
    ax.set_title("Perturbation-based evidence sensitivity")
    ax.grid(axis="y", alpha=0.20, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_primary_diagnostics(pred_df: pd.DataFrame, metric_df: pd.DataFrame, fig_dir: Path,
                             prefix: str, seed: int) -> None:
    """Paper-ready ROC and confusion-matrix diagnostics for the proposed model."""
    proposed = f"{prefix}_V8_Proposed_RC_QTG"
    sub = pred_df[pred_df["experiment"] == proposed].copy()
    if sub.empty:
        return
    y = sub["y_true"].to_numpy(int)
    score = sub["y_score"].to_numpy(float)
    pred = sub["y_pred"].to_numpy(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6.1, 5.3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Phishing"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"RC-QTG confusion matrix, seed {seed}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_RC_QTG_confusion_seed{seed}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, score)
        aucv = roc_auc_score(y, score)
        fig, ax = plt.subplots(figsize=(6.3, 5.3))
        ax.plot(fpr, tpr, linewidth=2.6, color="#7B2CBF", label=f"RC-QTG (AUC={aucv:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#6C757D")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.set_title(f"RC-QTG ROC curve, seed {seed}")
        ax.grid(alpha=0.18, linestyle="--")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prefix}_RC_QTG_ROC_seed{seed}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

def aggregate_repeats(cfg: Config, prefix: str) -> None:
    files = sorted((cfg.out_root / "results").glob(f"{prefix}_arbitration_seed*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    # Guard against stale pre-v8 files if the user reused the same result folder.
    df = df[df["experiment"].astype(str).str.contains(r"_V[1-8]_", regex=True)].copy()
    save_csv(df, cfg.out_root / "tables" / f"{prefix}_arbitration_all_repeats.csv")

    rows = []
    for exp, g in df.groupby("experiment"):
        row = {"experiment": exp, "n_runs": len(g)}
        for m in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier", "mcc", "fpr", "fnr", "train_time_sec", "predict_time_sec", "pair_interaction_activation_rate"]:
            if m in g.columns:
                mean, low, high = ci95(g[m])
                row[m] = mean
                row[f"{m}_ci_low"] = low
                row[f"{m}_ci_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)
    save_csv(summary, cfg.out_root / "tables" / f"{prefix}_arbitration_summary_ci.csv")
    write_latex_tables(cfg, prefix, summary)

    # Aggregate paired primary comparisons across the repeated outer/test splits.
    paired_files = sorted((cfg.out_root / "results").glob(f"{prefix}_paired_primary_tests_seed*.csv"))
    if paired_files:
        paired = pd.concat([pd.read_csv(x) for x in paired_files], ignore_index=True)
        save_csv(paired, cfg.out_root / "tables" / f"{prefix}_paired_primary_tests_all.csv")
        paired_summary = []
        for baseline, g in paired.groupby("baseline"):
            d = g["f1_difference"].astype(float).to_numpy()
            mean, lo, hi = ci95(d)
            paired_summary.append({
                "baseline": baseline,
                "mean_f1_difference_RC_QTG_minus_baseline": mean,
                "ci_low": lo, "ci_high": hi,
                "positive_difference_runs": int(np.sum(d > 0)),
                "negative_difference_runs": int(np.sum(d < 0)),
                "ties": int(np.sum(d == 0)),
                "n_runs": int(len(d)),
            })
        save_csv(pd.DataFrame(paired_summary), cfg.out_root / "tables" / f"{prefix}_paired_primary_summary.csv")

    # Colorful publication figure with repeated-run means and 95% CIs.
    if not summary.empty:
        order = summary.sort_values("experiment")
        names = [_short_method_name(x) for x in order["experiment"]]
        palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
        x = np.arange(len(order))
        f1 = order["f1"].to_numpy(float)
        f1err = np.vstack([f1 - order["f1_ci_low"].to_numpy(float), order["f1_ci_high"].to_numpy(float) - f1])
        fig, ax = plt.subplots(figsize=(12.8, 5.8))
        ax.bar(x, f1, color=palette[:len(order)], edgecolor="white", linewidth=0.8, alpha=0.95)
        ax.errorbar(x, f1, yerr=f1err, fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=24, ha="right")
        lower = max(0.0, float(np.nanmin(f1)) - 0.08)
        ax.set_ylim(lower, 1.01)
        ax.set_ylabel("F1-score")
        ax.set_title(f"{prefix}: repeated arbitration with 95% confidence intervals")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        fig.tight_layout()
        fig.savefig(cfg.out_root / "figures" / f"{prefix}_publication_f1_ci.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Quantum-specific ablation: classical nonlinear, local quantum,
        # entangled quantum, and reliability-calibrated proposed circuit.
        wanted = ["V5_RFF_Control", "V6_QTG_NoEntangle", "V7_QTG_Entangled", "V8_Proposed_RC_QTG"]
        qrows = []
        for w in wanted:
            q = order[order["experiment"].astype(str).str.endswith(w)]
            if not q.empty:
                qrows.append(q.iloc[0])
        if qrows:
            qdf = pd.DataFrame(qrows)
            qnames = [_short_method_name(x) for x in qdf["experiment"]]
            qx = np.arange(len(qdf))
            fig, ax = plt.subplots(figsize=(9.6, 5.4))
            for metric, color, marker in [("f1", "#7B2CBF", "o"), ("roc_auc", "#2A9D8F", "s"), ("pr_auc", "#E76F51", "^")]:
                ax.plot(qx, qdf[metric].astype(float), marker=marker, markersize=8,
                        linewidth=2.4, color=color, label=metric.upper().replace("ROC_AUC", "ROC-AUC").replace("PR_AUC", "PR-AUC"))
            ax.set_xticks(qx); ax.set_xticklabels(qnames, rotation=15, ha="right")
            ax.set_ylim(max(0.0, float(np.nanmin(qdf[["f1","roc_auc","pr_auc"]].to_numpy())) - 0.08), 1.01)
            ax.set_ylabel("Score")
            ax.set_title(f"{prefix}: matched classical and quantum ablation")
            ax.grid(alpha=0.20, linestyle="--")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(cfg.out_root / "figures" / f"{prefix}_quantum_ablation.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

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
            f"{latex_ci(r.get('roc_auc'), r.get('roc_auc_ci_low'), r.get('roc_auc_ci_high'))} & "
            f"{latex_ci(r.get('brier'), r.get('brier_ci_low'), r.get('brier_ci_high'))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    (cfg.out_root / "tables" / f"{prefix}_corrected_arbitration_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # Compact 2x3 ablation layout using latest seed files.
    arb_files = sorted((cfg.out_root / "results").glob(f"{prefix}_arbitration_seed*.csv"))
    rob_files = sorted((cfg.out_root / "robustness").glob(f"{prefix}_robustness_seed*.csv"))
    faith_files = sorted((cfg.out_root / "faithfulness").glob(f"{prefix}_faithfulness_summary_seed*.csv"))
    if not arb_files:
        return
    arb = pd.read_csv(arb_files[-1])
    rob = pd.read_csv(rob_files[-1]) if rob_files else pd.DataFrame(columns=["model","condition","accuracy","f1","roc_auc"])
    faith = pd.read_csv(faith_files[-1]) if faith_files else pd.DataFrame(columns=["modality","mean_deletion_impact","mean_contribution","mean_sufficiency_score"])

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
        ab.append(f"{name} & {latex_num(r['accuracy'])} & {latex_num(r['precision'])} & {latex_num(r['recall'])} & {latex_num(r['f1'])} & {latex_num(r['roc_auc'])} \\\\")
    ab.append(r"\bottomrule\end{tabular}}")
    ab.append(r"\vspace{0.25cm}")
    ab.append(r"\caption*{(c) Robustness under corrupted evidence}")
    ab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{llccc}\toprule Model & Condition & Acc. & F1 & AUC \\\midrule")
    for _, r in rob.head(12).iterrows():
        ab.append(f"{r['model']} & {str(r['condition']).replace('_',' ')} & {latex_num(r['accuracy'])} & {latex_num(r['f1'])} & {latex_num(r['roc_auc'])} \\\\")
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
    channel_counts = evidence[[f"mask_{m}" for m in MODALITIES]].sum(axis=1)
    n_multi = int(np.sum(channel_counts >= 2))
    true_fusion = bool(prefix.startswith("INCIDENT") and n_multi > 0)
    save_json(
        {
            "prefix": prefix,
            "rows": len(evidence),
            "protocol_counts": evidence["protocol"].value_counts().to_dict() if "protocol" in evidence.columns else {},
            "observed_channel_counts": channel_counts.value_counts().sort_index().to_dict(),
            "rows_with_two_or_more_channels": n_multi,
            "multichannel_fraction": float(n_multi / max(len(evidence), 1)),
            "true_fusion_claim_allowed": true_fusion,
            "quantum_pair_interaction_claim_allowed": true_fusion,
            "strict_nested_outer_holdout": bool("outer_role" in evidence.columns),
        },
        cfg.out_root / "audit" / f"{prefix}_protocol_boundary.json",
    )
    all_runs = []
    seeds = [cfg.seed + i for i in range(cfg.repeats)]
    for seed in seeds:
        set_seed(seed)
        all_runs.append(train_arbiters_once(evidence, cfg, seed, prefix))
    aggregate_repeats(cfg, prefix)



def run_llm_email_oof(cfg: Config) -> Optional[pd.DataFrame]:
    """Leakage-controlled OOF DeBERTa email specialist with GPU-safe training.

    Key changes versus the earlier implementation:
      * tokenizes the whole corpus once instead of tokenizing inside __getitem__;
      * uses CUDA AMP on GPU and computes CrossEntropy in FP32;
      * uses configurable gradient accumulation; the current command can use physical batch 32 x 4 = effective 128;
      * clips gradients and caches each completed OOF fold;
      * resumes after interruption without retraining finished folds;
      * prints CUDA device/VRAM information at startup.
    """
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception as e:
        save_json({"status":"skipped", "reason":f"Install torch, transformers, sentencepiece: {e}"},
                  cfg.out_root/"llm"/"oof_status.json")
        return None

    llm_dir = cfg.out_root / "llm"
    cache_dir = cfg.out_root / "cache" / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    texts, y, meta = load_email(cfg)
    idx_all=np.arange(len(texts))
    if cfg.llm_max_rows and len(idx_all)>cfg.llm_max_rows:
        # Stratify on a joint label/domain code; train_test_split expects 1-D strata.
        joint = (y.astype(str) + "_" + meta["domain"].astype(str)).to_numpy()
        idx_all,_=train_test_split(idx_all, test_size=len(idx_all)-cfg.llm_max_rows,
                                    stratify=joint, random_state=cfg.seed)
        idx_all=np.sort(idx_all)
        texts=texts.iloc[idx_all].reset_index(drop=True)
        y=y.iloc[idx_all].reset_index(drop=True)
        meta=meta.iloc[idx_all].reset_index(drop=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props=torch.cuda.get_device_properties(0)
        print(f"[LLM] CUDA device: {torch.cuda.get_device_name(0)} | dedicated VRAM: {props.total_memory/1024**3:.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    else:
        msg = (
            "[LLM] ERROR: CUDA is unavailable in the active Python environment. "
            "DeBERTa OOF training on CPU is intentionally blocked because it can take many hours. "
            "Install a CUDA-enabled PyTorch build in this conda environment, then rerun. "
            "Use --allow_cpu_llm only if you explicitly want CPU training."
        )
        if not cfg.allow_cpu_llm:
            raise RuntimeError(msg)
        print(msg.replace("ERROR", "WARNING"))

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    tokenizer=AutoTokenizer.from_pretrained(cfg.llm_model, use_fast=True)

    # Tokenize once in large CPU batches. This removes the very slow tokenizer
    # call that previously occurred for every sample of every epoch.
    model_tag = hashlib.sha1(cfg.llm_model.encode("utf-8")).hexdigest()[:8]
    token_cache = cache_dir / f"email_tokens_{model_tag}_len{cfg.llm_max_length}.pt"
    if cfg.resume_cache and token_cache.exists():
        pack=torch.load(token_cache, map_location="cpu")
        input_ids=pack["input_ids"]
        attention_mask=pack["attention_mask"]
        if len(input_ids) != len(texts):
            token_cache.unlink(missing_ok=True)
            input_ids=attention_mask=None
    else:
        input_ids=attention_mask=None
    if input_ids is None:
        print(f"[LLM] tokenizing {len(texts):,} emails once (max_length={cfg.llm_max_length})")
        enc=tokenizer(texts.astype(str).tolist(), truncation=True, padding="max_length",
                      max_length=cfg.llm_max_length, return_tensors="pt")
        input_ids=enc["input_ids"].contiguous()
        attention_mask=enc["attention_mask"].contiguous()
        torch.save({"input_ids":input_ids,"attention_mask":attention_mask}, token_cache)
    labels_all=torch.as_tensor(y.to_numpy(), dtype=torch.long)

    train_signature = hashlib.sha1(
        f"{cfg.llm_model}|{cfg.llm_max_length}|{cfg.llm_epochs}|{cfg.llm_batch_size}|{cfg.llm_grad_accum}|{cfg.llm_lr}|{cfg.llm_oof_folds}".encode("utf-8")
    ).hexdigest()[:10]
    oof_path=cache_dir/f"deberta_oof_seed{cfg.seed}_{train_signature}.npy"
    done_path=cache_dir/f"deberta_oof_done_seed{cfg.seed}_{train_signature}.npy"
    if cfg.resume_cache and oof_path.exists() and done_path.exists():
        oof=np.load(oof_path)
        done=np.load(done_path).astype(bool)
        if len(oof)!=len(y) or len(done)!=len(y):
            oof=np.full(len(y),np.nan,dtype=float); done=np.zeros(len(y),dtype=bool)
    else:
        oof=np.full(len(y),np.nan,dtype=float); done=np.zeros(len(y),dtype=bool)

    skf=StratifiedKFold(n_splits=cfg.llm_oof_folds, shuffle=True, random_state=cfg.seed)
    fold_rows=[]
    amp_enabled=(device.type=="cuda")
    # Ada GPUs such as RTX 4070 Ti SUPER support BF16. BF16 is preferred
    # because it provides mixed-precision throughput without GradScaler and
    # avoids FP16-gradient unscale failures. Fall back to FP16+GradScaler
    # only when BF16 is unavailable.
    bf16_enabled = bool(amp_enabled and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if bf16_enabled else torch.float16
    use_grad_scaler = bool(amp_enabled and not bf16_enabled)
    if use_grad_scaler:
        try:
            scaler=torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler=torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler=None
    if amp_enabled:
        print(f"[LLM] mixed precision: {'BF16 (no GradScaler)' if bf16_enabled else 'FP16 + GradScaler'}")

    for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        if cfg.resume_cache and np.all(done[va]) and np.all(np.isfinite(oof[va])):
            print(f"[LLM] OOF fold {fold}/{cfg.llm_oof_folds}: cached")
            ps=oof[va]
            fm=binary_metrics(y.iloc[va].to_numpy(),(ps>=cfg.threshold).astype(int),ps)
            fm.update({"fold":fold,"n_train":len(tr),"n_valid":len(va),"cached":1})
            fold_rows.append(fm)
            continue

        print(f"[LLM] OOF fold {fold}/{cfg.llm_oof_folds}: physical batch={cfg.llm_batch_size}, "
              f"grad_accum={cfg.llm_grad_accum}, effective batch={cfg.llm_batch_size*cfg.llm_grad_accum}")
        model=AutoModelForSequenceClassification.from_pretrained(cfg.llm_model, num_labels=2)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.to(device)
        # Keep trainable parameters in FP32. Autocast controls compute dtype.
        # This prevents optimizers from receiving native FP16 parameters.
        model.float()
        if device.type=="cuda":
            model.gradient_checkpointing_enable()

        train_ds=TensorDataset(input_ids[tr], attention_mask[tr], labels_all[tr])
        val_ds=TensorDataset(input_ids[va], attention_mask[va], labels_all[va])
        loader_kwargs={"num_workers":cfg.llm_num_workers,"pin_memory":amp_enabled}
        train_dl=DataLoader(train_ds,batch_size=cfg.llm_batch_size,shuffle=True,**loader_kwargs)
        val_dl=DataLoader(val_ds,batch_size=max(cfg.llm_batch_size,32),shuffle=False,**loader_kwargs)

        opt=torch.optim.AdamW(model.parameters(),lr=cfg.llm_lr,weight_decay=0.01)
        counts=np.bincount(y.iloc[tr].to_numpy(),minlength=2)
        weights=torch.tensor([len(tr)/(2*max(1,c)) for c in counts],dtype=torch.float32,device=device)
        loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
        best_state=None; best_val=float("inf")

        for ep in range(cfg.llm_epochs):
            model.train(); losses=[]
            opt.zero_grad(set_to_none=True)
            pbar=tqdm(train_dl, desc=f"LLM OOF fold {fold}/{cfg.llm_oof_folds} epoch {ep+1}/{cfg.llm_epochs}")
            for step,(ids,mask,labels) in enumerate(pbar):
                ids=ids.to(device,non_blocking=True)
                mask=mask.to(device,non_blocking=True)
                labels=labels.to(device,non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits=model(input_ids=ids,attention_mask=mask).logits
                # Compute weighted CE in FP32 for stable class weighting.
                loss=loss_fn(logits.float(),labels.long())/max(1,cfg.llm_grad_accum)
                if use_grad_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if ((step+1)%max(1,cfg.llm_grad_accum)==0) or ((step+1)==len(train_dl)):
                    if use_grad_scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    if use_grad_scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                losses.append(float(loss.item())*max(1,cfg.llm_grad_accum))
                if step % 5 == 0 and device.type=="cuda":
                    used=torch.cuda.memory_allocated()/1024**3
                    reserved=torch.cuda.memory_reserved()/1024**3
                    pbar.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}",gpu=f"{used:.1f}/{reserved:.1f}GB")

            model.eval(); vloss=[]
            with torch.inference_mode():
                for ids,mask,labels in val_dl:
                    ids=ids.to(device,non_blocking=True); mask=mask.to(device,non_blocking=True)
                    labels=labels.to(device,non_blocking=True)
                    with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
                        logits=model(input_ids=ids,attention_mask=mask).logits
                    vloss.append(float(loss_fn(logits.float(),labels.long()).item()))
            mean_v=float(np.mean(vloss)) if vloss else float("inf")
            print(f"[LLM] fold {fold} epoch {ep+1}: train_loss={np.mean(losses):.4f}, val_loss={mean_v:.4f}")
            if mean_v<best_val:
                best_val=mean_v
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval(); ps=[]
        with torch.inference_mode():
            for ids,mask,labels in val_dl:
                ids=ids.to(device,non_blocking=True); mask=mask.to(device,non_blocking=True)
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
                    logits=model(input_ids=ids,attention_mask=mask).logits
                ps.extend(torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy().tolist())
        ps=np.asarray(ps,dtype=float)
        oof[va]=ps; done[va]=True
        np.save(oof_path,oof); np.save(done_path,done.astype(np.uint8))
        fold_m=binary_metrics(y.iloc[va].to_numpy(),(ps>=cfg.threshold).astype(int),ps)
        fold_m.update({"fold":fold,"n_train":len(tr),"n_valid":len(va),"cached":0})
        fold_rows.append(fold_m)
        save_csv(pd.DataFrame(fold_rows),llm_dir/"deberta_oof_fold_metrics.csv")
        del model,best_state,train_dl,val_dl,train_ds,val_ds,opt
        if device.type=="cuda":
            torch.cuda.empty_cache()

    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f"LLM OOF contains {int(np.sum(~np.isfinite(oof)))} missing predictions.")
    out=meta.copy()
    out["score_email_llm"]=oof
    out["pred_email_llm"]=(oof>=cfg.threshold).astype(int)
    save_csv(out,llm_dir/"deberta_email_oof_predictions.csv")
    save_csv(pd.DataFrame(fold_rows),llm_dir/"deberta_oof_fold_metrics.csv")
    save_json({
        "status":"completed","model":cfg.llm_model,"device":str(device),
        "physical_batch":cfg.llm_batch_size,"grad_accum":cfg.llm_grad_accum,
        "effective_batch":cfg.llm_batch_size*cfg.llm_grad_accum,
        "max_length":cfg.llm_max_length,"folds":cfg.llm_oof_folds,
        "epochs":cfg.llm_epochs,"rows":len(out), "cache_signature": train_signature
    },llm_dir/"oof_status.json")
    return out[["email_row_id","score_email_llm","pred_email_llm"]]

def _deberta_train_predict(cfg: Config, train_text: Sequence[str], train_y: Sequence[int],
                           eval_text: Sequence[str], seed: int, tag: str) -> np.ndarray:
    """Fine-tune DeBERTa on one source domain and score a disjoint target set."""
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception as exc:
        raise RuntimeError(f"DeBERTa domain-transfer dependencies unavailable: {exc}")

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not cfg.allow_cpu_llm:
        raise RuntimeError("CUDA is required for DeBERTa domain-transfer experiments. Use --allow_cpu_llm only intentionally.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, use_fast=True)
    tr_enc = tokenizer(list(map(str, train_text)), truncation=True, padding="max_length",
                       max_length=cfg.llm_max_length, return_tensors="pt")
    ev_enc = tokenizer(list(map(str, eval_text)), truncation=True, padding="max_length",
                       max_length=cfg.llm_max_length, return_tensors="pt")
    ytr = torch.as_tensor(np.asarray(train_y, dtype=int), dtype=torch.long)
    train_ds = TensorDataset(tr_enc["input_ids"], tr_enc["attention_mask"], ytr)
    eval_ds = TensorDataset(ev_enc["input_ids"], ev_enc["attention_mask"])
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=cfg.llm_batch_size, shuffle=True,
                          num_workers=cfg.llm_num_workers, pin_memory=pin)
    eval_dl = DataLoader(eval_ds, batch_size=max(cfg.llm_batch_size, 64), shuffle=False,
                         num_workers=cfg.llm_num_workers, pin_memory=pin)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.llm_model, num_labels=2)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device); model.float()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.llm_lr, weight_decay=0.01)
    counts = np.bincount(np.asarray(train_y, dtype=int), minlength=2)
    weights = torch.tensor([len(train_y) / (2 * max(1, c)) for c in counts], dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    amp = device.type == "cuda"
    bf16 = bool(amp and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    use_scaler = bool(amp and not bf16)
    scaler = None
    if use_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    accum = max(1, min(int(cfg.llm_grad_accum), 4))
    for ep in range(int(cfg.domain_transfer_epochs)):
        model.train(); opt.zero_grad(set_to_none=True); losses=[]
        pbar = tqdm(train_dl, desc=f"DeBERTa {tag} epoch {ep+1}/{cfg.domain_transfer_epochs}")
        for step, (ids, mask, labels) in enumerate(pbar):
            ids = ids.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), labels.long()) / accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if ((step + 1) % accum == 0) or ((step + 1) == len(train_dl)):
                if use_scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()) * accum)
            if step % 5 == 0:
                pbar.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}")

    model.eval(); scores=[]
    with torch.inference_mode():
        for ids, mask in eval_dl:
            ids = ids.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist())
    del model, opt, train_dl, eval_dl, train_ds, eval_ds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(scores, dtype=float)


def run_human_llm_generalization(cfg: Config) -> None:
    """Source-separated Human↔LLM transfer for classical, DeBERTa, and dual email models.

    The target domain is never used for fitting, early stopping, threshold tuning,
    or fusion-weight estimation. Dual weights are estimated on a held-out
    calibration subset drawn only from the source domain, then both specialists
    are refit on the complete source domain before scoring the target domain.
    """
    out_path = cfg.out_root / "results" / "human_llm_true_generalization.csv"
    if cfg.resume_cache and out_path.exists():
        try:
            cached = pd.read_csv(out_path)
            required_methods = {"classical", "deberta", "dual"}
            if len(cached) >= 6 and required_methods.issubset(set(cached.get("method", []))):
                print("[human/LLM generalization] complete v8 results: cached")
                return
        except Exception:
            pass

    X, y, meta = load_email(cfg)
    data = pd.DataFrame({"text": X.astype(str), "label": y.astype(int), "domain": meta["domain"].astype(str)})
    rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []

    for dindex, (train_domain, test_domain) in enumerate([("human", "llm"), ("llm", "human")]):
        train = data[data["domain"] == train_domain].reset_index(drop=True)
        test = data[data["domain"] == test_domain].reset_index(drop=True)
        if len(train) == 0 or len(test) == 0 or train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        direction = f"{train_domain}_to_{test_domain}"
        print(f"[human/LLM] {direction}: train={len(train):,}, test={len(test):,}")

        fit_idx, cal_idx = train_test_split(
            np.arange(len(train)), test_size=cfg.domain_transfer_val_fraction,
            stratify=train["label"].to_numpy(), random_state=cfg.seed + 900 + dindex,
        )
        fit = train.iloc[fit_idx].reset_index(drop=True)
        cal = train.iloc[cal_idx].reset_index(drop=True)

        # Calibration-only models. Target-domain rows are not touched here.
        c_cal_model = build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=cfg.seed + dindex)
        c_cal_model.fit(fit["text"], fit["label"])
        c_cal_score = get_score(c_cal_model, cal["text"])
        d_cal_score = _deberta_train_predict(cfg, fit["text"].tolist(), fit["label"].to_numpy(),
                                             cal["text"].tolist(), cfg.seed + 1000 + dindex, f"{direction}_cal")
        bc = float(brier_score_loss(cal["label"], np.clip(c_cal_score, 0, 1)))
        bd = float(brier_score_loss(cal["label"], np.clip(d_cal_score, 0, 1)))
        wc = 1.0 / max(bc, 1e-6); wd = 1.0 / max(bd, 1e-6)
        alpha_c = float(wc / (wc + wd)); alpha_d = float(wd / (wc + wd))

        # Refit on the full source domain only, then score the untouched target.
        c_model = build_text_model(cfg.max_text_features, cfg.svd_components, quantum=False, seed=cfg.seed + 50 + dindex)
        t0 = time.perf_counter(); c_model.fit(train["text"], train["label"]); c_train_time = time.perf_counter() - t0
        t0 = time.perf_counter(); c_score = get_score(c_model, test["text"]); c_pred_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        d_score = _deberta_train_predict(cfg, train["text"].tolist(), train["label"].to_numpy(),
                                         test["text"].tolist(), cfg.seed + 2000 + dindex, f"{direction}_full")
        d_train_predict_time = time.perf_counter() - t0
        dual_score = alpha_c * c_score + alpha_d * d_score

        for method, score, train_time, pred_time in [
            ("classical", c_score, c_train_time, c_pred_time),
            ("deberta", d_score, d_train_predict_time, float("nan")),
            ("dual", dual_score, c_train_time + d_train_predict_time, float("nan")),
        ]:
            pred = (np.asarray(score) >= cfg.threshold).astype(int)
            metrics = binary_metrics(test["label"].to_numpy(), pred, score)
            metrics.update({
                "experiment": f"train_{train_domain}_test_{test_domain}_{method}",
                "direction": direction, "method": method,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "train_time_sec": float(train_time), "predict_time_sec": float(pred_time),
                "target_domain_used_for_model_selection": False,
                "fusion_classical_weight": alpha_c if method == "dual" else float("nan"),
                "fusion_deberta_weight": alpha_d if method == "dual" else float("nan"),
            })
            rows.append(metrics)
            pred_rows.append(pd.DataFrame({
                "direction": direction, "method": method,
                "y_true": test["label"].to_numpy(), "y_score": np.asarray(score), "y_pred": pred,
            }))
        save_json({
            "direction": direction, "calibration_source_domain": train_domain,
            "target_domain": test_domain, "target_used_for_weighting": False,
            "calibration_rows": int(len(cal)), "fit_rows": int(len(fit)),
            "classical_brier_cal": bc, "deberta_brier_cal": bd,
            "classical_weight": alpha_c, "deberta_weight": alpha_d,
        }, cfg.out_root / "llm" / f"domain_transfer_{direction}_fusion_weights.json")

    result = pd.DataFrame(rows)
    save_csv(result, out_path)
    if pred_rows:
        save_csv(pd.concat(pred_rows, ignore_index=True), cfg.out_root / "predictions" / "human_llm_domain_transfer_predictions.csv")

    if not result.empty:
        directions = list(result["direction"].unique())
        methods = ["classical", "deberta", "dual"]
        colors = {"classical": "#4C78A8", "deberta": "#F58518", "dual": "#7B2CBF"}
        x = np.arange(len(directions)); width = 0.24
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for j, method in enumerate(methods):
            vals = [float(result[(result["direction"] == d) & (result["method"] == method)]["f1"].iloc[0])
                    if not result[(result["direction"] == d) & (result["method"] == method)].empty else np.nan for d in directions]
            ax.bar(x + (j - 1) * width, vals, width=width, label=method.capitalize(), color=colors[method], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([d.replace("_to_", " → ").title() for d in directions])
        ax.set_ylim(0, 1.02); ax.set_ylabel("F1-score")
        ax.set_title("Email authoring-domain transfer")
        ax.grid(axis="y", alpha=0.20, linestyle="--"); ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(cfg.out_root / "figures" / "human_llm_domain_transfer_v8.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def collect_all_results(cfg: Config) -> None:
    files: List[Path] = []
    for sub in ["results", "robustness", "faithfulness"]:
        files.extend(sorted((cfg.out_root / sub).glob("*.csv")))
    frames: List[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df.insert(0, "source_file", p.name)
            frames.append(df)
        except Exception as exc:
            save_json({"file": str(p), "error": str(exc)}, cfg.out_root / "logs" / f"collect_error_{p.stem}.json")
    if not frames:
        return
    long = pd.concat(frames, ignore_index=True, sort=False)
    save_csv(long, cfg.out_root / "tables" / "all_results_long.csv")
    try:
        with pd.ExcelWriter(cfg.out_root / "tables" / "all_results.xlsx") as writer:
            long.to_excel(writer, sheet_name="all_results_long", index=False)
            used = {"all_results_long"}
            for p in files[:30]:
                try:
                    base = p.stem[:31] or "sheet"
                    sheet = base
                    k = 1
                    while sheet in used:
                        suffix = f"_{k}"
                        sheet = (base[:31-len(suffix)] + suffix)
                        k += 1
                    used.add(sheet)
                    pd.read_csv(p).to_excel(writer, sheet_name=sheet, index=False)
                except Exception:
                    pass
    except Exception as exc:
        save_json({"excel_error": str(exc)}, cfg.out_root / "logs" / "excel_export_error.json")


def write_professor_response(cfg: Config) -> None:
    md = f"""# QTrustAgent-X v8 Experimental Audit

This report is generated by `qtrustagentx_v8_publication_quantum.py`.

| No. | Concern | v8 experimental correction | Status |
|---:|---|---|---|
| 1 | Genuine multi-channel fusion was unsupported | `INCIDENT` mode now performs strict manifest integrity checks, rejects unresolved/duplicated references and label conflicts, and evaluates only rows with at least two observed modalities. `DISJOINT` remains explicitly evidence-level. | Conditional on a genuine incident manifest. Code cannot create missing real-world correspondences. |
| 2 | Architecture/code mismatch | Four modality scores and four masks feed an explicit graph and four-qubit Qiskit circuits. Circuit QASM, depth, gate counts, and figures are saved. | Resolved in implementation; manuscript must follow v8. |
| 3 | Agentic claims exceeded implementation | Reports describe modular specialist arbitration and do not claim autonomous planning, memory, negotiation, or autonomous tool use. | Resolved by claim boundary. |
| 4 | Novelty was not isolated | Eight matched variants now include average, true majority, linear late fusion, TGA, RFF, QTG-no-entanglement, QTG-entangled, and RC-QTG. | Resolved experimentally. |
| 5 | Poisoning bypassed the trained arbiter | Fixed trained models are evaluated after score corruption and channel removal using identical perturbations per seed. | Resolved. |
| 6 | Explanation metrics were not faithfulness tests | RC-QTG receives deletion, single-modality sufficiency, comprehensiveness, and normalized perturbation-contribution diagnostics. | Resolved as sensitivity/faithfulness diagnostics, not causal explanation. |
| 7 | Human-vs-LLM test was not transfer | Human→LLM and LLM→Human now compare classical, DeBERTa, and source-calibrated dual specialists. Target-domain labels never set fusion weights. | Resolved. |
| 8 | Evaluation leakage / weak holdout | DISJOINT primary results use a strict outer holdout. Inner OOF specialist scores are created only within outer-train data; outer-test specialist scores come from models fit only on outer-train rows. Repeated seeds and CIs are saved. | Resolved for DISJOINT. INCIDENT still depends on the supplied manifest and should use campaign/time grouping when metadata exist. |
| 9 | Mathematical dimensions were unclear | Input is 8-D (4 scores + 4 masks); TGA is explicit; QTG readout is 20 observables; RC-QTG is 24-D including four reliability summaries. | Resolved. |
| 10 | Strong controls were missing | RFF is the matched classical nonlinear control; QTG-no-entanglement isolates circuit locality; QTG-entangled isolates pair interactions; paired bootstrap comparisons are saved. | Resolved. |
| 11 | Deployment claims were premature | Runtime, calibration, FPR/FNR, confusion counts, GPU information, and Qiskit execution metadata are recorded. | Report prototype measurements only. |

## Quantum claim boundary

The full experiment uses Qiskit's ideal Statevector circuit simulator. It is valid to state that the method uses **Qiskit quantum circuits executed in simulation**. It is not valid to claim IBM QPU execution or quantum advantage unless a separate hardware experiment is run and reported.

In `DISJOINT` mode each row has one active modality, so pair interactions are inactive. The no-entanglement and entangled QTG controls should therefore match. A claim about cross-modal quantum interaction requires `INCIDENT` results with two or more observed modalities per row.

## Primary output files

- `tables/DISJOINT_arbitration_summary_ci.csv`
- `tables/DISJOINT_paired_primary_summary.csv`
- `tables/DISJOINT_nested_specialists_summary_ci.csv`
- `results/human_llm_true_generalization.csv`
- `figures/DISJOINT_publication_f1_ci.png`
- `figures/DISJOINT_quantum_ablation.png`
- `figures/human_llm_domain_transfer_v8.png`
- `audit/quantum_circuits/*`
- `audit/quantum_pair_mechanism_summary.json` and `figures/quantum_pair_mechanism_audit.png`
- `audit/DISJOINT_strict_nested_protocol.json`

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
        ("2", "Architecture and implementation mismatch", "Implemented four-node modality trust graph and actual four-qubit Qiskit circuit feature map. Saved audit files.", "Rewrite equations to match released code."),
        ("3", "Weak agentic justification", "Specialist modules, arbitration, audit trail, and report generation are implemented. No autonomous planning is claimed.", "Use specialist-agent pipeline wording unless planning is added."),
        ("4", "Novelty components not validated", "Added eight matched variants: average, true majority, learned late fusion, TGA, RFF, QTG without entanglement, QTG with entanglement, and proposed RC-QTG.", "F1 is the predeclared primary endpoint; quantum claims require the matched controls."),
        ("5", "Poisoning did not test arbiter", "Poisoning and missing-channel tests now pass through trained arbiters.", "Report limited robustness only."),
        ("6", "Explanation was not faithfulness", "Added deletion, sufficiency, comprehensiveness, and per-instance contribution tests.", "Do not report uncomputed explanation precision, recall, F1, or AUC."),
        ("7", "Human vs. LLM was subgroup testing", "Added target-disjoint human-to-LLM and LLM-to-human transfer for classical, DeBERTa, and source-calibrated dual email specialists.", "Target labels never tune models or fusion weights."),
        ("8", "Protocol lacked rigor", "DISJOINT uses a strict outer holdout, inner OOF specialist training, repeated seeds, Student-t CIs, paired tests, calibration/error metrics, and source-integrity audits.", "Temporal/campaign separation still requires source metadata."),
        ("9", "Dimensional notation problems", "Graph input is explicitly four modality scores plus four masks. Trust encoder outputs graph-transformed features.", "Rewrite math around four-node score matrix."),
        ("10", "Comparison insufficient", "Added linear late fusion, TGA, matched RFF, local QTG, entangled QTG, and RC-QTG with identical held-out rows and paired inference.", "No quantum advantage claim is made from simulator results."),
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
Replace BERT, decoded-QR, learned FC semantic compression, and vague trust-graph claims unless those exact components are implemented elsewhere. The v8 code uses:
- TF-IDF/SVD text specialists;
- engineered URL features;
- QR image statistics or QR image specialist depending on model availability;
- strict outer-holdout DISJOINT evaluation with inner-OOF specialist scores;
- four-node trust graph over modality scores and masks;
- a matched RFF nonlinear control;
- matched Qiskit four-qubit local/no-entanglement and entangled circuit feature maps;
- reliability-calibrated RC-QTG as the proposed quantum-circuit arbiter.

## Results section
Split results into:
1. specialist performance;
2. eight matched arbitration/ablation variants with paired inference;
3. incident-level fusion if manifest exists;
4. robustness under corrupted and missing channels;
5. explanation deletion/sufficiency faithfulness;
6. human vs LLM true cross-domain evaluation;
7. deployment metrics.

## Claims to avoid
- Do not claim "industrial readiness"; claim "deployment prototype".
- Do not claim "true multi-channel fusion" without INCIDENT mode.
- Do not claim "faithful explanation" without deletion/sufficiency results.
- Report the Qiskit simulator, four-qubit circuit, encoding, and circuit depth; do not imply quantum-hardware execution.
- Do not call logistic regression majority voting.

"""
    (cfg.out_root / "reports" / "MANUSCRIPT_PATCH_NOTES.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--incident_manifest", type=Path, default=None)
    parser.add_argument("--mode", choices=["all", "disjoint", "evidence", "incident"], default="all", help="evidence is an alias for disjoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--qr_limit", type=int, default=20000)
    parser.add_argument("--max_text_features", type=int, default=25000)
    parser.add_argument("--svd_components", type=int, default=256)
    parser.add_argument("--quantum_layers", type=int, default=2)
    parser.add_argument("--reliability_power", type=float, default=1.0)
    parser.add_argument("--quantum_pair_strength", type=float, default=0.35)
    parser.add_argument("--llm_model", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--llm_oof_folds", type=int, default=5)
    parser.add_argument("--llm_epochs", type=int, default=3)
    parser.add_argument("--llm_batch_size", type=int, default=32, help="Physical GPU microbatch. Default 32 for RTX 4070 16GB.")
    parser.add_argument("--llm_grad_accum", type=int, default=4, help="Gradient accumulation. Publication default 32x4 gives effective batch 128.")
    parser.add_argument("--llm_max_length", type=int, default=256)
    parser.add_argument("--allow_cpu_llm", action="store_true", help="Allow extremely slow CPU transformer training. CUDA is required by default.")
    parser.add_argument("--llm_num_workers", type=int, default=0, help="Keep 0 on Windows unless tested.")
    parser.add_argument("--no_resume_cache", action="store_true", help="Disable stage/fold cache and recompute everything.")
    parser.add_argument("--sklearn_jobs", type=int, default=8)
    parser.add_argument("--llm_lr", type=float, default=2e-5)
    parser.add_argument("--llm_max_rows", type=int, default=0)
    parser.add_argument("--rff_dim", type=int, default=512)
    parser.add_argument("--rff_gamma", type=float, default=0.5)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--domain_transfer_epochs", type=int, default=3)
    parser.add_argument("--no_strict_nested_disjoint", action="store_true", help="Disable the strict two-level outer holdout for disjoint evidence. Not recommended for final paper results.")
    parser.add_argument("--artifacts_only", action="store_true", help="Do not retrain. Regenerate quantum circuit PNG/JPEG/PDF/SVG and paper figures from cached V8 models/results.")
    parser.add_argument("--threshold", type=float, default=0.50)
    import sys
    sys.argv = [a for a in sys.argv if str(a).strip()]
    args = parser.parse_args()
    if args.mode == "evidence":
        args.mode = "disjoint"

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
        quantum_layers=args.quantum_layers,
        reliability_power=args.reliability_power,
        quantum_pair_strength=args.quantum_pair_strength,
        llm_model=args.llm_model, llm_oof_folds=args.llm_oof_folds,
        llm_epochs=args.llm_epochs, llm_batch_size=args.llm_batch_size,
        llm_grad_accum=args.llm_grad_accum, llm_max_length=args.llm_max_length,
        llm_num_workers=args.llm_num_workers, allow_cpu_llm=args.allow_cpu_llm, resume_cache=not args.no_resume_cache,
        sklearn_jobs=args.sklearn_jobs, llm_lr=args.llm_lr, llm_max_rows=args.llm_max_rows,
        rff_dim=args.rff_dim, rff_gamma=args.rff_gamma, bootstrap_samples=args.bootstrap_samples,
        domain_transfer_epochs=args.domain_transfer_epochs,
        strict_nested_disjoint=not args.no_strict_nested_disjoint,
        artifacts_only=args.artifacts_only,
        threshold=args.threshold,
    )

    set_seed(cfg.seed)
    ensure_dirs(cfg)
    save_json(asdict(cfg), cfg.out_root / "config" / "run_config.json")
    run_dataset_integrity_audits(cfg)
    save_json({
        "qiskit_available": QISKIT_AVAILABLE,
        "qiskit_import_error": QISKIT_IMPORT_ERROR,
        "required_packages": ["qiskit>=2.1,<3", "transformers>=4.45", "sentencepiece", "torch"],
        "optional_packages": ["qiskit-aer>=0.17", "pylatexenc"],
        "quantum_qubits": 4,
        "quantum_layers": cfg.quantum_layers,
        "quantum_shots": 0,
        "reliability_power": cfg.reliability_power,
        "quantum_pair_strength": cfg.quantum_pair_strength,
        "execution": "Qiskit Statevector ideal circuit simulation",
        "qpu_execution": False,
        "rff_dim": cfg.rff_dim,
        "rff_gamma": cfg.rff_gamma,
        "strict_nested_disjoint": cfg.strict_nested_disjoint,
    }, cfg.out_root / "audit" / "qiskit_environment.json")

    start = time.perf_counter()

    if cfg.artifacts_only:
        if not QISKIT_AVAILABLE:
            raise RuntimeError(
                "Qiskit is required to regenerate circuit figures. Install qiskit and rerun --artifacts_only."
            )
        run_quantum_mechanism_audit(cfg)
        prefixes = []
        if cfg.mode in ["all", "disjoint"]:
            prefixes.append("DISJOINT")
        if cfg.mode in ["all", "incident"]:
            # Only regenerate INCIDENT artifacts when cached incident results/models exist.
            if list((cfg.out_root / "results").glob("INCIDENT_arbitration_seed*.csv")):
                prefixes.append("INCIDENT")
        if not prefixes:
            prefixes = ["DISJOINT"]
        artifact_summaries = []
        for prefix in prefixes:
            artifact_summaries.append(regenerate_quantum_artifacts_from_saved_models(cfg, prefix))
            regenerate_cached_figures(cfg, prefix)
        # Reuse already-computed human/LLM results and consolidated tables; no
        # transformer or specialist training is invoked in this mode.
        collect_all_results(cfg)
        write_professor_response(cfg)
        write_professor_status_latex(cfg)
        write_manuscript_patch_notes(cfg)
        final_audit = write_experiment_completeness_audit(cfg)
        save_json(
            {"runtime_seconds": round(time.perf_counter() - start, 3),
             "mode": "artifacts_only", "summaries": artifact_summaries,
             "final_status": final_audit.get("status")},
            cfg.out_root / "logs" / "artifact_regeneration_runtime.json",
        )
        print(f"Artifact regeneration completed. Outputs saved in: {cfg.out_root}")
        print(f"Final artifact status: {final_audit.get('status')}")
        return

    spec, models, meta = build_specialist_oof(cfg, cfg.seed)

    # build_specialist_oof() already runs and merges the leakage-controlled
    # DeBERTa OOF email specialist. Do not train/run it a second time here.
    if "score_email_llm" not in meta["email"].columns:
        llm_oof = run_llm_email_oof(cfg)
        if llm_oof is not None:
            meta["email"] = meta["email"].merge(llm_oof, on="email_row_id", how="left")

    if "score_email_llm" in meta["email"].columns:
        meta["email"]["score_email_llm"] = meta["email"]["score_email_llm"].fillna(0.5).astype(float)
        llm_metrics = binary_metrics(
            meta["email"]["label"].to_numpy(),
            (meta["email"]["score_email_llm"].to_numpy() >= cfg.threshold).astype(int),
            meta["email"]["score_email_llm"].to_numpy(),
        )
        llm_metrics["experiment"] = "EMAIL_DeBERTa_OOF_Specialist"
        llm_metrics["n"] = len(meta["email"])
        spec = pd.concat([spec, pd.DataFrame([llm_metrics])], ignore_index=True)
        if "score_email_fused" in meta["email"].columns:
            fused_score = meta["email"]["score_email_fused"].to_numpy(float)
            fused_metrics = binary_metrics(
                meta["email"]["label"].to_numpy(),
                (fused_score >= cfg.threshold).astype(int), fused_score,
            )
            fused_metrics["experiment"] = "EMAIL_Dual_CrossFitted_OOF_Secondary"
            fused_metrics["n"] = len(meta["email"])
            spec = pd.concat([spec, pd.DataFrame([fused_metrics])], ignore_index=True)

    save_csv(spec, cfg.out_root / "results" / "specialist_oof_primary.csv")

    if cfg.mode in ["all", "disjoint", "incident"] and not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit is required for the updated QTrustAgent-X quantum arbitration experiments. "
            "Install with: pip install qiskit>=1.2"
        )

    if QISKIT_AVAILABLE:
        run_quantum_mechanism_audit(cfg)

    if cfg.mode in ["all", "disjoint"]:
        if cfg.strict_nested_disjoint:
            run_nested_disjoint_pipeline(cfg)
        else:
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
            incident_all = incident.copy()
            incident = incident[incident["n_observed_channels"] >= 2].reset_index(drop=True)
            save_json({
                "manifest_rows": int(len(incident_all)),
                "rows_with_two_or_more_channels": int(len(incident)),
                "rows_excluded_as_single_channel": int((incident_all["n_observed_channels"] < 2).sum()),
                "channel_count_distribution": incident_all["n_observed_channels"].value_counts().sort_index().to_dict(),
                "true_fusion_rows_used": True
            }, cfg.out_root / "audit" / "incident_fusion_filter.json")
            if len(incident) == 0:
                raise RuntimeError("Incident manifest contains no cases with at least two observed modalities; true heterogeneous fusion cannot be evaluated.")
            run_pipeline(cfg, "INCIDENT", incident)

    run_human_llm_generalization(cfg)
    collect_all_results(cfg)
    write_professor_response(cfg)
    write_professor_status_latex(cfg)
    write_manuscript_patch_notes(cfg)
    write_experiment_completeness_audit(cfg)

    save_json({"runtime_seconds": round(time.perf_counter() - start, 3)}, cfg.out_root / "logs" / "runtime.json")
    print(f"Completed. Outputs saved in: {cfg.out_root}")
    print("Read reports/PROFESSOR_COMMENT_RESPONSE.md before revising the manuscript.")


if __name__ == "__main__":
    main()
