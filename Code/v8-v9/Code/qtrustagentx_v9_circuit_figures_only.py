"""
QTrustAgent-X v9: Circuit / Architecture Figures Only
=====================================================

Purpose
-------
Generate publication-ready, same-size PNG architecture panels for every
ablation/control used in the combined V0--V9 component table, without fitting
or evaluating any model.

Scientific boundary
-------------------
V0--V5 are classical controls and therefore do NOT have quantum circuits.
This script renders them as classical architecture diagrams and labels them
accordingly. V6, V7, V8, V9A, V9B, and V9C contain quantum-circuit cores.
V9B and V9C intentionally reuse the V7 QTG-pair circuit; they differ only in
post-quantum appended features. The diagrams make this explicit.

The quantum diagrams use an all-channel DESIGN TEMPLATE so the reviewer can
see the pair-capable topology. This is not presented as an empirical DISJOINT
row. Under the reported DISJOINT protocol, empirical rows have one active
modality and pair gates are inactive.

No training, predictions, metrics, robustness tests, or specialist models are
run by this script.

Example
-------
python qtrustagentx_v9_circuit_figures_only.py ^
  --v8_root D:/other/QTrustAgentX/QTrustAgentX_Results_v8 ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Results_v9 ^
  --seed 42

To render seed-specific RC-QTG reliability angles for seeds 42--46:
python qtrustagentx_v9_circuit_figures_only.py ^
  --v8_root D:/other/QTrustAgentX/QTrustAgentX_Results_v8 ^
  --out_root D:/other/QTrustAgentX/QTrustAgentX_Results_v9 ^
  --seed 42 --repeats 5 --all_seeds

Requirements
------------
pip install numpy pandas matplotlib pillow qiskit pylatexenc
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageOps, ImageDraw, ImageFont

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
    QISKIT_IMPORT_ERROR = ""
except Exception as exc:
    QuantumCircuit = None
    QISKIT_AVAILABLE = False
    QISKIT_IMPORT_ERROR = str(exc)

MODALITIES = ["URL", "Email", "SMS", "QR"]
EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Fixed output geometry for EVERY individual panel.
CANVAS_W = 3200
CANVAS_H = 1600
DPI = 200

# Design-template evidence used only to visualize topology.
TEMPLATE_SCORES = np.array([0.82, 0.74, 0.66, 0.91], dtype=float)
TEMPLATE_MASKS = np.ones(4, dtype=float)
TEMPLATE_ROW = np.concatenate([TEMPLATE_SCORES, TEMPLATE_MASKS])


@dataclass
class Config:
    v8_root: Path
    out_root: Path
    seed: int = 42
    repeats: int = 5
    all_seeds: bool = False
    layers: int = 2
    pair_strength: float = 0.35
    width: int = CANVAS_W
    height: int = CANVAS_H


def ensure_dirs(cfg: Config) -> Tuple[Path, Path]:
    fig_dir = cfg.out_root / "figures" / "ablation_architectures"
    audit_dir = cfg.out_root / "audit" / "ablation_architectures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, audit_dir


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def score_angle(p: float) -> float:
    return float(2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0))))


def load_reliability(v8_root: Path, seed: int) -> np.ndarray:
    p = v8_root / "audit" / f"DISJOINT_RC_QTG_reliability_seed{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing v8 reliability file: {p}\n"
            "This figure-only script needs the saved training-derived reliability "
            "values so V8/V9A use the actual fitted seed-specific circuit angles."
        )
    df = pd.read_csv(p)
    needed = {"modality", "reliability"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Unexpected reliability schema in {p}. Need columns {sorted(needed)}")
    rel = []
    for mod in ["url", "email", "sms", "qr"]:
        q = df[df["modality"].astype(str).str.lower() == mod]
        if q.empty:
            raise ValueError(f"Reliability for modality '{mod}' is missing in {p}")
        rel.append(float(q.iloc[0]["reliability"]))
    return np.asarray(rel, dtype=float)


def build_qtg_circuit(row: np.ndarray, *, layers: int, pair_strength: float,
                      entangled: bool, reliability: Optional[np.ndarray] = None) -> QuantumCircuit:
    if not QISKIT_AVAILABLE:
        raise RuntimeError(f"Qiskit is required to draw quantum circuits: {QISKIT_IMPORT_ERROR}")

    row = np.asarray(row, dtype=float)
    scores = np.clip(row[:4], 0.0, 1.0)
    masks = np.clip(row[4:8], 0.0, 1.0)
    rel = None if reliability is None else np.asarray(reliability, dtype=float)

    qc = QuantumCircuit(4)
    for layer in range(max(1, int(layers))):
        layer_scale = 1.0 + 0.10 * layer
        for q in range(4):
            if masks[q] <= 0.5:
                continue
            if rel is None:
                theta = score_angle(scores[q]) * layer_scale
                phase = np.pi * (scores[q] - 0.5) * 0.5
            else:
                theta = score_angle(scores[q]) * math.sqrt(max(rel[q], 1e-8)) * layer_scale
                phase = np.pi * (scores[q] - 0.5) * rel[q]
            qc.ry(float(theta), q)
            qc.rz(float(phase), q)

        if entangled:
            for i, j in EDGE_PAIRS:
                if masks[i] <= 0.5 or masks[j] <= 0.5:
                    continue
                agreement = 1.0 - abs(scores[i] - scores[j])
                signed_agreement = 2.0 * agreement - 1.0
                pair_rel = 1.0 if rel is None else math.sqrt(rel[i] * rel[j])
                phi = float(np.pi * pair_strength * pair_rel * signed_agreement)
                qc.cx(i, j)
                qc.rz(phi, j)
                qc.cx(i, j)

        if layer + 1 < max(1, int(layers)):
            for q in range(4):
                if masks[q] <= 0.5:
                    continue
                scale = 1.0 if rel is None else rel[q]
                qc.rx(float(0.25 * score_angle(scores[q]) * scale), q)
    return qc


def render_qiskit_to_png(qc: QuantumCircuit, out_path: Path) -> str:
    """Render actual Qiskit circuit. Returns rendering method."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig = qc.draw(output="mpl", fold=120)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return "qiskit_mpl"
    except Exception:
        # Dependency-light fallback, still representing the exact Qiskit circuit.
        txt = str(qc.draw(output="text", fold=120))
        lines = txt.splitlines() or [txt]
        fig, ax = plt.subplots(figsize=(18, max(5, 0.35 * len(lines) + 2)))
        ax.axis("off")
        ax.text(0.01, 0.99, txt, va="top", ha="left", family="monospace", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return "text_fallback"


def _box(ax, xy, w, h, text, *, linewidth=1.8, fontsize=13, linestyle="-"):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fill=False, linewidth=linewidth, linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, linewidth=1.6):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=linewidth)
    ax.add_patch(arr)


def draw_classical_architecture(title: str, subtitle: str, stages: Sequence[str], out_path: Path,
                                *, footer: str = "Classical control: no quantum circuit") -> None:
    fig = plt.figure(figsize=(16, 8), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.93, title, ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.875, subtitle, ha="center", va="center", fontsize=13)

    n = len(stages)
    left = 0.045
    right = 0.955
    usable = right - left
    gap = 0.025
    bw = (usable - gap * (n - 1)) / n
    y = 0.43; bh = 0.22
    for i, text in enumerate(stages):
        x = left + i * (bw + gap)
        _box(ax, (x, y), bw, bh, text, fontsize=12.5)
        if i < n - 1:
            _arrow(ax, (x + bw, y + bh/2), (x + bw + gap, y + bh/2))

    ax.text(0.5, 0.13, footer, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.075,
            "All panels use the same fixed canvas; scores/masks follow the DISJOINT arbitration interface.",
            ha="center", va="center", fontsize=10.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def add_quantum_wrapper(circuit_png: Path, title: str, subtitle: str, pre_text: str,
                        post_text: str, out_path: Path, footer: str) -> None:
    """Place an actual Qiskit circuit on a fixed-size publication canvas."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(canvas)

    # Use broadly available system fonts when possible; PIL default otherwise.
    def font(size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
        for c in candidates:
            try:
                if Path(c).exists():
                    return ImageFont.truetype(c, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    f_title = font(58, True)
    f_sub = font(34, False)
    f_box = font(33, True)
    f_footer = font(28, False)

    def centered_text(y: int, text: str, fnt):
        bb = draw.textbbox((0, 0), text, font=fnt)
        x = (CANVAS_W - (bb[2] - bb[0])) // 2
        draw.text((x, y), text, fill="black", font=fnt)

    centered_text(55, title, f_title)
    centered_text(135, subtitle, f_sub)

    # Input block.
    box_y0, box_y1 = 235, 390
    draw.rounded_rectangle((80, box_y0, 640, box_y1), radius=28, outline="black", width=4)
    wrapped = textwrap.wrap(pre_text, width=26)
    ytxt = box_y0 + 30
    for line in wrapped:
        bb = draw.textbbox((0, 0), line, font=f_box)
        draw.text((360 - (bb[2]-bb[0])//2, ytxt), line, fill="black", font=f_box)
        ytxt += 43

    # Circuit area.
    im = Image.open(circuit_png).convert("RGB")
    target_w, target_h = 1740, 900
    im.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    circuit_canvas = Image.new("RGB", (target_w, target_h), "white")
    px = (target_w - im.width) // 2
    py = (target_h - im.height) // 2
    circuit_canvas.paste(im, (px, py))
    cx, cy = 720, 260
    canvas.paste(circuit_canvas, (cx, cy))
    draw.rounded_rectangle((cx, cy, cx + target_w, cy + target_h), radius=24, outline="black", width=3)

    # Post-processing block.
    draw.rounded_rectangle((2520, box_y0, 3120, 520), radius=28, outline="black", width=4)
    wrapped = textwrap.wrap(post_text, width=28)
    ytxt = box_y0 + 28
    for line in wrapped:
        bb = draw.textbbox((0, 0), line, font=f_box)
        draw.text((2820 - (bb[2]-bb[0])//2, ytxt), line, fill="black", font=f_box)
        ytxt += 43

    # Arrows.
    draw.line((640, 312, 720, 312), fill="black", width=6)
    draw.polygon([(720, 312), (690, 295), (690, 329)], fill="black")
    draw.line((2460, 390, 2520, 390), fill="black", width=6)
    draw.polygon([(2520, 390), (2490, 373), (2490, 407)], fill="black")

    centered_text(1240, footer, f_footer)
    centered_text(1300,
                  "All-channel design template shown to expose pair-capable topology; empirical DISJOINT rows activate one modality.",
                  f_footer)
    centered_text(1350,
                  "Expectation readout: 4 Z + 4 X + 6 ZZ + 6 XX = 20 quantum observables.",
                  f_footer)

    # Force exact final dimensions.
    canvas = ImageOps.fit(canvas, (CANVAS_W, CANVAS_H), method=Image.Resampling.LANCZOS)
    canvas.save(out_path, format="PNG", dpi=(DPI, DPI))


def render_variant_set(cfg: Config, seed: int, canonical_names: bool) -> Dict[str, str]:
    fig_dir, audit_dir = ensure_dirs(cfg)
    seed_dir = fig_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    reliability = load_reliability(cfg.v8_root, seed)
    suffix = "" if canonical_names else f"_seed{seed}"

    files: Dict[str, str] = {}

    def out(name: str) -> Path:
        p = seed_dir / f"DISJOINT_{name}{suffix}.png"
        files[name] = str(p)
        return p

    # V0--V5 are deliberately classical, not fake quantum circuits.
    draw_classical_architecture(
        "V0 Source-prior mask-only control",
        "Tests whether modality identity alone predicts the class",
        ["4 availability masks", "Standardize", "Class-balanced logistic regression", "Phishing probability"],
        out("V0_source_prior_mask_only"),
    )
    draw_classical_architecture(
        "V1 Masked average",
        "Fixed non-learned score aggregation",
        ["4 specialist scores + masks", "Keep observed sources", "Masked probability mean", "Threshold at 0.5"],
        out("V1_masked_average"),
    )
    draw_classical_architecture(
        "V2 True majority vote",
        "Hard-vote aggregation over observed specialists",
        ["4 specialist scores + masks", "Threshold each observed score", "Majority of hard votes", "Final class"],
        out("V2_true_majority_vote"),
    )
    draw_classical_architecture(
        "V3 Learned late fusion",
        "Linear learned arbiter on the common score-and-mask interface",
        ["4 scores + 4 masks", "Standardize", "Class-balanced logistic regression", "Phishing probability"],
        out("V3_learned_late_fusion"),
    )
    draw_classical_architecture(
        "V4 TrustGraphArbitration (TGA)",
        "Classical four-node modality graph with mask-gated edges",
        ["4 scores + 4 masks", "4-node trust graph", "Graph diagnostics / transformed scores", "Logistic arbiter"],
        out("V4_TGA"),
    )
    draw_classical_architecture(
        "V5 Random Fourier feature control",
        "Matched classical nonlinear representation",
        ["4 scores + 4 masks", "Standardize", "512-D RFF mapping", "Class-balanced logistic regression"],
        out("V5_RFF_control"),
    )

    # Quantum core renderings.
    tmp_dir = audit_dir / f"seed{seed}" / "raw_qiskit"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    qc_v6 = build_qtg_circuit(TEMPLATE_ROW, layers=cfg.layers, pair_strength=cfg.pair_strength,
                              entangled=False, reliability=None)
    raw_v6 = tmp_dir / "V6_raw.png"
    method_v6 = render_qiskit_to_png(qc_v6, raw_v6)
    add_quantum_wrapper(
        raw_v6,
        "V6 QTG local",
        "Local four-qubit score encoding; no pair entanglement",
        "scores + masks",
        "20 observables -> standardize -> logistic arbiter",
        out("V6_QTG_local"),
        "Quantum core: local RY/RZ encoding with RX data re-uploading; no pair gates."
    )

    qc_v7 = build_qtg_circuit(TEMPLATE_ROW, layers=cfg.layers, pair_strength=cfg.pair_strength,
                              entangled=True, reliability=None)
    raw_v7 = tmp_dir / "V7_raw.png"
    method_v7 = render_qiskit_to_png(qc_v7, raw_v7)
    add_quantum_wrapper(
        raw_v7,
        "V7 QTG pair",
        "Matched QTG circuit with mask-gated pair operations",
        "scores + masks",
        "20 observables -> standardize -> logistic arbiter",
        out("V7_QTG_pair"),
        "Quantum core: V6 local encoding + CX-RZ-CX pair interactions when both endpoint masks are active."
    )

    qc_rel = build_qtg_circuit(TEMPLATE_ROW, layers=cfg.layers, pair_strength=cfg.pair_strength,
                               entangled=True, reliability=reliability)
    raw_rel = tmp_dir / "V8_V9A_raw.png"
    method_rel = render_qiskit_to_png(qc_rel, raw_rel)

    rel_txt = ", ".join(f"{m}={r:.3f}" for m, r in zip(MODALITIES, reliability))

    add_quantum_wrapper(
        raw_rel,
        "V8 Full RC-QTG",
        f"Reliability-calibrated circuit, seed {seed}: {rel_txt}",
        "scores + masks + training reliability",
        "20 observables + h1,h2,h3,h4 -> logistic arbiter",
        out("V8_RC_QTG_full"),
        "Quantum core: reliability-scaled local rotations and reliability-scaled pair phases."
    )

    add_quantum_wrapper(
        raw_rel,
        "V9A Reliability-circuit only",
        f"Same reliability-scaled quantum core as V8, seed {seed}: {rel_txt}",
        "scores + masks + training reliability",
        "20 observables only -> logistic arbiter",
        out("V9A_reliability_circuit_only"),
        "Ablation removes every appended post-quantum reliability statistic."
    )

    # V9B/V9C reuse V7's circuit by design. The wrapper is what changes.
    add_quantum_wrapper(
        raw_v7,
        "V9B Post-quantum features only",
        "Uncalibrated V7 QTG-pair quantum core",
        "scores + masks",
        "20 observables + h1,h2,h3,h4 -> logistic arbiter",
        out("V9B_post_quantum_features_only"),
        "Circuit is exactly V7; only the appended post-quantum feature vector changes."
    )

    add_quantum_wrapper(
        raw_v7,
        "V9C Active-score skip only",
        "Uncalibrated V7 QTG-pair quantum core",
        "scores + masks",
        "20 observables + h1 -> logistic arbiter",
        out("V9C_active_score_skip_only"),
        "Circuit is exactly V7; under DISJOINT, h1 equals the active specialist probability."
    )

    # Export exact circuit text/QASM and a machine-readable manifest.
    circuit_records = {}
    for tag, qc, render_method in [
        ("V6_QTG_local", qc_v6, method_v6),
        ("V7_QTG_pair", qc_v7, method_v7),
        ("V8_RC_QTG_full", qc_rel, method_rel),
        ("V9A_reliability_circuit_only", qc_rel, method_rel),
        ("V9B_post_quantum_features_only", qc_v7, method_v7),
        ("V9C_active_score_skip_only", qc_v7, method_v7),
    ]:
        base = audit_dir / f"seed{seed}" / tag
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".txt").write_text(str(qc.draw(output="text", fold=120)), encoding="utf-8")
        qasm_saved = False
        try:
            from qiskit import qasm3
            base.with_suffix(".qasm").write_text(qasm3.dumps(qc), encoding="utf-8")
            qasm_saved = True
        except Exception:
            pass
        circuit_records[tag] = {
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "operations": {str(k): int(v) for k, v in qc.count_ops().items()},
            "render_method": render_method,
            "qasm_saved": qasm_saved,
        }

    manifest = {
        "seed": seed,
        "fixed_png_dimensions": [CANVAS_W, CANVAS_H],
        "design_template_scores": TEMPLATE_SCORES.tolist(),
        "design_template_masks": TEMPLATE_MASKS.tolist(),
        "reliability": {m.lower(): float(r) for m, r in zip(MODALITIES, reliability)},
        "scientific_boundary": (
            "V0-V5 are classical and have no quantum circuit. V9B/V9C reuse the V7 quantum core. "
            "Quantum panels use an all-channel design template to expose pair-capable topology; "
            "empirical DISJOINT rows activate one modality, so pair gates are inactive."
        ),
        "files": files,
        "circuits": circuit_records,
    }
    save_json(manifest, audit_dir / f"ablation_architecture_manifest_seed{seed}.json")

    # Canonical copies with stable filenames for LaTeX, using the first requested seed.
    if canonical_names:
        canonical_dir = fig_dir / "latex"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        for key, src_str in files.items():
            src = Path(src_str)
            dst = canonical_dir / f"DISJOINT_{key}.png"
            Image.open(src).save(dst, "PNG", dpi=(DPI, DPI))

    return files


def make_contact_sheet(cfg: Config) -> Path:
    """Create a 3x4 overview image from canonical fixed-size panels."""
    fig_dir, _ = ensure_dirs(cfg)
    ldir = fig_dir / "latex"
    names = [
        "V0_source_prior_mask_only", "V1_masked_average", "V2_true_majority_vote", "V3_learned_late_fusion",
        "V4_TGA", "V5_RFF_control", "V6_QTG_local", "V7_QTG_pair",
        "V8_RC_QTG_full", "V9A_reliability_circuit_only", "V9B_post_quantum_features_only", "V9C_active_score_skip_only",
    ]
    thumb_w, thumb_h = 800, 400
    sheet = Image.new("RGB", (4 * thumb_w, 3 * thumb_h), "white")
    for idx, name in enumerate(names):
        p = ldir / f"DISJOINT_{name}.png"
        if not p.exists():
            raise FileNotFoundError(f"Canonical panel missing: {p}")
        im = Image.open(p).convert("RGB")
        im = ImageOps.fit(im, (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        x = (idx % 4) * thumb_w
        y = (idx // 4) * thumb_h
        sheet.paste(im, (x, y))
    out = fig_dir / "DISJOINT_V0_to_V9_component_architectures_contact_sheet.png"
    sheet.save(out, "PNG", dpi=(DPI, DPI))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render only V0--V9 architecture/circuit figures; no experiments are rerun.")
    parser.add_argument("--v8_root", type=Path, required=True,
                        help="Completed v8 results root containing audit/DISJOINT_RC_QTG_reliability_seed*.csv")
    parser.add_argument("--out_root", type=Path, required=True,
                        help="Existing/new v9 output root. Only figure/audit files are written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--all_seeds", action="store_true",
                        help="Also render seed-specific V8/V9A reliability-angle panels for every repeated seed.")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pair_strength", type=float, default=0.35)
    args = parser.parse_args()

    cfg = Config(
        v8_root=args.v8_root,
        out_root=args.out_root,
        seed=args.seed,
        repeats=args.repeats,
        all_seeds=args.all_seeds,
        layers=args.layers,
        pair_strength=args.pair_strength,
    )
    ensure_dirs(cfg)

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit is required because V6--V9C figures contain actual Qiskit circuit cores. "
            f"Import error: {QISKIT_IMPORT_ERROR}"
        )

    seeds = [cfg.seed + i for i in range(cfg.repeats)] if cfg.all_seeds else [cfg.seed]
    summaries = []
    for i, seed in enumerate(seeds):
        print(f"Rendering architecture/circuit panels for seed {seed} ...")
        files = render_variant_set(cfg, seed, canonical_names=(i == 0))
        summaries.append({"seed": seed, "n_panels": len(files)})

    contact = make_contact_sheet(cfg)
    save_json(
        {
            "mode": "figures_only",
            "no_model_training": True,
            "no_prediction_generation": True,
            "seeds": seeds,
            "panels_per_seed": 12,
            "individual_png_dimensions": [CANVAS_W, CANVAS_H],
            "contact_sheet": str(contact),
            "summaries": summaries,
        },
        cfg.out_root / "audit" / "ablation_architectures" / "FIGURES_ONLY_RUN.json",
    )

    print("Completed. No experiments were rerun.")
    print(f"Individual panels: {cfg.out_root / 'figures' / 'ablation_architectures'}")
    print(f"Canonical LaTeX panels: {cfg.out_root / 'figures' / 'ablation_architectures' / 'latex'}")
    print(f"Contact sheet: {contact}")


if __name__ == "__main__":
    main()
