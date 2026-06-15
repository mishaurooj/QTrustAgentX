import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)

# ============================================================
# Paths
# ============================================================
ROOT = Path(r"D:\other\QTrustAgentX\QTrustAgentX_Results")
RESULTS_DIR = ROOT / "results"
PRED_DIR = ROOT / "predictions"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = TABLE_DIR / "extra_q1_baseline_comparisons.xlsx"
OUT_TEX = TABLE_DIR / "extra_q1_tables.tex"

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 900,
    "axes.linewidth": 0.9,
})

# ============================================================
# Helpers
# ============================================================
def read_all_results():
    files = list(RESULTS_DIR.glob("*_results.csv"))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if "experiment" in df.columns:
                df["source_file"] = p.name
                frames.append(df)
        except Exception:
            pass

    if not frames:
        raise FileNotFoundError(f"No *_results.csv files found in {RESULTS_DIR}")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["experiment"], keep="last")
    return out


def load_prediction(exp):
    p = PRED_DIR / f"{exp}_predictions.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        return None

    if "y_score" not in df.columns:
        df["y_score"] = np.nan

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")
    df = df.dropna(subset=["y_true", "y_pred"])
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    return df


def safe_auc(y, score):
    try:
        if pd.isna(score).all():
            return np.nan
        return roc_auc_score(y, score)
    except Exception:
        return np.nan


def safe_ap(y, score):
    try:
        if pd.isna(score).all():
            return np.nan
        return average_precision_score(y, score)
    except Exception:
        return np.nan


def safe_brier(y, score):
    try:
        if pd.isna(score).all():
            return np.nan
        score = np.clip(score, 0, 1)
        return brier_score_loss(y, score)
    except Exception:
        return np.nan


def metrics_from_pred(exp):
    df = load_prediction(exp)
    if df is None:
        return None

    y = df["y_true"].values
    yp = df["y_pred"].values
    ys = df["y_score"].values

    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()

    return {
        "experiment": exp,
        "n_test": len(df),
        "accuracy": accuracy_score(y, yp),
        "precision": precision_score(y, yp, zero_division=0),
        "recall": recall_score(y, yp, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y, yp, zero_division=0),
        "auc": safe_auc(y, ys),
        "pr_auc": safe_ap(y, ys),
        "brier": safe_brier(y, ys),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }


def bootstrap_ci(exp, metric="f1", n_boot=1000, seed=42):
    df = load_prediction(exp)
    if df is None:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    y = df["y_true"].values
    yp = df["y_pred"].values
    ys = df["y_score"].values
    n = len(df)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pp = yp[idx]
        ss = ys[idx]

        try:
            if metric == "accuracy":
                vals.append(accuracy_score(yy, pp))
            elif metric == "f1":
                vals.append(f1_score(yy, pp, zero_division=0))
            elif metric == "auc":
                vals.append(safe_auc(yy, ss))
            elif metric == "pr_auc":
                vals.append(safe_ap(yy, ss))
        except Exception:
            pass

    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        return np.nan, np.nan

    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def pct(x):
    if pd.isna(x):
        return "--"
    return f"{x:.4f}"


# ============================================================
# Experiment groups
# ============================================================
GROUPS = {
    "URL": {
        "proposed": "URL_QuantumGraph_Agent",
        "baselines": [
            "URL_RF_Baseline",
            "URL_ExtraTrees_Strong",
            "URL_GraphTrust_Agent",
            "URL_NoQuantum_GraphOnly",
            "URL_NoGraph_QuantumOnly",
        ],
    },
    "SMS": {
        "proposed": "SMS_QuantumText_Agent",
        "baselines": [
            "SMS_TFIDF_SVM_Baseline",
            "SMS_SemanticCompressed_Agent",
            "SMS_NoQuantum_TextOnly",
        ],
    },
    "Email": {
        "proposed": "EMAIL_QuantumText_Agent",
        "baselines": [
            "EMAIL_TFIDF_SVM_Baseline",
            "EMAIL_SemanticCompressed_Agent",
            "EMAIL_NoQuantum_TextOnly",
        ],
    },
    "QR": {
        "proposed": "QR_ResNet18_Agent",
        "baselines": [
            "QR_Handcrafted_RF",
            "QR_Handcrafted_Quantum",
        ],
    },
    "Agentic": {
        "proposed": "Agentic_QuantumArbitration_X",
        "baselines": [
            "Agentic_MajorityVote_Baseline",
            "Agentic_RiskArbitration_X",
        ],
    },
}

# ============================================================
# Main tables
# ============================================================
all_results = read_all_results()

metric_rows = []
for exp in all_results["experiment"].dropna().unique():
    m = metrics_from_pred(exp)
    if m is not None:
        metric_rows.append(m)

pred_metrics = pd.DataFrame(metric_rows)

if pred_metrics.empty:
    pred_metrics = all_results.copy()

# Add confidence intervals
ci_rows = []
for exp in pred_metrics["experiment"].dropna().unique():
    f1_l, f1_u = bootstrap_ci(exp, "f1")
    auc_l, auc_u = bootstrap_ci(exp, "auc")
    ci_rows.append({
        "experiment": exp,
        "f1_ci95": f"[{pct(f1_l)}, {pct(f1_u)}]",
        "auc_ci95": f"[{pct(auc_l)}, {pct(auc_u)}]",
    })

ci_df = pd.DataFrame(ci_rows)
pred_metrics = pred_metrics.merge(ci_df, on="experiment", how="left")

# ============================================================
# Strongest baseline comparison
# ============================================================
comparison_rows = []

for family, spec in GROUPS.items():
    proposed = spec["proposed"]
    candidates = [e for e in spec["baselines"] if e in pred_metrics["experiment"].values]

    prop_row = pred_metrics[pred_metrics["experiment"] == proposed]
    if prop_row.empty or not candidates:
        continue

    base_pool = pred_metrics[pred_metrics["experiment"].isin(candidates)].copy()
    base_pool = base_pool.sort_values(["f1", "auc", "accuracy"], ascending=False)
    best_base = base_pool.iloc[0]
    prop = prop_row.iloc[0]

    comparison_rows.append({
        "family": family,
        "best_baseline": best_base["experiment"],
        "proposed": proposed,
        "baseline_accuracy": best_base.get("accuracy", np.nan),
        "proposed_accuracy": prop.get("accuracy", np.nan),
        "delta_accuracy": prop.get("accuracy", np.nan) - best_base.get("accuracy", np.nan),
        "baseline_f1": best_base.get("f1", np.nan),
        "proposed_f1": prop.get("f1", np.nan),
        "delta_f1": prop.get("f1", np.nan) - best_base.get("f1", np.nan),
        "baseline_auc": best_base.get("auc", np.nan),
        "proposed_auc": prop.get("auc", np.nan),
        "delta_auc": prop.get("auc", np.nan) - best_base.get("auc", np.nan),
        "interpretation": (
            "Proposed improves over strongest baseline"
            if prop.get("f1", np.nan) > best_base.get("f1", np.nan)
            else "Strongest baseline remains higher on this isolated setting"
        )
    })

best_baseline_table = pd.DataFrame(comparison_rows)

# ============================================================
# Full ranked model table
# ============================================================
rank_cols = [
    "experiment", "n_test", "accuracy", "precision", "recall",
    "specificity", "f1", "auc", "pr_auc", "brier",
    "fp", "fn", "f1_ci95", "auc_ci95"
]
rank_table = pred_metrics[[c for c in rank_cols if c in pred_metrics.columns]].copy()
rank_table = rank_table.sort_values(["f1", "auc", "accuracy"], ascending=False)

# ============================================================
# Error profile table
# ============================================================
error_cols = ["experiment", "tp", "tn", "fp", "fn", "fpr", "fnr", "precision", "recall", "specificity"]
error_table = pred_metrics[[c for c in error_cols if c in pred_metrics.columns]].copy()
error_table = error_table.sort_values(["fnr", "fpr"], ascending=True)

# ============================================================
# Threshold sensitivity for probability models
# ============================================================
threshold_rows = []

for exp in pred_metrics["experiment"].dropna().unique():
    dfp = load_prediction(exp)
    if dfp is None or dfp["y_score"].isna().all():
        continue

    y = dfp["y_true"].values
    s = dfp["y_score"].values

    for tau in np.arange(0.10, 0.91, 0.05):
        yp = (s >= tau).astype(int)
        threshold_rows.append({
            "experiment": exp,
            "threshold": round(float(tau), 2),
            "accuracy": accuracy_score(y, yp),
            "precision": precision_score(y, yp, zero_division=0),
            "recall": recall_score(y, yp, zero_division=0),
            "f1": f1_score(y, yp, zero_division=0),
        })

threshold_table = pd.DataFrame(threshold_rows)

# ============================================================
# Save Excel
# ============================================================
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    rank_table.to_excel(writer, sheet_name="ranked_all_models", index=False)
    best_baseline_table.to_excel(writer, sheet_name="best_baseline_vs_proposed", index=False)
    error_table.to_excel(writer, sheet_name="error_profile", index=False)
    threshold_table.to_excel(writer, sheet_name="threshold_sensitivity", index=False)
    all_results.to_excel(writer, sheet_name="raw_result_files", index=False)

# ============================================================
# Figures
# ============================================================
def short_name(exp):
    rep = {
        "URL_RF_Baseline": "URL-RF",
        "URL_ExtraTrees_Strong": "URL-ET",
        "URL_GraphTrust_Agent": "URL-GT",
        "URL_QuantumGraph_Agent": "URL-QG",
        "URL_NoQuantum_GraphOnly": "URL-G",
        "URL_NoGraph_QuantumOnly": "URL-Q",
        "SMS_TFIDF_SVM_Baseline": "SMS-SVM",
        "SMS_SemanticCompressed_Agent": "SMS-SC",
        "SMS_QuantumText_Agent": "SMS-QT",
        "SMS_NoQuantum_TextOnly": "SMS-T",
        "EMAIL_TFIDF_SVM_Baseline": "E-SVM",
        "EMAIL_SemanticCompressed_Agent": "E-SC",
        "EMAIL_QuantumText_Agent": "E-QT",
        "EMAIL_NoQuantum_TextOnly": "E-T",
        "QR_ResNet18_Agent": "QR-R18",
        "QR_Handcrafted_RF": "QR-RF",
        "QR_Handcrafted_Quantum": "QR-HQ",
        "Agentic_MajorityVote_Baseline": "MV",
        "Agentic_RiskArbitration_X": "Risk-X",
        "Agentic_QuantumArbitration_X": "QArb-X",
    }
    return rep.get(exp, exp[:12])


# Figure 1: proposed vs strongest baseline
if not best_baseline_table.empty:
    families = best_baseline_table["family"].tolist()
    x = np.arange(len(families))
    width = 0.34

    plt.figure(figsize=(10, 4.6))
    plt.bar(x - width/2, best_baseline_table["baseline_f1"], width, label="Best baseline", edgecolor="black")
    plt.bar(x + width/2, best_baseline_table["proposed_f1"], width, label="Proposed", edgecolor="black")

    plt.xticks(x, families)
    plt.ylabel("F1-score")
    plt.ylim(0, 1.05)
    plt.title("Proposed Models vs Strongest Available Baselines")
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "extra_proposed_vs_best_baseline_f1.png", dpi=900, bbox_inches="tight")
    plt.savefig(FIG_DIR / "extra_proposed_vs_best_baseline_f1.pdf", bbox_inches="tight")
    plt.close()


# Figure 2: compact ranked model comparison
top_models = rank_table.head(16).copy()
plt.figure(figsize=(11, 5.2))
x = np.arange(len(top_models))
plt.bar(x, top_models["f1"], edgecolor="black")
plt.xticks(x, [short_name(e) for e in top_models["experiment"]], rotation=35, ha="right")
plt.ylabel("F1-score")
plt.ylim(0, 1.05)
plt.title("Ranked Model Comparison by F1-score")
plt.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "extra_ranked_model_comparison_f1.png", dpi=900, bbox_inches="tight")
plt.savefig(FIG_DIR / "extra_ranked_model_comparison_f1.pdf", bbox_inches="tight")
plt.close()


# Figure 3: ROC overlays by family
for family, spec in GROUPS.items():
    exps = [spec["proposed"]] + spec["baselines"]
    exps = [e for e in exps if load_prediction(e) is not None]

    plt.figure(figsize=(5.5, 4.8))
    plotted = False

    for exp in exps:
        dfp = load_prediction(exp)
        if dfp is None or dfp["y_score"].isna().all():
            continue

        y = dfp["y_true"].values
        s = dfp["y_score"].values

        try:
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            plt.plot(fpr, tpr, linewidth=1.8, label=f"{short_name(exp)} AUC={auc:.3f}")
            plotted = True
        except Exception:
            pass

    if plotted:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{family} ROC Comparison")
        plt.legend(frameon=True, fontsize=8)
        plt.grid(linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"extra_{family.lower()}_roc_overlay.png", dpi=900, bbox_inches="tight")
        plt.savefig(FIG_DIR / f"extra_{family.lower()}_roc_overlay.pdf", bbox_inches="tight")
    plt.close()


# Figure 4: PR overlays by family
for family, spec in GROUPS.items():
    exps = [spec["proposed"]] + spec["baselines"]
    exps = [e for e in exps if load_prediction(e) is not None]

    plt.figure(figsize=(5.5, 4.8))
    plotted = False

    for exp in exps:
        dfp = load_prediction(exp)
        if dfp is None or dfp["y_score"].isna().all():
            continue

        y = dfp["y_true"].values
        s = dfp["y_score"].values

        try:
            precision, recall, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            plt.plot(recall, precision, linewidth=1.8, label=f"{short_name(exp)} AP={ap:.3f}")
            plotted = True
        except Exception:
            pass

    if plotted:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{family} Precision-Recall Comparison")
        plt.legend(frameon=True, fontsize=8)
        plt.grid(linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"extra_{family.lower()}_pr_overlay.png", dpi=900, bbox_inches="tight")
        plt.savefig(FIG_DIR / f"extra_{family.lower()}_pr_overlay.pdf", bbox_inches="tight")
    plt.close()


# Figure 5: threshold sensitivity for proposed agentic model
agentic_exp = "Agentic_QuantumArbitration_X"
ts = threshold_table[threshold_table["experiment"] == agentic_exp].copy()

if not ts.empty:
    plt.figure(figsize=(7.2, 4.6))
    for metric in ["precision", "recall", "f1", "accuracy"]:
        plt.plot(ts["threshold"], ts[metric], marker="o", linewidth=1.8, label=metric.capitalize())

    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Threshold Sensitivity of Agentic Quantum Arbitration")
    plt.grid(linestyle=":", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "extra_agentic_threshold_sensitivity.png", dpi=900, bbox_inches="tight")
    plt.savefig(FIG_DIR / "extra_agentic_threshold_sensitivity.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# LaTeX tables
# ============================================================
def latex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def make_latex_best_baseline(df):
    if df.empty:
        return ""

    lines = []
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{Proposed Models Compared with Strongest Available Baselines}")
    lines.append("\\label{tab:extra_best_baseline}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{llcccccc}")
    lines.append("\\hline")
    lines.append("Family & Best Baseline & Base F1 & Prop. F1 & $\\Delta$F1 & Base AUC & Prop. AUC & $\\Delta$AUC \\\\")
    lines.append("\\hline")

    for _, r in df.iterrows():
        lines.append(
            f"{latex_escape(r['family'])} & {latex_escape(short_name(r['best_baseline']))} & "
            f"{pct(r['baseline_f1'])} & {pct(r['proposed_f1'])} & {pct(r['delta_f1'])} & "
            f"{pct(r['baseline_auc'])} & {pct(r['proposed_auc'])} & {pct(r['delta_auc'])} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def make_latex_ranked(df, n=12):
    use = df.head(n).copy()

    lines = []
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{Ranked Model Comparison with Error Profile and Confidence Intervals}")
    lines.append("\\label{tab:extra_ranked_models}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{lccccccccc}")
    lines.append("\\hline")
    lines.append("Model & Acc. & Prec. & Rec. & Spec. & F1 & AUC & PR-AUC & FP & FN \\\\")
    lines.append("\\hline")

    for _, r in use.iterrows():
        lines.append(
            f"{latex_escape(short_name(r['experiment']))} & "
            f"{pct(r.get('accuracy', np.nan))} & {pct(r.get('precision', np.nan))} & "
            f"{pct(r.get('recall', np.nan))} & {pct(r.get('specificity', np.nan))} & "
            f"{pct(r.get('f1', np.nan))} & {pct(r.get('auc', np.nan))} & "
            f"{pct(r.get('pr_auc', np.nan))} & {int(r.get('fp', 0)) if not pd.isna(r.get('fp', np.nan)) else '--'} & "
            f"{int(r.get('fn', 0)) if not pd.isna(r.get('fn', np.nan)) else '--'} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("% Auto-generated extra Q1 comparison tables. Do not edit values manually.\n\n")
    f.write(make_latex_best_baseline(best_baseline_table))
    f.write("\n\n")
    f.write(make_latex_ranked(rank_table, n=12))

print("DONE")
print(f"Excel saved: {OUT_XLSX}")
print(f"LaTeX saved: {OUT_TEX}")
print(f"Figures saved in: {FIG_DIR}")