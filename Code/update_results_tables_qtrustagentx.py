from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

ROOT = Path(r"D:\other\QTrustAgentX\QTrustAgentX_Results")
RESULTS_DIR = ROOT / "results"
PRED_DIR = ROOT / "predictions"
TABLE_DIR = ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_TEX = TABLE_DIR / "updated_qtrustagentx_results_table.tex"
OUT_CSV = TABLE_DIR / "updated_qtrustagentx_results_table.csv"


def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.4f}"


def load_result_times():
    rows = []
    for p in RESULTS_DIR.glob("*_results.csv"):
        try:
            df = pd.read_csv(p)
            if "experiment" in df.columns:
                rows.append(df)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    return df.drop_duplicates("experiment", keep="last")


def load_pred_metrics(exp):
    p = PRED_DIR / f"{exp}_predictions.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        return None

    if "y_score" not in df.columns:
        df["y_score"] = df["y_pred"]

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce").fillna(df["y_pred"])
    df = df.dropna(subset=["y_true", "y_pred"])

    y_true = df["y_true"].astype(int).values
    y_pred = df["y_pred"].astype(int).values
    y_score = df["y_score"].astype(float).values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = np.nan

    try:
        brier = brier_score_loss(y_true, np.clip(y_score, 0, 1))
    except Exception:
        brier = np.nan

    return {
        "experiment": exp,
        "n_test": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


TABLE_ROWS = [
    ("(a) Strong Deployable Modality Specialists", [
        ("URL-ET", "URL_ExtraTrees_Strong"),
        ("SMS-SVM", "SMS_TFIDF_SVM_Baseline"),
        ("E-T", "EMAIL_NoQuantum_TextOnly"),
        ("QR-R18", "QR_ResNet18_Agent"),
    ]),
    ("(b) Agentic Arbitration", [
        ("Majority", "Agentic_MajorityVote_Baseline"),
        ("Risk-X", "Agentic_RiskArbitration_X"),
        ("QArb-X", "Agentic_QuantumArbitration_X"),
    ]),
    ("(c) Quantum and Graph Ablation", [
        ("URL-Q", "URL_NoGraph_QuantumOnly"),
        ("URL-G", "URL_NoQuantum_GraphOnly"),
        ("URL-QG", "URL_QuantumGraph_Agent"),
        ("SMS-QT", "SMS_QuantumText_Agent"),
        ("E-QT", "EMAIL_QuantumText_Agent"),
        ("QR-HQ", "QR_Handcrafted_Quantum"),
    ]),
]

times = load_result_times()

all_rows = []
for section, items in TABLE_ROWS:
    for short, exp in items:
        m = load_pred_metrics(exp)
        if m is None:
            continue

        t = times[times["experiment"] == exp]
        if len(t):
            m["train_time_sec"] = t.iloc[0].get("train_time_sec", np.nan)
            m["predict_time_sec"] = t.iloc[0].get("predict_time_sec", np.nan)
        else:
            m["train_time_sec"] = np.nan
            m["predict_time_sec"] = np.nan

        m["section"] = section
        m["model"] = short
        all_rows.append(m)

out = pd.DataFrame(all_rows)
out.to_csv(OUT_CSV, index=False)


latex = []
latex.append(r"\begin{table*}[!t]")
latex.append(r"\centering")
latex.append(r"\tiny")
latex.append(r"\caption{Updated QTrustAgent-X Results with Deployment-Oriented Phishing Detection Metrics. Acc.: Accuracy, Prec.: Precision, Rec.: Recall, Spec.: Specificity, PR-AUC: Precision-Recall AUC, FPR: False Positive Rate, FNR: False Negative Rate.}")
latex.append(r"\label{tab:qtrustagentx_updated_results}")
latex.append(r"\begin{threeparttable}")
latex.append(r"\resizebox{\textwidth}{!}{")
latex.append(r"\begin{tabular}{lcccccccccccc}")
latex.append(r"\toprule")
latex.append(r"\textbf{Model} & \textbf{N} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{Spec.} & \textbf{F1} & \textbf{AUC} & \textbf{PR-AUC} & \textbf{Brier} & \textbf{FPR} & \textbf{FNR} & \textbf{FP/FN} \\")
latex.append(r"\midrule")

for section, g in out.groupby("section", sort=False):
    latex.append(rf"\multicolumn{{13}}{{c}}{{\textbf{{{section}}}}} \\")
    latex.append(r"\midrule")

    for _, r in g.iterrows():
        latex.append(
            f"{r['model']} & "
            f"{int(r['n_test'])} & "
            f"{fmt(r['accuracy'])} & "
            f"{fmt(r['precision'])} & "
            f"{fmt(r['recall'])} & "
            f"{fmt(r['specificity'])} & "
            f"{fmt(r['f1'])} & "
            f"{fmt(r['auc'])} & "
            f"{fmt(r['pr_auc'])} & "
            f"{fmt(r['brier'])} & "
            f"{fmt(r['fpr'])} & "
            f"{fmt(r['fnr'])} & "
            f"{int(r['fp'])}/{int(r['fn'])} \\\\"
        )

    latex.append(r"\midrule")

latex[-1] = r"\bottomrule"
latex.append(r"\end{tabular}}")
latex.append(r"\begin{tablenotes}")
latex.append(r"\footnotesize")
latex.append(r"\item URL-ET: URL ExtraTrees; SMS-SVM: SMS TF-IDF SVM; E-T: Email Text-Only; QR-R18: QR ResNet18; QArb-X: Quantum Arbitration; Risk-X: Risk Arbitration; QG: Quantum-Graph; QT: Quantum-Text; HQ: Handcrafted Quantum.")
latex.append(r"\item Lower Brier, FPR, and FNR values indicate better calibrated and safer phishing detection behavior. These metrics should be interpreted with F1, recall, and specificity rather than accuracy alone.")
latex.append(r"\end{tablenotes}")
latex.append(r"\end{threeparttable}")
latex.append(r"\vspace{-0.4cm}")
latex.append(r"\end{table*}")

OUT_TEX.write_text("\n".join(latex), encoding="utf-8")

print("Saved CSV:", OUT_CSV)
print("Saved LaTeX:", OUT_TEX)