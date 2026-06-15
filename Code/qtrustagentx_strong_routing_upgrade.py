from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(r"D:\other\QTrustAgentX\QTrustAgentX_Results")
PRED = ROOT / "predictions"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"

RESULTS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

STRONG_AGENTS = {
    "URL_ExtraTrees_Strong": "URL",
    "SMS_TFIDF_SVM_Baseline": "SMS",
    "EMAIL_NoQuantum_TextOnly": "Email",
    "QR_ResNet18_Agent": "QR",
}

rows = []

for exp, modality in STRONG_AGENTS.items():
    path = PRED / f"{exp}_predictions.csv"
    if not path.exists():
        print(f"Missing: {path}")
        continue

    df = pd.read_csv(path)
    df["experiment"] = exp
    df["modality"] = modality

    if "y_score" not in df.columns:
        df["y_score"] = df["y_pred"]

    rows.append(df[["experiment", "modality", "y_true", "y_pred", "y_score"]])

all_pred = pd.concat(rows, ignore_index=True)

y_true = all_pred["y_true"].astype(int).values
y_pred = all_pred["y_pred"].astype(int).values
y_score = pd.to_numeric(all_pred["y_score"], errors="coerce").fillna(all_pred["y_pred"]).values

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

metrics = {
    "experiment": "QTrustAgentX_StrongRouting",
    "n_test": len(all_pred),
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "specificity": tn / max(tn + fp, 1),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "auc": roc_auc_score(y_true, y_score),
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
}

pd.DataFrame([metrics]).to_csv(
    RESULTS / "qtrustagentx_strong_routing_results.csv",
    index=False
)

all_pred.to_csv(
    PRED / "QTrustAgentX_StrongRouting_predictions.csv",
    index=False
)

latex = f"""
\\begin{{table}}[!t]
\\centering
\\caption{{Strong Modality-Routing Result of QTrustAgent-X}}
\\label{{tab:strong_routing}}
\\begin{{tabular}}{{lccccc}}
\\hline
Method & Acc. & Prec. & Rec. & F1 & AUC \\\\
\\hline
QTrustAgent-X Strong Routing & {metrics['accuracy']:.4f} & {metrics['precision']:.4f} & {metrics['recall']:.4f} & {metrics['f1']:.4f} & {metrics['auc']:.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

(TABLES / "qtrustagentx_strong_routing_table.tex").write_text(latex, encoding="utf-8")

print(pd.DataFrame([metrics]).T)
print("Saved strong routing results.")