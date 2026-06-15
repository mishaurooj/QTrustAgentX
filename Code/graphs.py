import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# =========================
# Paths
# =========================
csv_path = Path(r"D:\other\QTrustAgentX\QTrustAgentX_Results\explanations\agent_evidence_scores.csv")
out_dir = Path(r"C:\Users\muroo\figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# =========================
# Basic cleanup
# =========================
df["modality"] = df["modality"].astype(str).str.lower()
df["agent"] = df["agent"].astype(str)
df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
df["agent_pred"] = pd.to_numeric(df["agent_pred"], errors="coerce")
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["risk_score", "agent_pred", "label", "modality", "agent"])

modality_order = ["url", "email", "sms", "qr"]
df = df[df["modality"].isin(modality_order)]

modality_labels = {"url": "URL", "email": "Email", "sms": "SMS", "qr": "QR"}

colors = {
    "url": "#2F6DB5",
    "email": "#D94B3D",
    "sms": "#4BA64F",
    "qr": "#7A4CC2",
    "benign": "#8FBCE6",
    "phishing": "#F28B82",
}

agent_abbrev = {
    "EMAIL_QuantumText_Agent": "E-QT",
    "QR_ResNet18_Agent": "QR-R18",
    "QR_Handcrafted_Quantum": "QR-HQ",
    "SMS_QuantumText_Agent": "S-QT",
    "URL_QuantumGraph_Agent": "U-QG",
}

def short_agent_name(name):
    return agent_abbrev.get(name, name.replace("_Agent", "").replace("_", "-"))

# =========================
# IEEE compact style
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 900,
    "axes.linewidth": 0.9,
})

fig, axes = plt.subplots(
    1, 3,
    figsize=(15.2, 4.2),
    gridspec_kw={"width_ratios": [1.0, 0.95, 1.05]},
    constrained_layout=False
)

plt.subplots_adjust(
    left=0.045,
    right=0.985,
    top=0.88,
    bottom=0.17,
    wspace=0.16
)

# ==========================================================
# (a) Risk-score separation
# ==========================================================
ax = axes[0]

positions, data_to_plot, box_colors, xlabels = [], [], [], []
pos = 1.0

for m in modality_order:
    benign_scores = df[(df["modality"] == m) & (df["label"] == 0)]["risk_score"].values
    phishing_scores = df[(df["modality"] == m) & (df["label"] == 1)]["risk_score"].values

    data_to_plot.extend([benign_scores, phishing_scores])
    positions.extend([pos, pos + 0.26])
    box_colors.extend([colors["benign"], colors["phishing"]])
    xlabels.append(pos + 0.13)
    pos += 0.88

bp = ax.boxplot(
    data_to_plot,
    positions=positions,
    widths=0.21,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.1),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8)
)

for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.90)
    patch.set_edgecolor("black")

ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
ax.set_title("(a) Risk-score separation", fontweight="bold", pad=4)
ax.set_ylabel("Risk score")
ax.set_ylim(-0.03, 1.03)
ax.set_xticks(xlabels)
ax.set_xticklabels([modality_labels[m] for m in modality_order])
ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.legend(
    handles=[
        Patch(facecolor=colors["benign"], edgecolor="black", label="Benign"),
        Patch(facecolor=colors["phishing"], edgecolor="black", label="Phishing"),
    ],
    loc="upper left",
    frameon=True,
    borderpad=0.35,
    handlelength=1.6
)

# ==========================================================
# (b) Agent reliability
# ==========================================================
ax = axes[1]

summary = (
    df.groupby(["modality", "agent"], group_keys=False)
    .apply(lambda x: pd.Series({
        "accuracy": (x["agent_pred"] == x["label"]).mean()
    }))
    .reset_index()
)

summary["modality_order"] = summary["modality"].map({m: i for i, m in enumerate(modality_order)})
summary = summary.sort_values(["modality_order", "accuracy"], ascending=[True, False]).reset_index(drop=True)
summary["agent_short"] = summary["agent"].map(short_agent_name)

x = np.arange(len(summary))
bar_colors = [colors.get(m, "#777777") for m in summary["modality"]]

ax.bar(
    x,
    summary["accuracy"],
    color=bar_colors,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.92,
    width=0.68
)

ax.set_title("(b) Agent reliability", fontweight="bold", pad=4)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(summary["agent_short"], rotation=0, ha="center")
ax.grid(axis="y", linestyle=":", alpha=0.35)

for i, row in summary.iterrows():
    ax.text(
        i,
        row["accuracy"] + 0.02,
        f"{row['accuracy']:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

legend_handles = [
    Patch(facecolor=colors[m], edgecolor="black", label=modality_labels[m])
    for m in modality_order if m in summary["modality"].unique()
]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.03),
    frameon=True,
    ncol=2,
    borderpad=0.30,
    handlelength=1.3,
    columnspacing=0.8
)

# ==========================================================
# (c) Evidence agreement
# ==========================================================
ax = axes[2]

df_agree = df.copy()
df_agree["local_id"] = df_agree.groupby("agent").cumcount()

pivot = df_agree.pivot_table(
    index="local_id",
    columns="agent",
    values="risk_score",
    aggfunc="mean"
)

desired_agents = [
    "URL_QuantumGraph_Agent",
    "EMAIL_QuantumText_Agent",
    "SMS_QuantumText_Agent",
    "QR_ResNet18_Agent",
    "QR_Handcrafted_Quantum",
]
desired_agents = [a for a in desired_agents if a in pivot.columns]
pivot = pivot[desired_agents]

corr = pivot.corr(method="pearson")
agent_names = [short_agent_name(c) for c in corr.columns]

im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn")

ax.set_title("(c) Evidence agreement", fontweight="bold", pad=4)
ax.set_xticks(np.arange(len(agent_names)))
ax.set_yticks(np.arange(len(agent_names)))
ax.set_xticklabels(agent_names, rotation=0)
ax.set_yticklabels(agent_names)

for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        val = corr.iloc[i, j]
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="black",
            fontweight="bold"
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.015)
cbar.set_label("Correlation", labelpad=4)
cbar.ax.tick_params(labelsize=9)

# =========================
# Save
# =========================
png_path = out_dir / "qtrustagentx_agent_evidence_analysis_1x3_compact.png"
pdf_path = out_dir / "qtrustagentx_agent_evidence_analysis_1x3_compact.pdf"

plt.savefig(png_path, dpi=900, bbox_inches="tight", pad_inches=0.025)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
plt.close()

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")