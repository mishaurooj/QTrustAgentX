"""
QTrustAgentX Industrial Streamlit App
Single-file app for trained QTrustAgentX models.

Run:
    conda activate qtrustagentx
    streamlit run qtrustagentx_app.py
"""

from pathlib import Path
import json, math, re, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from PIL import Image
except Exception:
    Image = None

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_DIR = Path(r"D:\other\QTrustAgentX\QTrustAgentX_Results\models")

st.set_page_config(
    page_title="QTrustAgentX Industrial Phishing Inspector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1rem;}
.big-title {font-size: 34px; font-weight: 800; color: #0f172a;}
.subtitle {font-size: 16px; color: #475569; margin-bottom: 18px;}
.card {background: white; padding: 18px; border-radius: 16px; border: 1px solid #e5e7eb; box-shadow: 0 4px 16px rgba(15,23,42,0.06);}
.safe {background:#ecfdf5; border-left:6px solid #16a34a; padding:12px; border-radius:10px;}
.warn {background:#fffbeb; border-left:6px solid #d97706; padding:12px; border-radius:10px;}
.risk {background:#fef2f2; border-left:6px solid #dc2626; padding:12px; border-radius:10px;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Utility
# -----------------------------
def clamp01(x):
    try:
        return float(max(0.0, min(1.0, float(x))))
    except Exception:
        return 0.5


def risk_band(score):
    score = clamp01(score)
    if score >= 0.85:
        return "Critical", "#dc2626", "risk"
    if score >= 0.65:
        return "High", "#ea580c", "risk"
    if score >= 0.45:
        return "Uncertain", "#d97706", "warn"
    return "Low", "#16a34a", "safe"


def load_model(model_dir: Path, filename: str):
    path = model_dir / filename
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_models(model_dir_str: str):
    d = Path(model_dir_str)
    return {
        "url_qg": load_model(d, "URL_QuantumGraph_Agent.joblib"),
        "url_strong": load_model(d, "URL_ExtraTrees_Strong.joblib"),
        "email_qt": load_model(d, "EMAIL_QuantumText_Agent.joblib"),
        "email_strong": load_model(d, "EMAIL_NoQuantum_TextOnly.joblib"),
        "sms_qt": load_model(d, "SMS_QuantumText_Agent.joblib"),
        "sms_strong": load_model(d, "SMS_TFIDF_SVM_Baseline.joblib"),
        "qr_hq": load_model(d, "QR_Handcrafted_Quantum.joblib"),
        "arb_majority": load_model(d, "Agentic_MajorityVote_Baseline.joblib"),
        "arb_risk": load_model(d, "Agentic_RiskArbitration_X.joblib"),
        "arb_quantum": load_model(d, "Agentic_QuantumArbitration_X.joblib"),
        "qr_resnet_exists": (d / "QR_ResNet18_Agent.pt").exists(),
    }


def get_score(model, X):
    if model is None:
        return None
    try:
        if hasattr(model, "predict_proba"):
            return clamp01(model.predict_proba(X)[:, 1][0])
    except Exception:
        pass
    try:
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function(X), dtype=float)
            if len(s) == 1:
                return clamp01(1 / (1 + np.exp(-s[0])))
            return clamp01((s[0] - s.min()) / (s.max() - s.min() + 1e-12))
    except Exception:
        pass
    try:
        return clamp01(model.predict(X)[0])
    except Exception:
        return None


def url_heuristic(url: str):
    u = str(url or "").lower().strip()
    score = 0.10
    score += 0.18 if len(u) > 75 else 0
    score += 0.18 if any(k in u for k in ["login", "verify", "secure", "account", "update", "bank", "wallet"]) else 0
    score += 0.18 if any(k in u for k in ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly"]) else 0
    score += 0.10 if u.count("-") >= 2 else 0
    score += 0.10 if u.count(".") >= 4 else 0
    score += 0.10 if "@" in u else 0
    score -= 0.08 if u.startswith("https://") else 0
    return clamp01(score)


def url_demo_features(url: str):
    u = str(url or "").strip()
    host = re.sub(r"^https?://", "", u).split("/")[0]
    return pd.DataFrame([{
        "url_len": len(u),
        "host_len": len(host),
        "num_dots": u.count("."),
        "num_hyphen": u.count("-"),
        "num_at": u.count("@"),
        "num_question": u.count("?"),
        "num_equal": u.count("="),
        "num_slash": u.count("/"),
        "num_digits": sum(c.isdigit() for c in u),
        "has_https": int(u.lower().startswith("https://")),
        "has_ip_like": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", u))),
        "has_shortener": int(any(k in u.lower() for k in ["bit.ly", "tinyurl", "t.co", "goo.gl"])),
        "has_login": int("login" in u.lower()),
        "has_verify": int("verify" in u.lower()),
        "has_secure": int("secure" in u.lower()),
        "has_account": int("account" in u.lower()),
    }])


def predict_url(model, url: str):
    fallback = url_heuristic(url)
    if not url.strip():
        return 0.0, "empty"
    s = get_score(model, url_demo_features(url))
    return (s, "model") if s is not None else (fallback, "transparent heuristic")


def text_heuristic(text: str, mode="email"):
    t = str(text or "").lower()
    score = 0.12
    score += 0.22 if any(k in t for k in ["urgent", "immediately", "verify", "suspended", "locked", "expired", "unusual activity"]) else 0
    score += 0.18 if any(k in t for k in ["password", "login", "account", "credential", "security", "bank", "wallet", "payment"]) else 0
    score += 0.16 if any(k in t for k in ["click", "scan", "open", "confirm", "update", "reset"]) else 0
    score += 0.12 if "http://" in t or "https://" in t else 0
    score += 0.08 if mode == "sms" and len(t) < 180 and "urgent" in t else 0
    score += 0.08 if re.search(r"\b\d{4,}\b", t) else 0
    return clamp01(score)


def predict_text(model, text: str, mode="email"):
    fallback = text_heuristic(text, mode)
    if not text.strip():
        return 0.0, "empty"
    s = get_score(model, [text])
    return (s, "model") if s is not None else (fallback, "transparent heuristic")


def qr_features(img):
    img = img.convert("L").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    density = arr.mean()
    std = arr.std()
    edge_h = np.abs(np.diff(arr, axis=1)).mean()
    edge_v = np.abs(np.diff(arr, axis=0)).mean()
    blocks = []
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            blocks.append(arr[i:i+16, j:j+16].mean())
    return np.asarray([[density, std, edge_h, edge_v] + blocks], dtype=np.float32)


def predict_qr(model, uploaded):
    if uploaded is None or Image is None:
        return 0.0, "empty"
    try:
        img = Image.open(uploaded).convert("RGB")
        if model is not None:
            s = get_score(model, qr_features(img))
            if s is not None:
                return s, "model"
        gray = img.convert("L").resize((128, 128))
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        return clamp01(0.25 + 0.45 * arr.std() + 0.30 * np.abs(np.diff(arr, axis=0)).mean()), "transparent heuristic"
    except Exception:
        return 0.5, "transparent heuristic"


def agentic_features(scores):
    vals = np.array(list(scores.values()), dtype=float)
    risk_score = float(vals.mean())
    return pd.DataFrame([{
        "risk_score": risk_score,
        "agent_pred": int(risk_score >= 0.5),
        "risk_margin": abs(risk_score - 0.5),
        "high_confidence": int(abs(risk_score - 0.5) > 0.35),
        "mod_email": 1,
        "mod_qr": 1,
        "mod_sms": 1,
        "mod_url": 1,
    }])


def quantum_arbitration(model, scores):
    if model is not None:
        s = get_score(model, agentic_features(scores))
        if s is not None:
            return s, "trained QArb-X"
    vals = np.array(list(scores.values()), dtype=float)
    conf = np.abs(vals - 0.5) * 2
    w = conf / (conf.sum() + 1e-9)
    if w.sum() == 0:
        w = np.ones_like(vals) / len(vals)
    return clamp01(float(np.sum(vals * w))), "trust-weighted fallback"


def contributions(scores):
    raw = {k: abs(v - 0.5) for k, v in scores.items()}
    total = sum(raw.values()) + 1e-9
    return {k: v / total for k, v in raw.items()}


def agreement(scores):
    names = list(scores.keys())
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = math.exp(-((scores[a] - scores[b]) ** 2) / (2 * 0.2 ** 2))
    return names, mat


def gauge(score):
    band, color, _ = risk_band(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, .45], "color": "#dcfce7"},
                {"range": [.45, .65], "color": "#fef3c7"},
                {"range": [.65, .85], "color": "#fed7aa"},
                {"range": [.85, 1], "color": "#fecaca"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "value": .5},
        },
        title={"text": f"Final Risk: {band}"},
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def reason_codes(scores, url, email, sms):
    reasons = []
    if scores.get("URL-QG", 0) >= .65:
        reasons.append("Suspicious URL or domain-level evidence detected.")
    if scores.get("Email-QT", 0) >= .65:
        reasons.append("Email text contains phishing-style wording or credential request patterns.")
    if scores.get("SMS-QT", 0) >= .65:
        reasons.append("SMS contains smishing indicators such as urgency, verification, or embedded links.")
    if scores.get("QR-HQ/R18", 0) >= .65:
        reasons.append("QR evidence indicates potentially malicious visual or structural patterns.")
    if any(k in (url or "").lower() for k in ["login", "verify", "secure", "account"]):
        reasons.append("URL contains security or account-verification terms.")
    if any(k in ((email or "") + " " + (sms or "")).lower() for k in ["urgent", "suspended", "locked", "verify immediately"]):
        reasons.append("Message uses urgency or account-pressure language.")
    return reasons or ["No dominant phishing indicator found. Review uncertain cases manually."]


def actions(score):
    if score >= .85:
        return ["Block URL", "Quarantine message", "Disable QR destination", "Create incident ticket", "Notify SOC analyst"]
    if score >= .65:
        return ["Warn user", "Send to analyst review", "Log evidence", "Apply temporary URL hold"]
    if score >= .45:
        return ["Request manual review", "Check sender reputation", "Check link redirection path"]
    return ["Allow with logging", "Continue monitoring"]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("QTrustAgentX")
model_dir = st.sidebar.text_input("Model folder", str(DEFAULT_MODEL_DIR))
threshold = st.sidebar.slider("Decision threshold", .10, .90, .50, .01)
mode = st.sidebar.radio("Mode", ["Multi-Channel Scan", "Single URL", "Single Email", "Single SMS", "QR Scan", "Batch CSV Demo", "System Dashboard"])
models_loaded = load_models(model_dir)

with st.sidebar.expander("Model status"):
    for name in [
        "URL_QuantumGraph_Agent.joblib", "EMAIL_QuantumText_Agent.joblib", "SMS_QuantumText_Agent.joblib",
        "QR_Handcrafted_Quantum.joblib", "QR_ResNet18_Agent.pt", "Agentic_QuantumArbitration_X.joblib"
    ]:
        st.write(("✅ " if (Path(model_dir) / name).exists() else "⚠️ ") + name)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="big-title">QTrustAgentX Industrial Phishing Inspector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Specialist agents, trust graph reasoning, quantum arbitration, and analyst-ready explanations.</div>', unsafe_allow_html=True)

if mode == "System Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Agents", "5")
    c2.metric("Modalities", "4")
    c3.metric("Decision Engines", "3")
    c4.metric("Report Export", "JSON")
    st.info("Choose a scan mode from the sidebar.")
    st.stop()

if mode == "Batch CSV Demo":
    f = st.file_uploader("Upload CSV with columns: url, email, sms", type=["csv"])
    if f:
        df = pd.read_csv(f)
        st.dataframe(df.head(20), use_container_width=True)
        st.info("For production batch processing, apply the same scoring functions row by row and export the report.")
    st.stop()

sample_url = "https://secure-paypal-login.verify-user.com/account/update"
sample_email = "Subject: Microsoft Security Alert\n\nWe detected unusual activity. Please verify your account immediately."
sample_sms = "URGENT: Your bank account is locked. Verify now: http://bit.ly/verify-bank"

url = ""
email = ""
sms = ""
qr = None

if mode in ["Multi-Channel Scan", "Single URL"]:
    url = st.text_input("URL Evidence", sample_url if mode == "Multi-Channel Scan" else "")
if mode in ["Multi-Channel Scan", "Single Email"]:
    email = st.text_area("Email Evidence", sample_email if mode == "Multi-Channel Scan" else "", height=150)
if mode in ["Multi-Channel Scan", "Single SMS"]:
    sms = st.text_area("SMS Evidence", sample_sms if mode == "Multi-Channel Scan" else "", height=90)
if mode in ["Multi-Channel Scan", "QR Scan"]:
    qr = st.file_uploader("QR Evidence", type=["png", "jpg", "jpeg", "bmp", "webp"])

if not st.button("Analyse Threat", type="primary", use_container_width=True):
    st.info("Enter evidence and click Analyse Threat.")
    st.stop()

# -----------------------------
# Inference
# -----------------------------
t0 = time.perf_counter()
scores = {}
sources = {}

if url.strip() or mode in ["Multi-Channel Scan", "Single URL"]:
    scores["URL-QG"], sources["URL-QG"] = predict_url(models_loaded["url_qg"], url)
if email.strip() or mode in ["Multi-Channel Scan", "Single Email"]:
    scores["Email-QT"], sources["Email-QT"] = predict_text(models_loaded["email_qt"], email, "email")
if sms.strip() or mode in ["Multi-Channel Scan", "Single SMS"]:
    scores["SMS-QT"], sources["SMS-QT"] = predict_text(models_loaded["sms_qt"], sms, "sms")
if qr is not None or mode in ["Multi-Channel Scan", "QR Scan"]:
    scores["QR-HQ/R18"], sources["QR-HQ/R18"] = predict_qr(models_loaded["qr_hq"], qr)

if not scores:
    st.error("No evidence provided.")
    st.stop()

final_score, arb_src = quantum_arbitration(models_loaded["arb_quantum"], scores)
final_label = "PHISHING" if final_score >= threshold else "LEGITIMATE"
elapsed_ms = (time.perf_counter() - t0) * 1000
band, color, css_class = risk_band(final_score)
contrib = contributions(scores)

# -----------------------------
# Results
# -----------------------------
c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Final Decision")
    st.markdown(f'<div class="{css_class}"><h2>{final_label}</h2><b>Risk band:</b> {band}</div>', unsafe_allow_html=True)
    st.caption(f"Arbitration: {arb_src}")
    st.markdown('</div>', unsafe_allow_html=True)
c2.metric("Risk Score", f"{final_score:.3f}")
c3.metric("Decision Time", f"{elapsed_ms:.1f} ms")
c4.metric("Threshold", f"{threshold:.2f}")

left, right = st.columns([.9, 1.1])
with left:
    st.plotly_chart(gauge(final_score), use_container_width=True)
with right:
    df_scores = pd.DataFrame({"Agent": list(scores.keys()), "Risk": list(scores.values())})
    fig = px.bar(df_scores, x="Agent", y="Risk", text=df_scores["Risk"].map(lambda x: f"{x:.2f}"), color="Agent", range_y=[0, 1])
    fig.add_hline(y=threshold, line_dash="dash", line_color="black")
    fig.update_layout(height=330, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Agent Evidence")
    st.dataframe(pd.DataFrame({
        "Agent": list(scores.keys()),
        "Risk Score": [round(v, 4) for v in scores.values()],
        "Prediction": ["Phishing" if v >= threshold else "Legitimate" for v in scores.values()],
        "Source": [sources[k] for k in scores.keys()],
    }), use_container_width=True)
with col_b:
    st.subheader("Trust Graph Agreement")
    names, mat = agreement(scores)
    fig = px.imshow(mat, x=names, y=names, zmin=0, zmax=1, color_continuous_scale="RdYlGn", text_auto=".2f", labels={"color":"Trust"})
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Modality Contribution")
    df_con = pd.DataFrame({"Agent": list(contrib.keys()), "Contribution": list(contrib.values())})
    fig = px.pie(df_con, names="Agent", values="Contribution", hole=.48)
    fig.update_layout(height=330)
    st.plotly_chart(fig, use_container_width=True)
with col_d:
    st.subheader("Explanation")
    for i, r in enumerate(reason_codes(scores, url, email, sms), 1):
        st.write(f"**RC{i:02d}.** {r}")
    st.subheader("Recommended Actions")
    for a in actions(final_score):
        st.write(f"✓ {a}")

report = {
    "final_label": final_label,
    "final_risk_score": round(final_score, 6),
    "risk_band": band,
    "decision_threshold": threshold,
    "decision_time_ms": round(elapsed_ms, 3),
    "arbitration_source": arb_src,
    "agent_scores": {k: round(v, 6) for k, v in scores.items()},
    "source_used": sources,
    "contributions": {k: round(v, 6) for k, v in contrib.items()},
    "reason_codes": reason_codes(scores, url, email, sms),
    "recommended_actions": actions(final_score),
}

st.download_button("Download JSON Incident Report", json.dumps(report, indent=2), "qtrustagentx_incident_report.json", "application/json", use_container_width=True)

st.caption("If a trained model requires the original feature schema and the UI input does not provide it, the app uses transparent heuristic scoring for that field. For production, connect the same preprocessing used during training.")
