# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # =========================
# # Load model & columns
# # =========================
# model = pickle.load(open("model.pkl", "rb"))
# columns = pickle.load(open("columns.pkl", "rb"))  # saved X.columns

# st.set_page_config(page_title="Churn Prediction", layout="centered")

# st.title("📊 Customer Churn Prediction")
# st.write("Enter customer details to predict churn")

# # =========================
# # USER INPUTS
# # =========================

# st.subheader("Customer Information")

# # Numeric inputs
# tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
# monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

# # Derived feature (optional)
# total = tenure * monthly

# # Categorical inputs
# contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# payment = st.selectbox(
#     "Payment Method",
#     ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
# )

# security = st.selectbox("Online Security", ["Yes", "No"])
# support = st.selectbox("Tech Support", ["Yes", "No"])
# paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# # =========================
# # PREPROCESS FUNCTION
# # =========================

# def preprocess():
#     input_dict = {
#         "tenure": tenure,
#         "MonthlyCharges": monthly,
#         "TotalCharges": total,
#         "Contract": contract,
#         "InternetService": internet,
#         "PaymentMethod": payment,
#         "OnlineSecurity": security,
#         "TechSupport": support,
#         "PaperlessBilling": paperless
#     }

#     input_df = pd.DataFrame([input_dict])

#     # One-hot encode
#     input_df = pd.get_dummies(input_df)

#     # Align with training columns
#     input_df = input_df.reindex(columns=columns, fill_value=0)

#     return input_df


# # =========================
# # PREDICTION
# # =========================

# if st.button("Predict Churn"):

#     input_df = preprocess()

#     prediction = model.predict(input_df)[0]
#     probability = model.predict_proba(input_df)[0][1]

#     st.subheader("Result")

#     if prediction == 1:
#         st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2f}")
#     else:
#         st.success(f"✅ Customer is likely to STAY\n\nProbability: {probability:.2f}")

#     # Extra insight
#     st.subheader("Confidence Level")
#     st.progress(int(probability * 100))




import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# =========================
# Page Config (must be first)
# =========================
st.set_page_config(
    page_title="ChurnRadar | Customer Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom CSS — Dark Futuristic Theme
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #050A12;
    color: #E2E8F0;
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0D1F3C 0%, #050A12 60%),
                radial-gradient(ellipse at 80% 100%, #0A1628 0%, transparent 60%);
    min-height: 100vh;
}

/* ── Header ── */
.churn-header {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}

.churn-header::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 600px;
    height: 200px;
    background: radial-gradient(ellipse, rgba(0, 200, 255, 0.06) 0%, transparent 70%);
    pointer-events: none;
}

.header-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    color: #00C8FF;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: block;
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #94C8E8 50%, #00C8FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}

.header-sub {
    font-size: 1rem;
    color: #5A7FA0;
    margin-top: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.05em;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #00C8FF;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(0,200,255,0.3), transparent);
}

/* ── Cards / Panels ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.4), transparent);
}

.metric-card {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #5A7FA0;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #00C8FF;
    line-height: 1;
}

/* ── Result Banner ── */
.result-churn {
    background: linear-gradient(135deg, rgba(255,60,60,0.1) 0%, rgba(255,100,50,0.05) 100%);
    border: 1px solid rgba(255,80,80,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: pulseRed 2s ease-in-out infinite;
}
.result-stay {
    background: linear-gradient(135deg, rgba(0,220,130,0.1) 0%, rgba(0,180,255,0.05) 100%);
    border: 1px solid rgba(0,220,130,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: pulseGreen 2s ease-in-out infinite;
}

@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,60,60,0); }
    50% { box-shadow: 0 0 30px 0 rgba(255,60,60,0.15); }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,220,130,0); }
    50% { box-shadow: 0 0 30px 0 rgba(0,220,130,0.15); }
}

.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-verdict {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0.25rem 0;
}
.verdict-churn { color: #FF6060; }
.verdict-stay { color: #00DC82; }
.result-prob {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #5A7FA0;
    letter-spacing: 0.15em;
    margin-top: 0.5rem;
}

/* ── Probability Bar ── */
.prob-bar-wrap {
    margin: 1.5rem 0 0.5rem;
    position: relative;
}
.prob-bar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}
.prob-bar-fill-churn {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #FF9060, #FF4040);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.prob-bar-fill-stay {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00C8FF, #00DC82);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.prob-bar-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #5A7FA0;
    margin-top: 0.4rem;
    letter-spacing: 0.1em;
}

/* ── Factor Chips ── */
.factor-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.factor-chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    letter-spacing: 0.1em;
}
.chip-risk {
    background: rgba(255,80,80,0.1);
    border: 1px solid rgba(255,80,80,0.25);
    color: #FF9090;
}
.chip-safe {
    background: rgba(0,220,130,0.08);
    border: 1px solid rgba(0,220,130,0.2);
    color: #00DC82;
}
.chip-neutral {
    background: rgba(0,200,255,0.06);
    border: 1px solid rgba(0,200,255,0.15);
    color: #70C8E8;
}

/* ── Streamlit Widget Overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
}

label, .stSelectbox label, .stNumberInput label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    color: #5A7FA0 !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0070CC 0%, #00C8FF 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2.5rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 30px rgba(0, 200, 255, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 40px rgba(0, 200, 255, 0.4) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050A12; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 10px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)


# =========================
# Load model & columns
# =========================
@st.cache_resource
def load_model():
    # model = pickle.load(open(r"C:\Users\Admin\Downloads\churn\model.pkl", "rb"))
    # columns = pickle.load(open(r"C:\Users\Admin\Downloads\churn\columns.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))  # saved X.columns
    return model, columns

model, columns = load_model()


# =========================
# Header
# =========================
st.markdown("""
<div class="churn-header">
    <span class="header-tag">📡 AI-Powered Analytics</span>
    <h1 class="header-title">ChurnRadar</h1>
    <p class="header-sub">Predict customer attrition before it happens — in real time.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================
# Layout: Two columns
# =========================
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:

    # — Numeric Inputs —
    st.markdown('<div class="section-label">01 — Core Metrics</div>', unsafe_allow_html=True)
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    with c2:
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.5)

    total = tenure * monthly

    # Mini metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tenure</div>
            <div class="metric-value">{tenure}<span style="font-size:0.8rem;color:#5A7FA0">mo</span></div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monthly</div>
            <div class="metric-value">${monthly:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lifetime Value</div>
            <div class="metric-value">${total:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # — Plan & Billing —
    st.markdown('<div class="section-label">02 — Plan & Billing</div>', unsafe_allow_html=True)
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c4:
        payment = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # — Support Services —
    st.markdown('<div class="section-label">03 — Services & Support</div>', unsafe_allow_html=True)
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        security = st.selectbox("Online Security", ["Yes", "No"])
    with c6:
        support = st.selectbox("Tech Support", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # — Predict Button —
    predict_clicked = st.button("⚡ Run Churn Analysis")


# =========================
# Preprocess
# =========================
def preprocess():
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "OnlineSecurity": security,
        "TechSupport": support,
        "PaperlessBilling": paperless
    }
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)
    return input_df


# =========================
# Right Panel — Results
# =========================
with col_right:
    st.markdown('<div class="section-label">04 — Prediction Output</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3.5rem 2rem; min-height: 260px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
            <div style="font-size:2.5rem; margin-bottom:1rem; opacity:0.4;">📡</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.2em; color:#2A4060; text-transform:uppercase;">
                Awaiting Input<br><br>
                <span style="font-size:0.55rem; color:#1A3050;">Fill in customer details<br>and run the analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner(""):
            time.sleep(0.4)
            input_df = preprocess()
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

        pct = int(probability * 100)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-churn">
                <div class="result-icon">⚠️</div>
                <div class="result-verdict verdict-churn">HIGH CHURN RISK</div>
                <div style="font-size:3rem;font-weight:800;color:#FF4040;margin:0.25rem 0">{pct}%</div>
                <div class="result-prob">CHURN PROBABILITY</div>
            </div>""", unsafe_allow_html=True)
            bar_class = "prob-bar-fill-churn"
            status_color = "#FF6060"
        else:
            st.markdown(f"""
            <div class="result-stay">
                <div class="result-icon">✅</div>
                <div class="result-verdict verdict-stay">LIKELY TO STAY</div>
                <div style="font-size:3rem;font-weight:800;color:#00DC82;margin:0.25rem 0">{pct}%</div>
                <div class="result-prob">CHURN PROBABILITY</div>
            </div>""", unsafe_allow_html=True)
            bar_class = "prob-bar-fill-stay"
            status_color = "#00DC82"

        # Probability bar
        st.markdown(f"""
        <div class="prob-bar-wrap">
            <div class="prob-bar-track">
                <div class="{bar_class}" style="width:{pct}%"></div>
            </div>
            <div class="prob-bar-labels">
                <span>LOW RISK</span>
                <span>{pct}% CHURN SCORE</span>
                <span>HIGH RISK</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk factor analysis
        st.markdown('<div class="section-label">05 — Risk Factor Analysis</div>', unsafe_allow_html=True)
        #st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        factors = []

        # Contract risk
        if contract == "Month-to-month":
            factors.append(("Month-to-month contract", "risk"))
        elif contract == "Two year":
            factors.append(("Two-year contract", "safe"))
        else:
            factors.append(("One-year contract", "neutral"))

        # Tenure risk
        if tenure < 6:
            factors.append(("New customer (<6mo)", "risk"))
        elif tenure > 24:
            factors.append(("Loyal customer (>24mo)", "safe"))
        else:
            factors.append(("Moderate tenure", "neutral"))

        # Internet
        if internet == "Fiber optic":
            factors.append(("Fiber optic (higher churn)", "risk"))
        elif internet == "No":
            factors.append(("No internet service", "safe"))
        else:
            factors.append(("DSL service", "neutral"))

        # Security
        if security == "No":
            factors.append(("No online security", "risk"))
        else:
            factors.append(("Online security active", "safe"))

        # Support
        if support == "No":
            factors.append(("No tech support", "risk"))
        else:
            factors.append(("Tech support enrolled", "safe"))

        # Payment
        if payment == "Electronic check":
            factors.append(("Electronic check payment", "risk"))
        else:
            factors.append((payment, "neutral"))

        # Monthly charge
        if monthly > 75:
            factors.append((f"High monthly charge (${monthly:.0f})", "risk"))
        elif monthly < 30:
            factors.append((f"Low monthly charge (${monthly:.0f})", "safe"))

        chips_html = '<div class="factor-row">'
        for label, kind in factors:
            css = f"chip-{kind}"
            icon = "↑" if kind == "risk" else "↓" if kind == "safe" else "•"
            chips_html += f'<span class="factor-chip {css}">{icon} {label}</span>'
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Confidence breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        stay_pct = 100 - pct
        st.markdown(f"""
        <div class="glass-card">
            <div class="section-label" style="margin-bottom:1rem;">Model Confidence Breakdown</div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#00DC82;letter-spacing:0.1em;">STAY</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#E2E8F0;">{stay_pct}%</span>
            </div>
            <div class="prob-bar-track" style="margin-bottom:0.75rem;">
                <div style="height:100%;border-radius:100px;background:linear-gradient(90deg,#00C8FF,#00DC82);width:{stay_pct}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#FF6060;letter-spacing:0.1em;">CHURN</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#E2E8F0;">{pct}%</span>
            </div>
            <div class="prob-bar-track">
                <div style="height:100%;border-radius:100px;background:linear-gradient(90deg,#FF9060,#FF4040);width:{pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================
# Footer
# =========================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.55rem; letter-spacing:0.2em; color:#1A3050; text-transform:uppercase;">
    ChurnRadar &nbsp;·&nbsp; Powered by Machine Learning &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)















































