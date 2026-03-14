import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="SwipeSafe", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(135deg,#071e38,#0b2236);border:1px solid #0d3050;border-radius:16px;padding:2.5rem 3rem;margin-bottom:1.5rem;}
.hero-badge{font-size:.75rem;letter-spacing:.15em;color:#0af0b0;margin-bottom:.8rem;}
.hero-title{font-size:2.8rem;font-weight:800;color:#00c8ff;margin-bottom:.5rem;}
.hero-sub{color:#4a7fa0;font-size:.95rem;line-height:1.6;}
.stat-row{display:flex;gap:2rem;margin-top:1.2rem;flex-wrap:wrap;}
.stat{border-left:2px solid #0d3050;padding-left:.8rem;}
.stat .v{font-size:1.2rem;color:#00c8ff;font-weight:700;}
.stat .l{font-size:.7rem;color:#4a7fa0;text-transform:uppercase;}
.card{background:#0b2236;border:1px solid #0d3050;border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1rem;}
.card-title{font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#4a7fa0;margin-bottom:1rem;}
.result-fraud{background:rgba(255,60,95,.1);border:1px solid rgba(255,60,95,.5);border-radius:14px;padding:2rem;text-align:center;}
.result-safe{background:rgba(10,240,176,.1);border:1px solid rgba(10,240,176,.5);border-radius:14px;padding:2rem;text-align:center;}
.r-icon{font-size:3rem;}
.r-fraud{font-size:1.8rem;font-weight:800;color:#ff3c5f;}
.r-safe{font-size:1.8rem;font-weight:800;color:#0af0b0;}
.r-sub{font-size:.85rem;color:#4a7fa0;margin-top:.3rem;}
div.stButton>button{background:linear-gradient(135deg,#006bff,#00c8ff)!important;color:#000!important;font-weight:700!important;border:none!important;border-radius:10px!important;padding:.7rem 2rem!important;width:100%!important;font-size:1rem!important;}
div.stNumberInput>div>div>input{background:#071828!important;border:1px solid #0d3050!important;color:#cde8ff!important;border-radius:8px!important;}
label{color:#cde8ff!important;}
.stTabs [data-baseweb="tab-list"]{background:#071828!important;border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#4a7fa0!important;font-weight:600!important;border-radius:8px!important;}
.stTabs [aria-selected="true"]{background:#0b2236!important;color:#00c8ff!important;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH  = "paysim_50_model.pkl"
SCALER_PATH = "paysim_50_scaler.pkl"
IMG_ROC  = "paysim_50_roc_curve.png"
IMG_PR   = "paysim_50_pr_curve.png"
IMG_CM   = "paysim_50_confusion_matrix.png"
IMG_FEAT = "paysim_50_feature_importance.png"
THRESHOLD = 0.28

@st.cache_resource
def load():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

st.markdown("""
<div class="hero">
  <div class="hero-badge">● AI-POWERED · REAL-TIME FRAUD DETECTION</div>
  <div class="hero-title">🛡️ SwipeSafe</div>
  <div class="hero-sub">Advanced fraud detection powered by Random Forest trained on PaySim dataset.</div>
  <div class="stat-row">
    <div class="stat"><div class="v">98.39%</div><div class="l">Accuracy</div></div>
    <div class="stat"><div class="v">0.9997</div><div class="l">ROC-AUC</div></div>
    <div class="stat"><div class="v">0.28</div><div class="l">Threshold</div></div>
    <div class="stat"><div class="v">8</div><div class="l">Features</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

model_ok = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
if not model_ok:
    st.warning(f"PKL files not found in: {os.getcwd()}")
else:
    model, scaler = load()

tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction", "📂 Bulk Analysis", "📊 Model Analytics"])

with tab1:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown('<div class="card"><div class="card-title">📋 Transaction Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            step = st.number_input("Step (hour)", min_value=0, max_value=10000, value=300)
        with c2:
            amount = st.number_input("Amount", min_value=0.0, value=500.0, step=100.0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🏦 Origin Account</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            old_org = st.number_input("Balance Before Tx", min_value=0.0, value=5000.0, step=100.0, key="old_org")
        with c4:
            new_org = st.number_input("Balance After Tx", min_value=0.0, value=4500.0, step=100.0, key="new_org")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🎯 Destination Account</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            old_dest = st.number_input("Balance Before Tx", min_value=0.0, value=0.0, step=100.0, key="old_dest")
        with c6:
            new_dest = st.number_input("Balance After Tx", min_value=0.0, value=500.0, step=100.0, key="new_dest")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🔄 Transaction Type</div>', unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        with tc1:
            type_transfer = st.checkbox("💸 TRANSFER")
        with tc2:
            type_cashout = st.checkbox("🏧 CASH_OUT")
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("⚡ Analyse Transaction", disabled=not model_ok)

    with col2:
        st.markdown("#### 🤖 Prediction Output")
        if predict_btn and model_ok:
            X = pd.DataFrame([{
                "Time": step * 60, "Amount": amount,
                "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
                "type_TRANSFER": 1 if type_transfer else 0,
                "type_CASH_OUT": 1 if type_cashout else 0
            }])
            X[["Time","Amount"]] = scaler.transform(X[["Time","Amount"]])
            prob = model.predict_proba(X)[:,1][0]
            pred = 1 if prob >= THRESHOLD else 0
            pct  = round(prob * 100, 2)
            if pred == 1:
                st.markdown(f'<div class="result-fraud"><div class="r-icon">🚨</div><div class="r-fraud">FRAUDULENT</div><div class="r-sub">Score: {pct}% | Threshold: 28%</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-safe"><div class="r-icon">✅</div><div class="r-safe">LEGITIMATE</div><div class="r-sub">Score: {pct}% | Threshold: 28%</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Fraud Probability: {pct}%**")
            st.progress(float(prob))
            st.dataframe(pd.DataFrame({
                "Feature": ["Step","Amount","Old Bal Org","New Bal Org","Old Bal Dest","New Bal Dest","TRANSFER","CASH_OUT"],
                "Value": [step, f"{amount:,.0f}", f"{old_org:,.0f}", f"{new_org:,.0f}",
                          f"{old_dest:,.0f}", f"{new_dest:,.0f}",
                          "Yes" if type_transfer else "No",
                          "Yes" if type_cashout else "No"]
            }), use_container_width=True, hide_index=True)
        else:
            st.info("👈 Enter transaction details and click Analyse.")

with tab2:
    st.markdown("#### 📂 Bulk CSV Processing")
    st.markdown('<div class="card"><div class="card-title">Required Columns</div><code>step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type</code></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded and model_ok:
        df = pd.read_csv(uploaded)
        df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
        df['type_CASH_OUT']  = (df['type'] == 'CASH_OUT').astype(int)
        df['Time']   = df['step'] * 60
        df['Amount'] = df['amount']
        X = df[['Time','Amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type_TRANSFER','type_CASH_OUT']].copy()
        X[["Time","Amount"]] = scaler.transform(X[["Time","Amount"]])
        probs = model.predict_proba(X)[:,1]
        preds = (probs >= THRESHOLD).astype(int)
        df['Fraud_Probability'] = probs
        df['Status'] = pd.Series(preds).map({0:'✅ Legitimate', 1:'🚨 Fraud'})
        n_fraud = int(preds.sum())
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total", f"{len(df):,}")
        k2.metric("Fraudulent", f"{n_fraud:,}")
        k3.metric("Legitimate", f"{len(df)-n_fraud:,}")
        k4.metric("Avg Score", f"{probs.mean():.3f}")
        st.dataframe(df[['step','amount','type','Fraud_Probability','Status']].head(50), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), file_name="predictions.csv", use_container_width=True)

with tab3:
    st.markdown("#### 📊 Model Performance")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy","98.39%"); m2.metric("ROC-AUC","0.9997")
    m3.metric("Precision","98.44%"); m4.metric("Recall","98.39%")
    st.markdown("---")
    with st.expander("📄 Classification Report"):
        st.code("              precision    recall  f1-score\n   0     1.0000    0.9677    0.9836\n   1     0.9688    1.0000    0.9841\nAccuracy: 0.9839  ROC-AUC: 0.999664")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**📉 ROC Curve**")
        if os.path.exists(IMG_ROC): st.image(IMG_ROC, use_container_width=True)
    with c2:
        st.markdown("**📉 PR Curve**")
        if os.path.exists(IMG_PR): st.image(IMG_PR, use_container_width=True)
    c3,c4 = st.columns(2)
    with c3:
        st.markdown("**🔲 Confusion Matrix**")
        if os.path.exists(IMG_CM): st.image(IMG_CM, use_container_width=True)
    with c4:
        st.markdown("**🏅 Feature Importance**")
        if os.path.exists(IMG_FEAT): st.image(IMG_FEAT, use_container_width=True)
