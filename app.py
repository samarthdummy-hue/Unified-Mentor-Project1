"""
Bank Customer Churn — Predictive Risk Intelligence
Streamlit app.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, point Streamlit Cloud at app.py.

Requires artifacts/ produced by train.py (+ explain.py for the SHAP figure).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import prepare, add_features

HERE = os.path.dirname(__file__)
ART = HERE

st.set_page_config(
    page_title="Bank Churn Risk Intelligence",
    page_icon="\U0001F3E6",
    layout="wide",
)

# --------------------------------------------------------------------------
# Loaders (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(os.path.join(ART, "pipeline.pkl"))


@st.cache_resource
def load_metadata():
    with open(os.path.join(ART, "metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(HERE, "European_Bank.csv"))


try:
    pipe = load_pipeline()
    meta = load_metadata()
    data = load_data()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `python train.py` (and `python explain.py`) "
        "first to generate the artifacts/ folder."
    )
    st.stop()

DEFAULT_THRESHOLD = float(meta["threshold"])


def score(df_raw: pd.DataFrame) -> np.ndarray:
    """Raw rows -> churn probability. Identical transform path as training."""
    return pipe.predict_proba(prepare(df_raw))[:, 1]


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("\U0001F3E6 Bank Customer Churn — Risk Intelligence")
st.caption(
    f"Model: **{meta['model']}**  |  ROC-AUC: **{meta['roc_auc']:.3f}**  |  "
    f"Churn base rate: **{meta['base_rate']*100:.1f}%**  |  "
    f"Decision threshold (F1-optimal): **{DEFAULT_THRESHOLD:.2f}**"
)

tab_calc, tab_portfolio, tab_explain, tab_model = st.tabs(
    ["\U0001F9EE Risk Calculator & What-If", "\U0001F4CA Portfolio Analytics",
     "\U0001F50D Explainability", "\u2699\uFE0F Model Card"]
)

# ==========================================================================
# TAB 1 — Single-customer risk calculator + what-if simulator
# ==========================================================================
with tab_calc:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Customer inputs")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 95, 40)
        credit_score = st.slider("Credit score", 350, 850, 650)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 300000.0, 75000.0, step=1000.0)
        salary = st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=1000.0)
        num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
        has_card = st.checkbox("Has credit card", value=True)
        is_active = st.checkbox("Active member", value=True)

        threshold = st.slider(
            "Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
            help="Above this probability, the customer is flagged 'at risk'. "
                 "Lower it to catch more churners (higher recall, more false alarms).",
        )

    row = pd.DataFrame([{
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active), "EstimatedSalary": salary,
    }])
    prob = float(score(row)[0])
    flagged = prob >= threshold

    with right:
        st.subheader("Churn risk")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e53e3e" if flagged else "#38a169"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#e6fffa"},
                    {"range": [threshold * 100, 100], "color": "#fff5f5"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": threshold * 100,
                },
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=20, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        if flagged:
            st.error(f"AT RISK — churn probability {prob*100:.1f}% "
                     f"(\u2265 {threshold*100:.0f}% threshold)")
        else:
            st.success(f"Retained — churn probability {prob*100:.1f}% "
                       f"(< {threshold*100:.0f}% threshold)")

        # What-if: one-variable sensitivity on the live customer
        st.markdown("##### What-if sensitivity")
        var = st.selectbox(
            "Sweep a variable and watch risk move",
            ["Age", "NumOfProducts", "IsActiveMember", "Balance", "CreditScore", "Tenure"],
        )
        sweeps = {
            "Age": np.arange(18, 96, 2),
            "NumOfProducts": np.array([1, 2, 3, 4]),
            "IsActiveMember": np.array([0, 1]),
            "Balance": np.linspace(0, 250000, 30),
            "CreditScore": np.arange(350, 851, 20),
            "Tenure": np.arange(0, 11),
        }
        grid = pd.concat([row] * len(sweeps[var]), ignore_index=True)
        grid[var] = sweeps[var]
        grid_prob = score(grid) * 100
        fig = px.line(
            x=sweeps[var], y=grid_prob, markers=True,
            labels={"x": var, "y": "Churn probability (%)"},
        )
        fig.add_hline(y=threshold * 100, line_dash="dash", line_color="gray")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 2 — Portfolio analytics (scores the whole book live)
# ==========================================================================
with tab_portfolio:
    st.subheader("Live portfolio scoring")
    scored = data.copy()
    scored["churn_prob"] = score(scored)
    thr = st.slider("Portfolio threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
                    key="port_thr")
    scored["flag"] = (scored["churn_prob"] >= thr).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(scored):,}")
    c2.metric("Actual churn", f"{scored['Exited'].mean()*100:.1f}%")
    c3.metric("Flagged at risk", f"{scored['flag'].mean()*100:.1f}%")
    captured = scored.loc[scored["Exited"] == 1, "flag"].mean()
    c4.metric("Churners captured (recall)", f"{captured*100:.1f}%")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Predicted-probability distribution")
        hist = px.histogram(scored, x="churn_prob", nbins=40, color="Exited",
                            barmode="overlay", opacity=0.7,
                            labels={"churn_prob": "Predicted churn probability"})
        hist.add_vline(x=thr, line_dash="dash", line_color="black")
        hist.update_layout(height=350, margin=dict(t=10))
        st.plotly_chart(hist, use_container_width=True)
    with b:
        st.markdown("##### Churn rate by geography")
        geo = (scored.groupby("Geography")["Exited"].mean() * 100).reset_index()
        bar = px.bar(geo, x="Geography", y="Exited",
                     labels={"Exited": "Churn rate (%)"})
        bar.update_layout(height=350, margin=dict(t=10))
        st.plotly_chart(bar, use_container_width=True)

    st.markdown("##### Highest-risk customers")
    cols = ["CustomerId", "Surname", "Geography", "Age", "NumOfProducts",
            "IsActiveMember", "Balance", "churn_prob", "Exited"]
    st.dataframe(
        scored.sort_values("churn_prob", ascending=False)[cols].head(25)
        .style.format({"churn_prob": "{:.1%}", "Balance": "{:,.0f}"}),
        use_container_width=True,
    )

# ==========================================================================
# TAB 3 — Explainability
# ==========================================================================
with tab_explain:
    st.subheader("What drives churn")
    st.write(
        "Permutation importance measures how much ROC-AUC drops when a feature is "
        "shuffled. SHAP shows direction and magnitude per customer."
    )
    e1, e2 = st.columns(2)
    imp_png = os.path.join(ART, "importances.png")
    shap_png = os.path.join(ART, "shap_summary.png")
    if os.path.exists(imp_png):
        e1.image(imp_png, caption="Permutation importance", use_container_width=True)
    if os.path.exists(shap_png):
        e2.image(shap_png, caption="SHAP summary", use_container_width=True)

    st.markdown("##### Ranked drivers")
    imp = pd.Series(meta["importances"]).sort_values(ascending=False)
    st.bar_chart(imp)

# ==========================================================================
# TAB 4 — Model card
# ==========================================================================
with tab_model:
    st.subheader("Model card")
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", f"{meta['roc_auc']:.3f}")
    m2.metric("Recall @ threshold", f"{meta['recall']*100:.1f}%")
    m3.metric("Precision @ threshold", f"{meta['precision']*100:.1f}%")

    st.write(
        f"Accuracy is **{meta['accuracy']*100:.1f}%**, but a model that predicts "
        f"'nobody churns' already scores **{meta['naive_accuracy']*100:.1f}%** on this "
        "imbalanced data — which is why ROC-AUC and recall are the metrics that matter."
    )
    cm = np.array(meta["confusion_matrix"])
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
        x=["Retained", "Churn"], y=["Retained", "Churn"],
    )
    cm_fig.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(cm_fig, use_container_width=True)
    st.caption(
        "Trained on the European_Bank dataset (10,000 customers, 20.4% churn). "
        "Identifiers (CustomerId, Surname) and the constant Year column are excluded."
    )
