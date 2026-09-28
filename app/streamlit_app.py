import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ── Config ──────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(page_title="csurv — Customer Churn & CLV", layout="wide")
st.title("Customer Churn & Lifetime Value")
st.caption("Mixture cure survival model predictions")

# ── Sidebar: inputs ────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    db_path = st.text_input("Database path", value="data/demo.db")
    horizon_days = st.slider("Prediction horizon (days)", 30, 365, 90)
    avg_ticket = st.number_input("Avg ticket ($)", min_value=1.0, value=65.0, step=5.0)
    churn_reduction_factor = st.slider(
        "Churn reduction factor",
        min_value=0.50, max_value=1.00, value=0.90, step=0.05,
        help="Multiplier on churn probability (0.90 = 10% reduction)"
    )
    run = st.button("Run predictions", type="primary")


# ── Helper: call API ───────────────────────────────────────────
def api_post(endpoint: str, payload: dict):
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is uvicorn running?")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        st.stop()


# ── Main: run on button click ──────────────────────────────────
if run:
    payload = {
        "db_path": db_path,
        "horizon_days": horizon_days,
        "avg_ticket": avg_ticket,
        "churn_reduction_factor": churn_reduction_factor,
    }

    # ── Predictions ─────────────────────────────────────────────
    with st.spinner("Running predictions..."):
        pred_data = api_post("/predict", payload)

    df = pd.DataFrame(pred_data["predictions"])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", pred_data["n_customers"])
    col2.metric("Mean Churn Prob", f"{pred_data['mean_churn']:.1%}")
    col3.metric("Mean CLV", f"${pred_data['mean_clv']:.0f}")
    col4.metric("Horizon", f"{horizon_days} days")

    st.divider()

    # Two-column layout: table + distributions
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Customer Predictions")
        st.dataframe(
            df.style.format({
                "pi": "{:.1%}",
                "k": "{:.2f}",
                "lam": "{:.1f}",
                "mean_return": "{:.1f}",
                "median_return": "{:.1f}",
                "expected_visits": "{:.1f}",
                "clv": "${:.0f}",
            }),
            use_container_width=True,
            height=400,
        )

    with right:
        st.subheader("Churn Distribution")
        fig_hist = go.Figure(go.Histogram(x=df["pi"], nbinsx=30, marker_color="#7678E3"))
        fig_hist.update_layout(
            xaxis_title="Churn Probability",
            yaxis_title="Count",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Top churn risks ─────────────────────────────────────────
    st.subheader("Top 10 Highest Churn Risk")
    top10 = df.nlargest(10, "pi")[["visit_id", "pi", "mean_return", "expected_visits", "clv"]]
    st.dataframe(
        top10.style.format({
            "pi": "{:.1%}", "mean_return": "{:.1f}",
            "expected_visits": "{:.1f}", "clv": "${:.0f}",
        }),
        use_container_width=True,
    )

    st.divider()

    # ── Survival curve ──────────────────────────────────────────
    with st.spinner("Computing survival curve..."):
        surv_data = api_post("/survival", payload)

    st.subheader("Customer Survival Curve")

    surv_df = pd.DataFrame(surv_data["curve"])
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=surv_df["day"], y=surv_df["survival_probability"],
        mode="lines", line=dict(color="#7678E3", width=2),
        name="Survival probability",
    ))
    # Cure fraction floor
    cure = surv_data["cure_fraction"]
    fig_surv.add_hline(
        y=cure, line_dash="dash", line_color="gray",
        annotation_text=f"Cure fraction: {cure:.0%}",
        annotation_position="bottom right",
    )
    fig_surv.update_layout(
        xaxis_title="Days from now",
        yaxis_title="P(still active)",
        yaxis_range=[0, 1],
        margin=dict(l=40, r=20, t=20, b=40),
        height=400,
    )
    st.plotly_chart(fig_surv, use_container_width=True)

    st.caption(f"Based on {surv_data['n_customers']} customers")

    st.divider()

    # ── CLV Uplift ──────────────────────────────────────────────
    if churn_reduction_factor < 1.0:
        with st.spinner("Computing uplift..."):
            uplift_data = api_post("/uplift", payload)

        st.subheader("CLV Uplift Analysis")
        st.caption(f"Impact of reducing churn to {churn_reduction_factor:.0%} of current level")

        u1, u2, u3 = st.columns(3)
        u1.metric("Baseline CLV", f"${uplift_data['mean_clv_baseline']:.0f}")
        u2.metric("Reduced-Churn CLV", f"${uplift_data['mean_clv_reduced']:.0f}")
        u3.metric(
            "Uplift",
            f"${uplift_data['mean_clv_uplift']:.0f}",
            delta=f"{uplift_data['mean_clv_uplift_pct']:.1%}",
        )
    else:
        st.info("Set churn reduction factor below 1.0 in the sidebar to see uplift analysis.")

else:
    st.info("Configure parameters in the sidebar and click **Run predictions** to start.")