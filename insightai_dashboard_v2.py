import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InSightAI – Insider Threat Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 0.5rem; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .section-head {
        font-size: 14px; font-weight: 600;
        color: #1a1a2e; margin-bottom: 0.4rem;
        border-bottom: 2px solid #0d6efd;
        padding-bottom: 4px; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ── API CONFIG ────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
DATA = "data"

# ── HELPER: Call API or fallback to CSV ──────────────────────────────────────
def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def backend_online():
    try:
        r = requests.get(f"{API_BASE}/docs", timeout=3)
        return r.status_code == 200
    except:
        return False

# ── DATA LOADING (CSV fallback) ───────────────────────────────────────────────
@st.cache_data
def load_all():
    iso    = pd.read_csv(f"{DATA}/isolation_forest_results.csv")
    weekly = pd.read_csv(f"{DATA}/weekly_behavioral_risk_scores.csv")
    sna    = pd.read_csv(f"{DATA}/sna_final_with_mitre.csv", low_memory=False)
    shap_g = pd.read_csv(f"{DATA}/shap_global_feature_importance (4).csv")
    pred   = pd.read_csv(f"{DATA}/dashboard_ready_all_high_risk_explanations.csv")
    detail = pd.read_csv(f"{DATA}/dashboard_ready_all_high_risk_explanations_detailed.csv")
    fused  = pd.read_csv(f"{DATA}/fused_dashboard_payload.csv")

    weekly["week_dt"] = pd.to_datetime(weekly["week"], format="%d-%m-%Y", errors="coerce")
    min_date = weekly["week_dt"].min()
    weekly["week_number"] = ((weekly["week_dt"] - min_date).dt.days // 7) + 1

    sna_u = sna.drop_duplicates(subset="user").copy()
    return iso, weekly, sna_u, shap_g, pred, detail, fused

iso, weekly, sna_u, shap_g, pred, detail, fused = load_all()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 InSightAI")
    st.caption("Insider Threat Detection & Risk Prediction")
    st.markdown("---")

    # Backend status indicator
    online = backend_online()
    if online:
        st.success("🟢 Backend API: Online")
    else:
        st.warning("🔴 Backend API: Offline — using CSV data")

    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview",
        "🕸️ SNA Risk Analysis",
        "📈 Weekly Trends",
        "⚠️ High-Risk Predictions",
        "🔎 SHAP Explainability",
        "👤 User Profile",
        "🧪 Live Prediction"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** CERT r4.2")
    st.markdown(f"**Users monitored:** {len(iso):,}")
    st.markdown(f"**User-week records:** {len(weekly):,}")
    st.markdown(f"**Model:** Gradient Boosting")
    st.markdown(f"**Accuracy:** 0.98 | **F1:** 0.86")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 System Overview")
    st.caption("InSightAI — CERT r4.2 insider threat detection and risk prediction")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users Monitored", f"{len(iso):,}")
    c2.metric("Anomalous Users", f"{int(iso['is_anomalous'].sum()):,}",
              f"{iso['is_anomalous'].mean()*100:.2f}% flagged", delta_color="inverse")
    c3.metric("User-Week Records", f"{len(weekly):,}")
    c4.metric("High-Risk Predictions", f"{len(pred):,}", delta_color="inverse")
    c5.metric("Model Accuracy", "0.98")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-head">Weekly Average Risk Score</div>', unsafe_allow_html=True)
        wavg = weekly.groupby("week_number")["behavioral_risk_score"].mean().reset_index()
        fig = px.area(wavg, x="week_number", y="behavioral_risk_score",
                      labels={"week_number": "Week", "behavioral_risk_score": "Avg Risk Score"},
                      color_discrete_sequence=["#0d6efd"])
        fig.add_vline(x=20, line_dash="dash", line_color="red",
                      annotation_text="Behavioral Shift (Wk 20)", annotation_font_color="red")
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">MITRE ATT&CK Distribution</div>', unsafe_allow_html=True)
        mitre_counts = fused["MITRE_Technique"].fillna("Normal Activity").value_counts().reset_index()
        mitre_counts.columns = ["Technique", "Count"]
        fig2 = px.pie(mitre_counts, names="Technique", values="Count", hole=0.55,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
                           legend=dict(font=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-head">Model Performance Comparison</div>', unsafe_allow_html=True)
        models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Accuracy",  x=models, y=[0.94, 0.97, 0.98], marker_color="#0d6efd"))
        fig3.add_trace(go.Bar(name="Precision", x=models, y=[0.81, 0.85, 0.88], marker_color="#198754"))
        fig3.add_trace(go.Bar(name="F1-Score",  x=models, y=[0.79, 0.83, 0.86], marker_color="#fd7e14"))
        fig3.update_layout(barmode="group", height=280, yaxis=dict(range=[0.7,1.0]),
                           margin=dict(t=10,b=10,l=10,r=10),
                           plot_bgcolor="white", paper_bgcolor="white",
                           legend=dict(font=dict(size=11)))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-head">Combined Risk Score Distribution</div>', unsafe_allow_html=True)
        fig4 = px.histogram(fused, x="CombinedRiskScore", nbins=40,
                            labels={"CombinedRiskScore": "Combined Risk Score"},
                            color_discrete_sequence=["#dc3545"])
        fig4.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Top 10 Highest Risk Users</div>', unsafe_allow_html=True)
    top_users = (fused.groupby("UserID")["CombinedRiskScore"]
                 .agg(["mean","max"]).reset_index()
                 .rename(columns={"mean":"Avg Risk","max":"Peak Risk"})
                 .sort_values("Avg Risk", ascending=False).head(10))
    top_users["Avg Risk"] = top_users["Avg Risk"].round(2)
    top_users["Peak Risk"] = top_users["Peak Risk"].round(2)
    st.dataframe(top_users, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SNA RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🕸️ SNA Risk Analysis":
    st.title("🕸️ SNA Risk Analysis")
    st.caption("Multi-view social network risk scores — email 40%, file 35%, device 25%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Email Weight", "40%")
    c2.metric("File Weight", "35%")
    c3.metric("Device Weight", "25%")
    c4.metric("Risk Threshold", "0.60")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-head">Channel Risk Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for ch, color in [("email_risk","#0d6efd"),("file_risk","#fd7e14"),("device_risk","#dc3545")]:
            fig.add_trace(go.Histogram(x=sna_u[ch], name=ch.replace("_"," ").title(),
                                       nbinsx=30, opacity=0.7, marker_color=color))
        fig.update_layout(barmode="overlay", height=300,
                          margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">MITRE Technique Breakdown</div>', unsafe_allow_html=True)
        mt = sna_u["mitre_technique"].fillna("Low Risk").value_counts().reset_index()
        mt.columns = ["Technique","Count"]
        fig2 = px.bar(mt, y="Technique", x="Count", orientation="h",
                      color="Count", color_continuous_scale=["#e9ecef","#dc3545"])
        fig2.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                           coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-head">User SNA Risk Table</div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        tactic_filter = st.selectbox("Filter by MITRE Tactic",
                                      ["All"] + sorted(sna_u["mitre_tactic"].dropna().unique().tolist()))
    with col_f2:
        threshold = st.slider("Minimum SNA Risk Score", 0.0, 1.0, 0.0, 0.05)

    filtered_sna = sna_u.copy()
    if tactic_filter != "All":
        filtered_sna = filtered_sna[filtered_sna["mitre_tactic"] == tactic_filter]
    filtered_sna = filtered_sna[filtered_sna["sna_risk"] >= threshold]
    filtered_sna = filtered_sna.sort_values("sna_risk", ascending=False).reset_index(drop=True)

    display_cols = [c for c in ["user","email_risk","file_risk","device_risk",
                                 "sna_risk","mitre_technique","explanation"] if c in filtered_sna.columns]
    st.dataframe(
        filtered_sna[display_cols].head(200)
        .style.background_gradient(subset=["sna_risk"], cmap="Reds"),
        use_container_width=True, height=400
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WEEKLY TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Weekly Trends":
    st.title("📈 Weekly Behavioral Trends")
    st.caption("Dynamic weekly anomaly detection — 47,776 user-week records across 58 weeks")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total User-Weeks", f"{len(weekly):,}")
    c2.metric("Anomalous User-Weeks", f"{int(weekly['is_anomalous'].sum()):,}",
              f"{weekly['is_anomalous'].mean()*100:.1f}%", delta_color="inverse")
    c3.metric("Normal User-Weeks", f"{int((weekly['is_anomalous']==0).sum()):,}")

    st.markdown("---")
    st.markdown('<div class="section-head">Population Weekly Average Risk Score</div>', unsafe_allow_html=True)
    wavg = weekly.groupby("week_number")["behavioral_risk_score"].mean().reset_index()
    fig = px.area(wavg, x="week_number", y="behavioral_risk_score",
                  labels={"week_number":"Week","behavioral_risk_score":"Avg Risk Score"},
                  color_discrete_sequence=["#0d6efd"])
    fig.add_vline(x=20, line_dash="dash", line_color="red",
                  annotation_text="Behavioral Shift (Week 20)", annotation_font_color="red")
    fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Individual User Risk Trajectory</div>', unsafe_allow_html=True)
    user_list = sorted(weekly["user"].unique())
    selected_user = st.selectbox("Select User", user_list)

    user_data = weekly[weekly["user"] == selected_user].sort_values("week_number")
    fig2 = px.line(user_data, x="week_number", y="behavioral_risk_score",
                   markers=True, color_discrete_sequence=["#0d6efd"],
                   labels={"week_number":"Week","behavioral_risk_score":"Risk Score (1-10)"})
    fig2.add_hline(y=7, line_dash="dot", line_color="orange",
                   annotation_text="High Risk Threshold (7)")
    fig2.add_vline(x=20, line_dash="dash", line_color="red",
                   annotation_text="Behavioral Shift")
    anom_weeks = user_data[user_data["is_anomalous"] == 1]
    if not anom_weeks.empty:
        fig2.add_trace(go.Scatter(
            x=anom_weeks["week_number"], y=anom_weeks["behavioral_risk_score"],
            mode="markers", marker=dict(color="#dc3545", size=10, symbol="x"),
            name="Anomalous Week"
        ))
    fig2.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**User Statistics**")
        st.write(f"Mean Risk Score: **{user_data['behavioral_risk_score'].mean():.2f}**")
        st.write(f"Peak Risk Score: **{user_data['behavioral_risk_score'].max():.0f}**")
        st.write(f"Anomalous Weeks: **{int(user_data['is_anomalous'].sum())}** / {len(user_data)}")
    with col_b:
        st.markdown("**Weekly Data**")
        st.dataframe(user_data[["week","behavioral_risk_score","is_anomalous"]].reset_index(drop=True),
                     use_container_width=True, height=200)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HIGH-RISK PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ High-Risk Predictions":
    st.title("⚠️ High-Risk Predictions")
    st.caption("Gradient Boosting predictions on held-out test set — 696 high-risk cases")

    tp = int((pred["actual_label"] == 1).sum())
    fp = int((pred["actual_label"] == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", f"{len(pred):,}")
    c2.metric("True Positives", f"{tp:,}", f"{tp/len(pred)*100:.1f}%")
    c3.metric("False Positives", f"{fp:,}", f"{fp/len(pred)*100:.1f}%", delta_color="inverse")
    c4.metric("Avg Confidence", f"{pred['predicted_probability'].mean():.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-head">Prediction Confidence Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(pred, x="predicted_probability", nbins=30,
                           color_discrete_sequence=["#0d6efd"],
                           labels={"predicted_probability":"Predicted Probability"})
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">True vs False Positive</div>', unsafe_allow_html=True)
        tp_fp = pred["actual_label"].value_counts().reset_index()
        tp_fp.columns = ["Label","Count"]
        tp_fp["Label"] = tp_fp["Label"].map({1:"True Positive", 0:"False Positive"})
        fig2 = px.pie(tp_fp, names="Label", values="Count", hole=0.5,
                      color="Label",
                      color_discrete_map={"True Positive":"#198754","False Positive":"#dc3545"})
        fig2.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-head">All High-Risk Predictions</div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_prob = st.slider("Minimum Predicted Probability", 0.5, 1.0, 0.5, 0.05)
    with col_f2:
        result_filter = st.radio("Show", ["All","True Positives Only","False Positives Only"], horizontal=True)

    filtered_pred = pred[pred["predicted_probability"] >= min_prob].copy()
    if result_filter == "True Positives Only":
        filtered_pred = filtered_pred[filtered_pred["actual_label"] == 1]
    elif result_filter == "False Positives Only":
        filtered_pred = filtered_pred[filtered_pred["actual_label"] == 0]

    filtered_pred["Result"] = filtered_pred["actual_label"].map({1:"✅ True Positive", 0:"❌ False Positive"})
    filtered_pred["predicted_probability"] = filtered_pred["predicted_probability"].round(4)

    st.dataframe(
        filtered_pred[["row_index","predicted_probability","Result","explanation"]],
        use_container_width=True, height=450,
        column_config={
            "row_index": "Case Index",
            "predicted_probability": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.4f"),
            "Result": "Result",
            "explanation": "Explanation"
        },
        hide_index=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 SHAP Explainability":
    st.title("🔎 SHAP Explainability")
    st.caption("Global and local feature-level explanations for Gradient Boosting predictions")

    st.markdown('<div class="section-head">Global Feature Importance — Mean Absolute SHAP Values</div>',
                unsafe_allow_html=True)
    shap_sorted = shap_g.sort_values("Mean_Abs_SHAP", ascending=True)
    fig = px.bar(shap_sorted, x="Mean_Abs_SHAP", y="Feature", orientation="h",
                 color="Mean_Abs_SHAP",
                 color_continuous_scale=["#e9ecef","#0d6efd"],
                 labels={"Mean_Abs_SHAP":"Mean |SHAP|","Feature":""})
    fig.update_layout(height=500, margin=dict(t=10,b=10,l=10,r=10),
                      coloraxis_showscale=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Local Explanation — Select High-Risk Case</div>',
                unsafe_allow_html=True)
    case_list = sorted(detail["row_index"].unique())
    selected_case = st.selectbox("Select Case Index", case_list)

    case_data = detail[detail["row_index"] == selected_case].sort_values("shap_value", ascending=False)
    case_pred = pred[pred["row_index"] == selected_case]

    if not case_pred.empty:
        prob = case_pred["predicted_probability"].values[0]
        expl = case_pred["explanation"].values[0]
        actual = case_pred["actual_label"].values[0]
        result = "✅ True Positive" if actual == 1 else "❌ False Positive"
        st.info(f"**Case {selected_case}** | Predicted Probability: **{prob:.4f}** | {result}")
        st.success(f"**Explanation:** {expl}")

    colors = ["#dc3545" if v > 0 else "#0d6efd" for v in case_data["shap_value"]]
    fig2 = go.Figure(go.Bar(
        x=case_data["shap_value"], y=case_data["feature"],
        orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in case_data["shap_value"]],
        textposition="outside"
    ))
    fig2.add_vline(x=0, line_color="black", line_width=0.5)
    fig2.update_layout(height=400, margin=dict(t=10,b=10,l=10,r=10),
                       xaxis_title="SHAP Value",
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        case_data[["rank","feature","feature_value","shap_value","plain_explanation"]],
        use_container_width=True, height=300, hide_index=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — USER PROFILE (API-powered)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 User Profile":
    st.title("👤 User Risk Profile")
    st.caption("Drill down into individual user behavioral patterns and risk signals")

    user_list = sorted(weekly["user"].unique())
    selected_user = st.selectbox("Select User ID", user_list)

    user_weekly = weekly[weekly["user"] == selected_user].sort_values("week_number")
    user_fused  = fused[fused["UserID"] == selected_user].sort_values("Week")

    # Try API first, fallback to CSV
    api_user_data = api_get(f"/api/dashboard/user/{selected_user}")
    if api_user_data:
        user_sna_row = pd.Series(api_user_data)
        st.caption("🟢 SNA data loaded from Backend API")
    else:
        sna_match = sna_u[sna_u["user"] == selected_user]
        user_sna_row = sna_match.iloc[0] if not sna_match.empty else None
        st.caption("🔴 SNA data loaded from CSV (API offline)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Risk Score", f"{user_weekly['behavioral_risk_score'].mean():.2f}")
    c2.metric("Peak Risk Score", f"{user_weekly['behavioral_risk_score'].max():.0f}")
    c3.metric("Anomalous Weeks", f"{int(user_weekly['is_anomalous'].sum())}")
    if user_sna_row is not None and "sna_risk" in user_sna_row:
        c4.metric("SNA Risk", f"{float(user_sna_row['sna_risk']):.3f}")
    else:
        c4.metric("SNA Risk", "N/A")

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<div class="section-head">Weekly Risk Trajectory</div>', unsafe_allow_html=True)
        fig = px.line(user_weekly, x="week_number", y="behavioral_risk_score",
                      markers=True, color_discrete_sequence=["#0d6efd"],
                      labels={"week_number":"Week","behavioral_risk_score":"Risk Score (1-10)"})
        fig.add_hline(y=7, line_dash="dot", line_color="orange",
                      annotation_text="High Risk Threshold")
        fig.add_vline(x=20, line_dash="dash", line_color="red",
                      annotation_text="Behavioral Shift")
        anom = user_weekly[user_weekly["is_anomalous"] == 1]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom["week_number"], y=anom["behavioral_risk_score"],
                mode="markers", marker=dict(color="#dc3545", size=10, symbol="x"),
                name="Anomalous"
            ))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-head">SNA Channel Risk</div>', unsafe_allow_html=True)
        if user_sna_row is not None:
            channels = ["Email Risk","File Risk","Device Risk","SNA Risk"]
            values = [float(user_sna_row.get("email_risk", 0)),
                      float(user_sna_row.get("file_risk", 0)),
                      float(user_sna_row.get("device_risk", 0)),
                      float(user_sna_row.get("sna_risk", 0))]
            fig2 = go.Figure(go.Bar(
                x=channels, y=values,
                marker_color=["#0d6efd","#fd7e14","#dc3545","#198754"],
                text=[f"{v:.3f}" for v in values],
                textposition="outside"
            ))
            fig2.add_hline(y=0.6, line_dash="dot", line_color="red",
                           annotation_text="Threshold 0.6")
            fig2.update_layout(height=300, yaxis=dict(range=[0,1.1]),
                               margin=dict(t=10,b=10,l=10,r=10),
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

    if user_sna_row is not None:
        tactic = user_sna_row.get("mitre_tactic", None)
        technique = user_sna_row.get("mitre_technique", None)
        explanation = user_sna_row.get("explanation", None)
        if pd.notna(tactic) if isinstance(tactic, float) else tactic:
            st.warning(f"⚠️ **MITRE Tactic:** {tactic} | **Technique:** {technique}")
            if explanation:
                st.info(f"**Behavioral Explanation:** {explanation}")
        else:
            st.success("✅ **MITRE:** Low Risk — No suspicious tactic triggered")

    if not user_fused.empty:
        st.markdown("---")
        st.markdown('<div class="section-head">Combined Risk Score Over Time</div>', unsafe_allow_html=True)
        fig3 = px.line(user_fused, x="Week", y="CombinedRiskScore",
                       markers=True, color_discrete_sequence=["#198754"],
                       labels={"CombinedRiskScore":"Combined Risk Score"})
        fig3.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — LIVE PREDICTION (API-powered)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Live Prediction":
    st.title("🧪 Live Prediction")
    st.caption("Test the backend API with custom user feature inputs")

    if not backend_online():
        st.error("⚠️ Backend API is offline. Start Akshay's FastAPI server first:\n```\nuvicorn main:app --reload\n```")
        st.stop()

    st.markdown("---")
    tab1, tab2 = st.tabs(["🔴 Anomaly Detection", "🔮 Future Risk Prediction"])

    with tab1:
        st.markdown("### Static Anomaly Detection")
        st.caption("Enter user behavioral features to detect anomalous activity")

        col1, col2, col3 = st.columns(3)
        with col1:
            total_activity = st.number_input("Total Activity", value=150.0)
            odd_hour = st.number_input("Odd Hour Activity", value=5.0)
            active_days = st.number_input("Active Days", value=20.0)
        with col2:
            email_count = st.number_input("Email Count", value=50.0)
            file_access = st.number_input("File Access Count", value=30.0)
            device_use = st.number_input("Device Usage Count", value=2.0)
        with col3:
            unique_pcs = st.number_input("Unique PCs", value=3.0)
            attachments = st.number_input("Emails with Attachments", value=10.0)
            high_recip = st.number_input("High Recipient Emails", value=2.0)

        if st.button("🔍 Detect Anomaly", type="primary"):
            features = {
                "total_activity": total_activity,
                "odd_hour_activity": odd_hour,
                "active_days": active_days,
                "email_count": email_count,
                "file_access_count": file_access,
                "device_usage_count": device_use,
                "unique_pcs": unique_pcs,
                "emails_with_attachments": attachments,
                "high_recipient_emails": high_recip
            }
            with st.spinner("Running Isolation Forest..."):
                result = api_post("/api/predict/anomaly", {"features": features})

            if result:
                c1, c2, c3 = st.columns(3)
                c1.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
                c2.metric("Risk Level", result["risk_level"])
                if result["is_anomaly"]:
                    c3.error("🚨 ANOMALOUS USER")
                else:
                    c3.success("✅ NORMAL USER")
            else:
                st.error("API call failed. Check that the backend is running.")

    with tab2:
        st.markdown("### Future Risk Prediction")
        st.caption("Predict whether a user will be high-risk next week")

        col1, col2 = st.columns(2)
        with col1:
            combined_risk = st.number_input("Combined Risk Score", value=2.5, min_value=0.0, max_value=10.0)
            total_act = st.number_input("Total Activity", value=200.0, key="fr_act")
            risk_3week = st.number_input("3-Week Avg Risk", value=3.0, min_value=0.0, max_value=10.0)
        with col2:
            file_ratio = st.number_input("File Ratio", value=0.1, min_value=0.0, max_value=1.0)
            file_act = st.number_input("File Activity", value=50.0, key="fr_file")
            risk_change = st.number_input("Risk Change", value=0.5)

        if st.button("🔮 Predict Future Risk", type="primary"):
            features = {
                "combined_risk": combined_risk,
                "total_activity": total_act,
                "behavioral_risk_3week_avg": risk_3week,
                "file_ratio": file_ratio,
                "file_activity": file_act,
                "behavioral_risk_change": risk_change
            }
            with st.spinner("Running Gradient Boosting..."):
                result = api_post("/api/predict/future-risk", {"features": features})

            if result:
                c1, c2 = st.columns(2)
                c1.metric("Risk Probability", f"{result['risk_probability']:.4f}")
                if result["will_be_insider"]:
                    c2.error("🚨 HIGH RISK NEXT WEEK")
                else:
                    c2.success("✅ LOW RISK NEXT WEEK")

                # Also get live explanation from API
                with st.spinner("Getting explanation..."):
                    expl = api_post("/api/explain/live", {"features": features})
                if expl:
                    st.info(f"**Explanation:** {expl['explanation']}")
                    st.write("**Top drivers:**", ", ".join(expl["top_drivers"]))
            else:
                st.error("API call failed. Check that the backend is running.")
