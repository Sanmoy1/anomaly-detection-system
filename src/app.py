import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add current directory to path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detect_anomalies import run_models
from utils import clean, feature_engineer

st.set_page_config(page_title="Anomaly Detection System", layout="wide")

st.title("🕵️‍♂️ Fraud & Anomaly Detection Dashboard")
st.markdown("""
This dashboard allows you to analyze transaction data for anomalies using **Isolation Forest** and **Local Outlier Factor (LOF)**.
""")

# Sidebar
st.sidebar.header("Configuration")
contamination = st.sidebar.slider("Contamination (Expected % of Outliers)", 0.001, 0.1, 0.02, step=0.001)
uploaded_file = st.sidebar.file_uploader("Upload Transactions CSV", type=["csv"])

# Load Data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["date"])
    return df

# Helper to load default if no upload
default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "transactions.csv")

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.sidebar.success("Loaded uploaded file!")
elif os.path.exists(default_path):
    df_raw = load_data(default_path)
    st.sidebar.info(f"Using default dataset: {default_path}")
else:
    st.error("No data found. Please upload a CSV or generate data first.")
    st.stop()

# Pipeline
st.subheader("1. Data Pipeline Execution")

if st.button("Run Anomaly Detection"):
    with st.spinner("Cleaning and Feature Engineering..."):
        df_clean = clean(df_raw)
        df_features = feature_engineer(df_clean)
    
    with st.spinner(f"Training Models with contamination={contamination}..."):
        iso_labels, iso_score, lof_labels, lof_score, z_flags = run_models(df_features, contamination=contamination)
        
        # Assemble Results
        out = df_features.copy()
        out["iso_label"] = (iso_labels == -1).astype(int)
        out["lof_label"] = (lof_labels == -1).astype(int)
        out["zscore_label"] = z_flags.astype(int)
        out["votes"] = out[["iso_label", "lof_label", "zscore_label"]].sum(axis=1)
        
        # Normalize scores for 'Severity'
        s_iso = (iso_score - iso_score.min()) / max(1e-9, (iso_score.max() - iso_score.min()))
        s_lof = (lof_score - lof_score.min()) / max(1e-9, (lof_score.max() - lof_score.min()))
        out["severity"] = (s_iso + s_lof) / 2.0
    
    st.success("Analysis Complete!")
    
    # Visualizations
    st.divider()
    st.subheader("2. Dashboard Overview")
    
    col1, col2, col3 = st.columns(3)
    total_tx = len(out)
    n_anomalies = len(out[out["votes"] >= 2])
    pct_anomalies = (n_anomalies / total_tx) * 100
    
    col1.metric("Total Transactions", f"{total_tx:,}")
    col2.metric("High Confidence Anomalies", f"{n_anomalies:,}", delta_color="inverse")
    col3.metric("Anomaly Rate", f"{pct_anomalies:.2f}%")
    
    # Interactive Plot
    st.subheader("3. Time Series Analysis")
    
    # Plot anomalies on the time series
    fig = px.scatter(out, x="date", y="amount", color="severity", 
                     title="Transactions by Amount & Severity",
                     hover_data=["customer_id", "category", "votes"],
                     color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Anomalies Table
    st.subheader("4. Top Suspicious Transactions (Voting >= 2)")
    anomalies = out[out["votes"] >= 2].sort_values("severity", ascending=False).head(50)
    st.dataframe(anomalies.style.background_gradient(subset=["severity"], cmap="Reds"))
    
    # Category Breakdown
    st.subheader("5. Anomalies by Category")
    cat_counts = anomalies["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig_bar = px.bar(cat_counts, x="Category", y="Count", color="Count", title="Count of Anomalies per Category")
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("Click the button above to start the analysis.")
