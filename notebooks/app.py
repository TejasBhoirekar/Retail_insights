# Retail Customer Segmentation Dashboard

# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Retail Customer Segmentation",
    layout="wide",
    page_icon="🛍️"
)

st.title("🛍️ Retail Customer Segmentation Dashboard")
st.markdown("""
Analyze customer purchasing behavior in real time using **RFM Analysis** and **KMeans Clustering**.  
Upload your retail dataset (`retail_dataset.csv`) to view live segmentation results.
""")


# FILE UPLOAD
uploaded_file = st.file_uploader("📂 Upload your retail dataset CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    st.success("Dataset uploaded successfully!")

    # DATA CLEANING
    st.header("🧹 Data Cleaning")

    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Display quick metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Total Orders", df["InvoiceNo"].nunique())
    col2.metric("👥 Unique Customers", df["CustomerID"].nunique())
    col3.metric("💰 Total Revenue", f"${df['Revenue'].sum():,.2f}")

    st.dataframe(df.head(10))

    # RFM ANALYSIS
    st.header("💰 Customer Segmentation using RFM")

    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Revenue": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    rfm = rfm[rfm["Monetary"] > 0].dropna(subset=["Recency", "Frequency", "Monetary"])

    # RFM SCORING
    try:
        rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5,4,3,2,1])
        rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
        rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5])

        # Safe conversion (handles NaN gracefully)
        for col in ["R_score", "F_score", "M_score"]:
            rfm[col] = rfm[col].astype(float).fillna(0).astype(int)
    except ValueError:
        st.warning("⚠️ Not enough unique data points to create 5 quantile bins — using fewer bins.")
        n_bins = min(3, len(rfm))
        rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), n_bins, labels=range(n_bins, 0, -1))
        rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
        rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
        for col in ["R_score", "F_score", "M_score"]:
            rfm[col] = rfm[col].astype(float).fillna(0).astype(int)

    # Combine RFM metrics
    rfm["RFM_Sum"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    def label_customer(row):
        if row["RFM_Sum"] >= 13:
            return "Loyal"
        elif row["RFM_Sum"] >= 9:
            return "Potential"
        elif row["RFM_Sum"] >= 6:
            return "At Risk"
        else:
            return "Need Attention"

    rfm["Segment"] = rfm.apply(label_customer, axis=1)

    st.subheader("📊 RFM Segmentation Summary")
    st.dataframe(rfm[["CustomerID", "Recency", "Frequency", "Monetary", "RFM_Sum", "Segment"]].head(10))

    seg_summary = rfm["Segment"].value_counts().reset_index()
    seg_summary.columns = ["Segment", "Count"]
    st.bar_chart(seg_summary.set_index("Segment"))

    
    # KMEANS CLUSTERING
    st.header("🤖 KMeans Clustering (Live)")

    k_value = st.slider("Select number of clusters (K)", 2, 10, 4)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=50)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    st.write(f"KMeans clustering complete with **K = {k_value}**")

    # Cluster visualization
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    scatter = ax1.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"])
    ax1.set_xlabel("Recency (days)")
    ax1.set_ylabel("Monetary Value")
    ax1.set_title(f"KMeans Clusters (K={k_value}) - Recency vs Monetary")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    scatter = ax2.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Monetary Value")
    ax2.set_title(f"KMeans Clusters (K={k_value}) - Frequency vs Monetary")
    st.pyplot(fig2)

    
    # BUSINESS INSIGHTS
    st.header("💡 Business Insights & Recommendations")

    st.markdown("""
    **Key Insights:**
    - 🏆 **Loyal Customers**: Spend the most and purchase often — reward them with loyalty programs and early access sales.  
    - ⚠️ **At Risk Customers**: Used to buy frequently but have become inactive — send reactivation or “We miss you” campaigns.  
    - 🚀 **Potential Customers**: Regular, growing buyers — ideal for upselling and cross-selling.  
    - 🌱 **Need Attention**: New or low-engagement buyers — nurture them with welcome offers and next-purchase discounts.

    **Next Campaign Target:** 🎯 *Potential Customers* — easiest to convert into Loyal customers with personalized marketing.
    """)

else:
    st.info("👈 Please upload your `retail_dataset.csv` file to start live segmentation.")