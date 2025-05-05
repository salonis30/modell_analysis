import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Oil & Gas ML Comparative Dashboard", layout="wide")
st.title("📊 Oil & Gas ML Comparative Dashboard")

# Sidebar Page Selector
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🔴 Static Comparative Analysis", "⚙️ Dynamic Data Visualisation"])

# Load predefined dataset
@st.cache_data
def load_dataset():
    return pd.read_csv("oil_gas_production_india.csv")  # No "data/" folder needed


# Common ML function
def evaluate_models(X, y, models):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "R2 Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    return results

# STATIC PAGE (Without Upload)
if "Static" in page:
    st.subheader("📊 Comparative Analysis on Preloaded Dataset")
    df = load_dataset()

    st.sidebar.subheader("🎯 Select Target Column")
    target_col = st.sidebar.selectbox("Target Column", df.columns)

    st.sidebar.subheader("🧮 Select Feature Columns")
    feature_cols = st.sidebar.multiselect("Feature Columns", df.columns.drop(target_col))

    st.sidebar.subheader("📈 Select Chart Type")
    chart_type = st.sidebar.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor()
        }

        results = evaluate_models(X, y, models)

        result_df = pd.DataFrame(results).T
        st.subheader("📊 Model Performance Metrics")
        st.dataframe(result_df)

        st.subheader(f"📉 {chart_type} of Selected Metric")
        metric = st.selectbox("Metric to Visualize", result_df.columns)

        fig, ax = plt.subplots()
        if chart_type == "Bar Chart":
            ax.bar(result_df.index, result_df[metric], color='skyblue')
            ax.set_ylabel(metric)
        elif chart_type == "Line Chart":
            ax.plot(result_df.index, result_df[metric], marker='o', linestyle='-', color='orange')
            ax.set_ylabel(metric)
        elif chart_type == "Pie Chart":
            ax.pie(result_df[metric], labels=result_df.index, autopct='%1.1f%%')
            ax.axis("equal")
        ax.set_title(f"{metric} - {chart_type}")
        st.pyplot(fig)
    else:
        st.warning("Select at least one feature column to proceed.")


# DYNAMIC PAGE
if "Dynamic" in page:
    st.subheader("📁 Upload Two CSV Files for Side-by-Side Visualisation")
    file1 = st.file_uploader("Upload First CSV", type=["csv"], key="file1")
    file2 = st.file_uploader("Upload Second CSV", type=["csv"], key="file2")

    if file1 is not None and file2 is not None:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        col1, col2 = st.columns(2)
        with col1:
            st.write("📄 **First Dataset Preview**")
            st.dataframe(df1.head())
        with col2:
            st.write("📄 **Second Dataset Preview**")
            st.dataframe(df2.head())
    else:
        st.info("Upload two datasets for dynamic comparison.")     
