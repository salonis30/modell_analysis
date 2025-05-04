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

# Sidebar
st.sidebar.title("Select Page")
page = st.sidebar.radio("", ["🔴 Static Comparative Analysis", "⚙️ Dynamic Data Visualisation"])

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

# STATIC PAGE
if "Static" in page:
    st.subheader("📁 Upload Dataset for Static Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="static_file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        target_col = st.selectbox("🎯 Select the Target Column", df.columns)
        feature_cols = st.multiselect("🧮 Select Feature Columns", df.columns.drop(target_col))

        if feature_cols:
            X = df[feature_cols]
            y = df[target_col]

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor()
            }

            results = evaluate_models(X, y, models)

            st.subheader("📊 Model Comparison Results")
            result_df = pd.DataFrame(results).T
            st.dataframe(result_df)

            st.subheader("📈 Metric Comparison Graph")
            metric = st.selectbox("Select Metric", result_df.columns)

            fig, ax = plt.subplots()
            ax.bar(result_df.index, result_df[metric], color="skyblue")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} Comparison")
            st.pyplot(fig)
        else:
            st.warning("Please select at least one feature column.")
    else:
        st.info("Upload a dataset to begin analysis.")

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
