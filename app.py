import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="ML Comparison & Dynamic Visualizer", layout="wide")

# --- Function for ML Model Evaluation ---
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'R2 Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }

    return results


# --- Comparative Analysis Page ---
def comparative_analysis():
    st.header("📊 Comparative Analysis of ML Models")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df.head())

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Dataset must have at least 2 numeric columns.")
            return

        target_col = st.selectbox("Select Target Column", numeric_cols)
        feature_cols = st.multiselect("Select Feature Columns", [col for col in numeric_cols if col != target_col])

        if target_col and feature_cols:
            X = df[feature_cols]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results = evaluate_models(X_train, X_test, y_train, y_test)

            # Displaying metrics
            st.subheader("Model Performance Comparison")

            metrics = ['R2 Score', 'MAE', 'RMSE']
            for metric in metrics:
                fig, ax = plt.subplots()
                ax.bar(results.keys(), [results[model][metric] for model in results])
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                st.pyplot(fig)


# --- Dynamic Visualization Page ---
def dynamic_visualization():
    st.header("📈 Dynamic CSV File Comparison")

    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader("Upload First CSV File", type=["csv"], key="file1")

    with col2:
        file2 = st.file_uploader("Upload Second CSV File", type=["csv"], key="file2")

    if file1 and file2:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        st.subheader("📄 File 1 Preview")
        st.dataframe(df1.head())

        st.subheader("📄 File 2 Preview")
        st.dataframe(df2.head())

        numeric_cols1 = df1.select_dtypes(include=['number']).columns.tolist()
        numeric_cols2 = df2.select_dtypes(include=['number']).columns.tolist()

        st.subheader("📌 Select Columns to Compare")

        col1_selected = st.selectbox("Select Column from File 1", numeric_cols1)
        col2_selected = st.selectbox("Select Column from File 2", numeric_cols2)

        if col1_selected and col2_selected:
            fig, ax = plt.subplots()
            ax.plot(df1[col1_selected], label=f'File 1: {col1_selected}')
            ax.plot(df2[col2_selected], label=f'File 2: {col2_selected}')
            ax.set_title("Comparison of Selected Columns")
            ax.legend()
            st.pyplot(fig)


# --- Navigation ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Comparative Analysis", "Dynamic Visualization"])

if page == "Comparative Analysis":
    comparative_analysis()
else:
    dynamic_visualization()
