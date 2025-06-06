import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Set page configuration
st.set_page_config(page_title="ML Comparison & Dynamic Visualizer", layout="wide")

# Function for ML Model Evaluation
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

# Comparative Analysis Page
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
                fig, ax = plt.subplots(figsize=(1.8, 1.2))  # Smaller graph size
                ax.bar(results.keys(), [results[model][metric] for model in results], color=['skyblue', 'orange', 'green'])
                ax.set_title(f'{metric}', fontsize=8)
                ax.set_ylabel(metric, fontsize=6)
                ax.tick_params(axis='x', labelrotation=10, labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                fig.tight_layout(pad=0.5)
                st.pyplot(fig)

# Dynamic Visualization Page
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
            # Plot for File 1
            fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))  # Smaller graph size
            ax1.plot(df1[col1_selected], label=f'{col1_selected}', color='blue', linewidth=1)
            ax1.set_title(f"{col1_selected}", fontsize=9)
            ax1.legend(fontsize=7)
            ax1.tick_params(axis='both', labelsize=6)
            fig1.tight_layout(pad=0.5)
            st.pyplot(fig1)

            # Plot for File 2
            fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))  # Smaller graph size
            ax2.plot(df2[col2_selected], label=f'{col2_selected}', color='green', linewidth=1)
            ax2.set_title(f"{col2_selected}", fontsize=9)
            ax2.legend(fontsize=7)
            ax2.tick_params(axis='both', labelsize=6)
            fig2.tight_layout(pad=0.5)
            st.pyplot(fig2)

# Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Comparative Analysis", "Dynamic Visualization"])

if page == "Comparative Analysis":
    comparative_analysis()
else:
    dynamic_visualization()
