import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(layout="wide", page_title="ML Model Comparison Dashboard")

st.title("📊 Oil & Gas ML Comparative Dashboard")

# Sidebar page selection
page = st.sidebar.radio("Select Page", ["Static Comparative Analysis", "Dynamic Data Visualisation"])

# --------------------------------------
# PAGE 1: Static Comparative Analysis
# --------------------------------------
if page == "Static Comparative Analysis":
    st.header("🔍 Static: Comparative Analysis of ML Models")
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📁 Data Preview")
        st.dataframe(df.head())

        target_col = st.selectbox("🎯 Select the Target Column for Prediction", df.columns)
        feature_cols = st.multiselect("🧮 Select Feature Columns", df.columns.drop(target_col))

        if st.button("Run Comparison"):
            if not feature_cols:
                st.warning("Please select at least one feature column.")
            else:
                X = df[feature_cols]
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
                }

                results = {"Model": [], "R2 Score": [], "MAE": [], "RMSE": []}

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    results["Model"].append(name)
                    results["R2 Score"].append(r2_score(y_test, preds))
                    results["MAE"].append(mean_absolute_error(y_test, preds))
                    results["RMSE"].append(mean_squared_error(y_test, preds, squared=False))

                result_df = pd.DataFrame(results)
                st.subheader("📈 Model Performance Comparison")
                st.dataframe(result_df)

                # Plotting
                st.subheader("📊 Visual Comparison")
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                sns.barplot(x="Model", y="R2 Score", data=result_df, ax=ax[0])
                sns.barplot(x="Model", y="MAE", data=result_df, ax=ax[1])
                sns.barplot(x="Model", y="RMSE", data=result_df, ax=ax[2])
                ax[0].set_title("R2 Score Comparison")
                ax[1].set_title("Mean Absolute Error")
                ax[2].set_title("Root Mean Squared Error")
                st.pyplot(fig)

# --------------------------------------
# PAGE 2: Dynamic Data Visualisation
# --------------------------------------
elif page == "Dynamic Data Visualisation":
    st.header("📂 Dynamic: Visualisation of Two CSV Files")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload First CSV")
        file1 = st.file_uploader("Upload First File", type=["csv"], key="file1")

    with col2:
        st.subheader("Upload Second CSV")
        file2 = st.file_uploader("Upload Second File", type=["csv"], key="file2")

    if file1 and file2:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        st.subheader("📝 First File Preview")
        st.dataframe(df1.head())

        st.subheader("📝 Second File Preview")
        st.dataframe(df2.head())
