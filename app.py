import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Title of the app
st.title("Oil and Gas Production Prediction Dashboard")
st.markdown("This dashboard allows you to analyze and predict oil and gas production data using machine learning models. The available model is Linear Regression.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("lr_model_oil.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Initialize session state for model
if "lr_model_oil" not in st.session_state:
    st.session_state.lr_model_oil = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dynamic Prediction", "Static Comparison"])

if page == "Static Comparison":
    # Static page: File Upload and Data Comparison
    st.header("Static Data Comparison")
    st.markdown("""
    In this section, you can upload two CSV files, and we will compare their contents.
    """)
    
    uploaded_file_1 = st.file_uploader("Upload the first file", type=["csv"])
    uploaded_file_2 = st.file_uploader("Upload the second file", type=["csv"])
    
    if uploaded_file_1 and uploaded_file_2:
        try:
            # Read the uploaded CSV files
            df1 = pd.read_csv(uploaded_file_1)
            df2 = pd.read_csv(uploaded_file_2)
            
            # Show basic info
            st.write("**First File Preview**")
            st.write(df1.head())
            
            st.write("**Second File Preview**")
            st.write(df2.head())
            
            # Data Comparison (example: comparing common columns)
            common_columns = df1.columns.intersection(df2.columns)
            st.write(f"**Common columns in both datasets**: {common_columns}")
            
            # Display comparison results (Basic stats)
            st.write("**Statistical Summary of the First Dataset**")
            st.write(df1.describe())
            
            st.write("**Statistical Summary of the Second Dataset**")
            st.write(df2.describe())
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
    
elif page == "Dynamic Prediction":
    # Dynamic page: Prediction using the trained model
    st.header("Dynamic Prediction")
    
    # Main page components for prediction
    st.markdown("""
    Enter the following details to predict oil production using the trained model.
    """)
    
    # Input Fields for Prediction on the main page
    year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
    capex = st.number_input("Capex", min_value=0, step=1)
    employees = st.number_input("Employees", min_value=0, step=1)
    rig_count = st.number_input("Rig Count", min_value=0, step=1)
    price_per_barrel = st.number_input("Price per Barrel", min_value=0.0, step=0.1)
    
    # Prediction logic
    if st.session_state.lr_model_oil:
        model = st.session_state.lr_model_oil
        if hasattr(model, "coef_"):  # Check if the model is fitted
            if st.button("Predict Oil Production"):
                try:
                    # Prediction using the model
                    pred_oil = model.predict([[year, capex, employees, rig_count, price_per_barrel]])[0]
                    st.write(f"Predicted Oil Production: {pred_oil:.2f} barrels")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.error("The model is not fitted yet.")
    else:
        st.error("The model has not been loaded properly.")

