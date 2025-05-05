import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit page setup
st.set_page_config(page_title="Oil & Gas ML Dashboard", layout="wide")

# --- Load dataset ---
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/oil_gas_production_india.csv")
    except FileNotFoundError:
        return None

# --- Evaluate model ---
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "R2 Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions, squared=False)
    }

# --- Train and compare models ---
def model_comparison(df, target_col):
    st.subheader("📈 ML Model Comparison")

    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Show metrics table
    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

    # Plot metrics
    st.subheader("📊 Model Metrics Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Main UI ---
st.title("📊 Oil & Gas Production ML Dashboard")
st.markdown("Compare machine learning models for predicting oil and gas production.")

# Load default dataset if exists
df = load_dataset()

# If not found, ask user to upload
if df is None:
    st.warning("⚠️ Default dataset not found. Please upload your dataset.")
    uploaded_file = st.file_uploader("📤 Upload your Oil & Gas CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# If dataset loaded
if df is not None:
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
    else:
        target = st.selectbox("🎯 Select Target Column for Prediction", options=numeric_cols)
        model_comparison(df[numeric_cols], target)

else:
    st.stop()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit page setup
st.set_page_config(page_title="Oil & Gas ML Dashboard", layout="wide")

# --- Load dataset ---
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/oil_gas_production_india.csv")
    except FileNotFoundError:
        return None

# --- Evaluate model ---
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "R2 Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions, squared=False)
    }

# --- Train and compare models ---
def model_comparison(df, target_col):
    st.subheader("📈 ML Model Comparison")

    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Show metrics table
    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

    # Plot metrics
    st.subheader("📊 Model Metrics Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Main UI ---
st.title("📊 Oil & Gas Production ML Dashboard")
st.markdown("Compare machine learning models for predicting oil and gas production.")

# Load default dataset if exists
df = load_dataset()

# If not found, ask user to upload
if df is None:
    st.warning("⚠️ Default dataset not found. Please upload your dataset.")
    uploaded_file = st.file_uploader("📤 Upload your Oil & Gas CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# If dataset loaded
if df is not None:
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
    else:
        target = st.selectbox("🎯 Select Target Column for Prediction", options=numeric_cols)
        model_comparison(df[numeric_cols], target)

else:
    st.stop()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit page setup
st.set_page_config(page_title="Oil & Gas ML Dashboard", layout="wide")

# --- Load dataset ---
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/oil_gas_production_india.csv")
    except FileNotFoundError:
        return None

# --- Evaluate model ---
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "R2 Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions, squared=False)
    }

# --- Train and compare models ---
def model_comparison(df, target_col):
    st.subheader("📈 ML Model Comparison")

    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Show metrics table
    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

    # Plot metrics
    st.subheader("📊 Model Metrics Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Main UI ---
st.title("📊 Oil & Gas Production ML Dashboard")
st.markdown("Compare machine learning models for predicting oil and gas production.")

# Load default dataset if exists
df = load_dataset()

# If not found, ask user to upload
if df is None:
    st.warning("⚠️ Default dataset not found. Please upload your dataset.")
    uploaded_file = st.file_uploader("📤 Upload your Oil & Gas CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# If dataset loaded
if df is not None:
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
    else:
        target = st.selectbox("🎯 Select Target Column for Prediction", options=numeric_cols)
        model_comparison(df[numeric_cols], target)

else:
    st.stop()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit page setup
st.set_page_config(page_title="Oil & Gas ML Dashboard", layout="wide")

# --- Load dataset ---
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/oil_gas_production_india.csv")
    except FileNotFoundError:
        return None

# --- Evaluate model ---
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "R2 Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions, squared=False)
    }

# --- Train and compare models ---
def model_comparison(df, target_col):
    st.subheader("📈 ML Model Comparison")

    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Show metrics table
    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

    # Plot metrics
    st.subheader("📊 Model Metrics Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Main UI ---
st.title("📊 Oil & Gas Production ML Dashboard")
st.markdown("Compare machine learning models for predicting oil and gas production.")

# Load default dataset if exists
df = load_dataset()

# If not found, ask user to upload
if df is None:
    st.warning("⚠️ Default dataset not found. Please upload your dataset.")
    uploaded_file = st.file_uploader("📤 Upload your Oil & Gas CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# If dataset loaded
if df is not None:
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
    else:
        target = st.selectbox("🎯 Select Target Column for Prediction", options=numeric_cols)
        model_comparison(df[numeric_cols], target)

else:
    st.stop()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit page setup
st.set_page_config(page_title="Oil & Gas ML Dashboard", layout="wide")

# --- Load dataset ---
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/oil_gas_production_india.csv")
    except FileNotFoundError:
        return None

# --- Evaluate model ---
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "R2 Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions, squared=False)
    }

# --- Train and compare models ---
def model_comparison(df, target_col):
    st.subheader("📈 ML Model Comparison")

    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Show metrics table
    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

    # Plot metrics
    st.subheader("📊 Model Metrics Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Main UI ---
st.title("📊 Oil & Gas Production ML Dashboard")
st.markdown("Compare machine learning models for predicting oil and gas production.")

# Load default dataset if exists
df = load_dataset()

# If not found, ask user to upload
if df is None:
    st.warning("⚠️ Default dataset not found. Please upload your dataset.")
    uploaded_file = st.file_uploader("📤 Upload your Oil & Gas CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# If dataset loaded
if df is not None:
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
    else:
        target = st.selectbox("🎯 Select Target Column for Prediction", options=numeric_cols)
        model_comparison(df[numeric_cols], target)

else:
    st.stop()
