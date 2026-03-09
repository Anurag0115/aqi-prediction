"""
Air Quality Index (AQI) Prediction App
A Streamlit web app for predicting AQI from pollutant readings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Page config - hide sidebar
st.set_page_config(page_title="AQI Prediction", page_icon="🌍", layout="wide", initial_sidebar_state="collapsed")

# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none}
</style>
""", unsafe_allow_html=True)

# Path to data
DATA_PATH = Path(__file__).parent / "city_day.csv"
MODEL_PATH = Path(__file__).parent / "aqi_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

# AQI bucket mapping
def get_aqi_bucket(aqi: float) -> str:
    """Convert AQI value to category."""
    if pd.isna(aqi) or aqi < 0:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["AQI"]).copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df


@st.cache_data
def prepare_features(df):
    """Prepare X, y and feature columns."""
    drop_columns = ["City", "Date", "AQI_Bucket"]
    df_clean = df.drop(columns=[c for c in drop_columns if c in df.columns])
    target = "AQI"
    features = [c for c in df_clean.columns if c != target]
    X = df_clean[features]
    y = df_clean[target]
    return X, y, features


def train_model(X_train, y_train, X_test, y_test, features):
    """Train XGBoost model and return model, scaler."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.2,
        max_depth=5,
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def main():
    st.title("🌍 Air Quality Index (AQI) Prediction")
    st.markdown("Predict AQI from pollutant measurements (PM2.5, PM10, CO, NO₂, and more).")
    st.markdown("[📦 View on GitHub](https://github.com/Anurag0115/aqi-prediction)")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Please ensure `city_day.csv` is in the same folder as the app.")
        return

    df = load_and_preprocess_data()
    X, y, features = prepare_features(df)

    # Load or train model
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        import joblib
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with st.spinner("Training model..."):
            model, scaler = train_model(X_train, y_train, X_test, y_test, features)
        import joblib
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    # --- Section 1: Manual prediction ---
    st.header("Predict AQI")

    defaults = {
        "PM2.5": 70, "PM10": 120, "NO": 15, "NO2": 25, "NOx": 25,
        "NH3": 15, "CO": 2, "SO2": 20, "O3": 30, "Benzene": 2,
        "Toluene": 5, "Xylene": 2,
    }

    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(features):
        with cols[i % 3]:
            val = defaults.get(feat, 10)
            values[feat] = st.number_input(
                feat,
                min_value=0.0,
                max_value=1000.0,
                value=float(val),
                step=1.0,
                key=feat,
            )

    if st.button("Predict AQI"):
        input_row = pd.DataFrame([values])[features]
        input_scaled = scaler.transform(input_row)
        pred = model.predict(input_scaled)[0]
        pred = max(0, pred)
        bucket = get_aqi_bucket(pred)
        st.success(f"**Predicted AQI: {pred:.1f}** — *{bucket}*")
        colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
        idx = min(int(pred / 100) if pred < 500 else 5, 5)
        st.markdown(
            f'<div style="padding:10px; border-radius:8px; background:{colors[idx]}; color:black; text-align:center;">Category: {bucket}</div>',
            unsafe_allow_html=True,
        )

    # --- Section 2: Upload dataset & Actual vs Predicted graph ---
    st.header("Enter Your Dataset")

    st.markdown(
        "Upload a CSV with pollutant columns. If it includes an **AQI** column, you'll see Actual vs Predicted graph."
    )
    st.caption("Required columns: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)

            missing = [c for c in features if c not in user_df.columns]
            if missing:
                st.error(f"Missing required columns: **{', '.join(missing)}**")
            else:
                X_user = user_df[features].copy()
                numeric_cols = X_user.select_dtypes(include=["number"]).columns
                X_user[numeric_cols] = X_user[numeric_cols].fillna(X_user[numeric_cols].mean())
                X_user = X_user.fillna(0)

                X_user_scaled = scaler.transform(X_user)
                predictions = model.predict(X_user_scaled)
                predictions = np.maximum(predictions, 0)

                has_actual = "AQI" in user_df.columns

                if has_actual:
                    actual = user_df["AQI"].values

                    # Show max 150 points for a clean, readable graph
                    max_points = 150
                    n = min(len(actual), max_points)
                    x = np.arange(1, n + 1)
                    actual_plot = actual[:n]
                    pred_plot = predictions[:n]

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(x, actual_plot, label="Actual AQI", color="#2196F3", linewidth=2)
                    ax.plot(x, pred_plot, label="Predicted AQI", color="#FF9800", linewidth=2)
                    ax.set_xlabel("Sample", fontsize=11)
                    ax.set_ylabel("AQI Value", fontsize=11)
                    ax.set_title("Actual vs Predicted AQI", fontsize=13)
                    ax.legend(loc="upper right", fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.set_xlim(1, n)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    if len(actual) > max_points:
                        st.caption(f"Showing first {max_points} of {len(actual)} samples. Download CSV for full results.")
                else:
                    result_df = pd.DataFrame({
                        "Predicted_AQI": np.round(predictions, 1),
                        "Predicted_Category": [get_aqi_bucket(p) for p in predictions],
                    })
                    st.dataframe(result_df, use_container_width=True)

                # Download
                result_df = pd.DataFrame({
                    "Predicted_AQI": np.round(predictions, 1),
                    "Predicted_Category": [get_aqi_bucket(p) for p in predictions],
                })
                if has_actual:
                    result_df["Actual_AQI"] = actual
                csv = result_df.to_csv(index=False)
                st.download_button("Download results as CSV", data=csv, file_name="aqi_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
