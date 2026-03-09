"""
Air Quality Index (AQI) Prediction App
A Streamlit web app for predicting AQI from pollutant readings.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(page_title="AQI Prediction", page_icon="🌍", layout="wide")

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
    """Train XGBoost model and return model, scaler, metrics."""
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
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, scaler, {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    st.title("🌍 Air Quality Index (AQI) Prediction")
    st.markdown(
        "Predict AQI from pollutant measurements (PM2.5, PM10, CO, NO₂, and more)."
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Please ensure `city_day.csv` is in the same folder as the app.")
        return

    # Sidebar
    st.sidebar.header("Navigation")
    st.sidebar.markdown("[📦 View on GitHub](https://github.com/Anurag0115/aqi-prediction)")
    page = st.sidebar.radio(
        "Choose a page",
        ["🔮 Predict AQI", "📊 Data Exploration", "🤖 Model Info"],
    )

    # Load data
    df = load_and_preprocess_data()
    X, y, features = prepare_features(df)

    if page == "🔮 Predict AQI":
        st.header("Predict AQI from Pollutant Values")

        # Load or train model
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            import joblib
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            st.cache_data.clear()
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            with st.spinner("Training model..."):
                model, scaler, metrics = train_model(
                    X_train, y_train, X_test, y_test, features
                )
            import joblib
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)

        # Input sliders (use typical ranges from dataset)
        defaults = {
            "PM2.5": 70, "PM10": 120, "NO": 15, "NO2": 25, "NOx": 25,
            "NH3": 15, "CO": 2, "SO2": 20, "O3": 30, "Benzene": 2,
            "Toluene": 5, "Xylene": 2,
        }

        st.subheader("Enter pollutant values (µg/m³)")
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
            pred = max(0, pred)  # AQI should be non-negative

            bucket = get_aqi_bucket(pred)
            st.success(f"**Predicted AQI: {pred:.1f}** — *{bucket}*")

            # Color bar
            colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
            buckets = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
            idx = min(int(pred / 100) if pred < 500 else 5, 5)
            st.markdown(
                f'<div style="padding:10px; border-radius:8px; background:{colors[idx]}; color:black; text-align:center;">'
                f"Category: {bucket}</div>",
                unsafe_allow_html=True,
            )

    elif page == "📊 Data Exploration":
        st.header("Data Exploration")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total records", f"{len(df):,}")
            st.metric("Features", len(features))
        with col2:
            st.metric("Mean AQI", f"{y.mean():.1f}")
            st.metric("Date range", f"{df['Date'].min()} → {df['Date'].max()}")

        st.subheader("Sample data")
        display_cols = ["City", "Date"] + features[:6] + ["AQI"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols].head(20).style.background_gradient(subset=["AQI"], cmap="RdYlGn_r"))

    else:
        st.header("Model Info")
        st.markdown("""
        - **Model:** XGBoost Regressor  
        - **Best params:** learning_rate=0.2, max_depth=5, n_estimators=100  
        - **Metrics:** MAE, RMSE, R²  
        - **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene  
        """)
        if X is not None and len(X) > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model, scaler, metrics = train_model(X_train, y_train, X_test, y_test, features)
            st.metric("MAE", f"{metrics['MAE']:.2f}")
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            st.metric("R² Score", f"{metrics['R2']:.2f}")


if __name__ == "__main__":
    main()
