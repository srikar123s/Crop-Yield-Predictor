# =====================================================
# 🌾 Intelligent Crop Yield Prediction Dashboard
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor
import plotly.graph_objects as go

# -----------------------------------------------------
# 🌱 Load Model and Preprocessors
# -----------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = TabNetRegressor()
    model.load_model(r"C:\Users\srikar\OneDrive\Desktop\all projects\predictive\tabnet_crop_yield_model_v2.zip")

    scaler_X = joblib.load(r"C:\Users\srikar\OneDrive\Desktop\all projects\predictive\scaler_X.pkl")
    scaler_y = joblib.load(r"C:\Users\srikar\OneDrive\Desktop\all projects\predictive\scaler_y.pkl")
    label_encoders = joblib.load(r"C:\Users\srikar\OneDrive\Desktop\all projects\predictive\label_encoders.pkl")

    return model, scaler_X, scaler_y, label_encoders


# -----------------------------------------------------
# 🧠 Helper Functions
# -----------------------------------------------------
def safe_encode(label_encoder, value):
    try:
        return label_encoder.transform([value])[0]
    except Exception:
        return -1  # unseen category


def predict_yield(crop, season, state, area, production, rainfall, fertilizer, pesticide):
    model_loaded, scaler_X, scaler_y, label_encoders = load_artifacts()

    # Encode categorical values
    crop_enc = safe_encode(label_encoders['Crop'], crop)
    season_enc = safe_encode(label_encoders['Season'], season)
    state_enc = safe_encode(label_encoders['State'], state)

    # Derived features
    prod_per_area = production / (area + 1)
    rain_fert_ratio = rainfall / (fertilizer + 1)
    crop_year = 0  # Auto-filled placeholder

    input_df = pd.DataFrame([{
        'Crop_Year': crop_year,
        'Area': area,
        'Production': production,
        'Annual_Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide,
        'Prod_per_Area': prod_per_area,
        'Rainfall_Fertilizer_Ratio': rain_fert_ratio
    }])

    input_scaled = scaler_X.transform(input_df)
    input_data = np.concatenate([[crop_enc, season_enc, state_enc], input_scaled.flatten()]).reshape(1, -1)

    y_pred_scaled = model_loaded.predict(input_data)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return round(float(y_pred[0][0]), 3)


# -----------------------------------------------------
# 🎨 Streamlit Page Configuration
# -----------------------------------------------------
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        color: #2e7d32;
        font-family: "Segoe UI", sans-serif;
    }

    div.block-container {
        max-width: 900px;
        margin: auto;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h2, h3 {
        text-align: center;
        color: #1b5e20;
    }

    label {
        font-weight: 600 !important;
        color: #33691e !important;
    }

    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# 🌾 Header
# -----------------------------------------------------
st.markdown("""
<h2>🌾 Intelligent Crop Yield Predictor</h2>
<p style='text-align:center; color:gray;'>
Leverage AI-powered predictions for sustainable agriculture 🌱
</p>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------
# 📋 Input Section
# -----------------------------------------------------
st.subheader("📥 Enter Crop & Environmental Details")

_, _, _, label_encoders = load_artifacts()
crop_options = sorted(label_encoders['Crop'].classes_.tolist())
season_options = sorted(label_encoders['Season'].classes_.tolist())
state_options = sorted(label_encoders['State'].classes_.tolist())

with st.form("prediction_form"):
    st.markdown("### 🌱 Basic Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        crop = st.selectbox("Crop", crop_options)
    with col2:
        season = st.selectbox("Season", season_options)
    with col3:
        state = st.selectbox("State", state_options)

    st.markdown("---")
    st.markdown("### 🌦️ Environmental & Input Details")

    # ---- Row 1: 3 inputs ----
    colA, colB, colC = st.columns(3)
    with colA:
        area_str = st.text_input("Area (hectares)", placeholder="e.g., 2.5")
    with colB:
        rainfall_str = st.text_input("💧 Annual Rainfall (mm)", placeholder="e.g., 950")
    with colC:
        production_str = st.text_input("🌾 Production (tons)", placeholder="e.g., 4.2")

    # ---- Row 2: 2 inputs ----
    colD, colE = st.columns(2)
    with colD:
        fertilizer_str = st.text_input("🌱 Fertilizer Used (kg)", placeholder="e.g., 120")
    with colE:
        pesticide_str = st.text_input("💊 Pesticide Used (kg)", placeholder="e.g., 15")

    # Convert safely to float
    def safe_float(value):
        try:
            return float(value)
        except:
            return None

    area = safe_float(area_str)
    rainfall = safe_float(rainfall_str)
    production = safe_float(production_str)
    fertilizer = safe_float(fertilizer_str)
    pesticide = safe_float(pesticide_str)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🌿 Predict Crop Yield", use_container_width=True)

# -----------------------------------------------------
# 📊 Prediction Result
# -----------------------------------------------------
if submitted:
    if None in [area, rainfall, production, fertilizer, pesticide]:
        st.warning("⚠️ Please fill in all numeric fields before predicting.")
    else:
        with st.spinner("⏳ Running AI model to predict yield..."):
            try:
                result = predict_yield(
                    crop=crop,
                    season=season,
                    state=state,
                    area=area,
                    production=production,
                    rainfall=rainfall,
                    fertilizer=fertilizer,
                    pesticide=pesticide
                )

                st.success("✅ Prediction Complete!")

                # Metric cards
                st.markdown("### 🌾 Predicted Yield Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Yield", f"{result} tons/ha")
                col2.metric("Crop", crop)
                col3.metric("Season", season)

                
                st.markdown(f'''
                <div style="padding: 10px; border-left: 5px solid #00f; background-color: #e6f0ff; color: black;">
                🌱 Based on the entered parameters, the estimated yield for <b>{crop}</b> in <b>{state} ({season} season)</b> is <b>{result} tons/hectare</b>.
                </div>
                ''', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Prediction failed due to: {e}")

# -----------------------------------------------------
# 🌿 Footer
# -----------------------------------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Developed with ❤️ by <b>Srikar</b> | Powered by <b>TabNet</b> & <b>Streamlit</b>
</p>
""", unsafe_allow_html=True)
