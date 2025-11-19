import streamlit as st
import numpy as np
import pickle
import base64
import requests
import os

# PAGE CONFIG
st.set_page_config(page_title="AI Crop Advisor", layout="centered")

# CUSTOM CSS FOR ATTRACTIVE UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 50%, #ccfbf1 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #15803d, #059669, #0d9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
    }
    
    .sub-header {
        text-align: center;
        color: #15803d;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 2px solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .crop-card {
        background: linear-gradient(135deg, #f0fdf4, #d1fae5);
        border-color: #86efac;
    }
    
    .fertilizer-card {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-color: #93c5fd;
    }
    
    .organic-card {
        background: linear-gradient(135deg, #f0fdf4, #d1fae5);
        border-color: #86efac;
    }
    
    .weather-card {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border-color: #7dd3fc;
    }
    
    .carbon-card {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border-color: #fcd34d;
    }
    
    /* Weather grid */
    .weather-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .weather-item {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #16a34a, #059669);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #15803d, #047857);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* List items */
    .list-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background: rgba(255, 255, 255, 0.6);
        border-left: 4px solid;
    }
    
    .fertilizer-item {
        border-left-color: #3b82f6;
    }
    
    .organic-item {
        border-left-color: #22c55e;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #15803d;
    }
    
    /* Nutrient label */
    .nutrient-label {
        color: #15803d;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
</style>
""", unsafe_allow_html=True)

# BACKGROUND IMAGE 
def set_bg(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Uncomment the line below if you have bg3.png in your directory
#set_bg("bg.png")

# ================================
# LOAD MODEL
# ================================
with open("final_crop_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
le_crop = data["label_encoder"]
features = data["features"]
fert_map = data["fertilizer_map"]
organic_alts = data["organic_alts"]

# ================================
# WEATHER API
# ================================
API_KEY = "279f8f1e5b0fe5fb9d73bd87b15724fd"

def get_weather(city):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        res = requests.get(url, params={"q": city, "appid": API_KEY, "units": "metric"}).json()

        temp = res["main"]["temp"]
        hum = res["main"]["humidity"]
        rain = res.get("rain", {}).get("1h", 0)
        soil_m = round((hum/100)*0.6 + (rain*0.4), 3)

        return temp, hum, rain, soil_m
    except:
        return None, None, None, None

# ================================
# FEATURE ENGINEERING
# ================================
def make_features(N,P,K,pH,temp,rain):
    total = N + P + K
    npk = N/(P+K+1e-6)
    ph_class = 0 if pH<=5.5 else (1 if pH<=7 else 2)

    row = []
    for f in features:
        if f=="Nitrogen": row.append(N)
        elif f=="Phosphorus": row.append(P)
        elif f=="Potassium": row.append(K)
        elif f=="pH": row.append(pH)
        elif f=="Temperature": row.append(temp)
        elif f=="Rainfall": row.append(rain)
        elif f=="Total_Nutrients": row.append(total)
        elif f=="NPK_Ratio": row.append(npk)
        elif f=="pH_Class": row.append(ph_class)
        else: row.append(0)
    return np.array(row).reshape(1,-1)

# ================================
# PAGE HEADER
# ================================
st.markdown("<h1 class='main-header'>🌿 EcoFarm Advisor </h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Crop & Fertilizer Recommendation System</h2>", unsafe_allow_html=True)

# ================================
# INPUT SECTION
# ================================
st.markdown("<div class='input-section'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #15803d; margin-bottom: 1.5rem;'>📋 Enter Soil & Location Details</h3>", unsafe_allow_html=True)

# Soil Nutrients
st.markdown("<p class='nutrient-label' style='font-size: 1.1rem; margin-top: 1rem;'>🌱 Soil Nutrients</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='nutrient-label'>Nitrogen (N)</p>", unsafe_allow_html=True)
    N = st.number_input("N", min_value=0, max_value=300, value=None, label_visibility="collapsed", key="nitrogen")

with col2:
    st.markdown("<p class='nutrient-label'>Phosphorus (P)</p>", unsafe_allow_html=True)
    P = st.number_input("P", min_value=0, max_value=300, value=None, label_visibility="collapsed", key="phosphorus")

with col3:
    st.markdown("<p class='nutrient-label'>Potassium (K)</p>", unsafe_allow_html=True)
    K = st.number_input("K", min_value=0, max_value=300, value=None, label_visibility="collapsed", key="potassium")

# Soil Properties
st.markdown("<p style='margin-top: 1.5rem;'></p>", unsafe_allow_html=True)

col4, col5 = st.columns(2)

with col4:
    pH = st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, value=None, step=0.1)

with col5:
    soil_type = st.selectbox("🏞️ Soil Type", ["", "Black", "Red", "Clay", "Alluvial", "Sandy"])

# Location
city = st.text_input("📍 City / District", placeholder="Enter your city name...")

st.markdown("</div>", unsafe_allow_html=True)

# Predict Button
predict_btn = st.button("🌾 Get Recommendation")

# ================================
# PREDICTION
# ================================
if predict_btn:
    # Validation
    if not city.strip():
        st.error("❌ Please enter a city name")
    elif not soil_type:
        st.error("❌ Please select a soil type")
    elif N is None or P is None or K is None or pH is None:
        st.error("❌ Please fill in all nutrient values and pH")
    else:
        with st.spinner("🔄 Analyzing your inputs..."):
            temp, hum, rain, soil_moist = get_weather(city)
            if temp is None:
                st.error("❌ Could not fetch weather data. Please check the city name.")
                st.stop()

            X = make_features(N,P,K,pH,temp,rain)
            pred = model.predict(X)[0]
            crop = le_crop.inverse_transform([pred])[0]

            ferts = fert_map.get(crop.lower(), ["No data"])
            carbon = round(N*6.3 + P*1.5 + K*0.8, 2)

            # Display Results
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            
            # Recommended Crop
            st.markdown("""
            <div class='result-card crop-card'>
                <div class='section-header'>🌾 Recommended Crop</div>
                <div style='text-align: center; font-size: 2.5rem; font-weight: 700; color: #15803d; padding: 1rem;'>
                    {}
                </div>
            </div>
            """.format(crop.upper()), unsafe_allow_html=True)
            
            # Fertilizer Suggestions
            st.markdown("""
            <div class='result-card fertilizer-card'>
                <div class='section-header' style='color: #1e40af;'>🧪 Fertilizer Suggestions</div>
            """, unsafe_allow_html=True)
            
            for f in ferts:
                st.markdown(f"<div class='list-item fertilizer-item'>• {f}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Organic Alternatives
            st.markdown("""
            <div class='result-card organic-card'>
                <div class='section-header'>🌿 Organic Alternatives</div>
            """, unsafe_allow_html=True)
            
            for group in organic_alts.values():
                for alt in group:
                    st.markdown(f"<div class='list-item organic-item'>− {alt}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Weather Information
            st.markdown("""
            <div class='result-card weather-card'>
                <div class='section-header' style='color: #0369a1;'>⛅ Weather Conditions</div>
                <div class='weather-grid'>
                    <div class='weather-item' style='background: #fef3c7; border-color: #fbbf24;'>
                        <div style='font-size: 2rem;'>🌡️</div>
                        <div style='color: #92400e; font-size: 0.9rem; margin-top: 0.5rem;'>Temperature</div>
                        <div style='font-weight: 600; color: #78350f; font-size: 1.2rem;'>{} °C</div>
                    </div>
                    <div class='weather-item' style='background: #dbeafe; border-color: #60a5fa;'>
                        <div style='font-size: 2rem;'>💧</div>
                        <div style='color: #1e3a8a; font-size: 0.9rem; margin-top: 0.5rem;'>Humidity</div>
                        <div style='font-weight: 600; color: #1e40af; font-size: 1.2rem;'>{} %</div>
                    </div>
                    <div class='weather-item' style='background: #e0f2fe; border-color: #7dd3fc;'>
                        <div style='font-size: 2rem;'>🌧️</div>
                        <div style='color: #0c4a6e; font-size: 0.9rem; margin-top: 0.5rem;'>Rainfall</div>
                        <div style='font-weight: 600; color: #075985; font-size: 1.2rem;'>{} mm</div>
                    </div>
                    <div class='weather-item' style='background: #ccfbf1; border-color: #5eead4;'>
                        <div style='font-size: 2rem;'>💦</div>
                        <div style='color: #134e4a; font-size: 0.9rem; margin-top: 0.5rem;'>Soil Moisture</div>
                        <div style='font-weight: 600; color: #115e59; font-size: 1.2rem;'>{}</div>
                    </div>
                </div>
            </div>
            """.format(temp, hum, rain, soil_moist), unsafe_allow_html=True)
            
            # Carbon Emission
            st.markdown("""
            <div class='result-card carbon-card'>
                <div class='section-header' style='color: #92400e;'>🏭 Carbon Emission Estimate</div>
                <div style='text-align: center; padding: 1.5rem; background: rgba(255, 255, 255, 0.5); border-radius: 0.5rem; border: 2px solid #fbbf24;'>
                    <div style='font-size: 2.5rem; font-weight: 700; color: #92400e;'>{}</div>
                    <div style='color: #78350f; font-size: 1rem; margin-top: 0.5rem;'>kg CO₂e / ha</div>
                </div>
            </div>
            """.format(carbon), unsafe_allow_html=True)

