import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
#from keras.models import load_model


# Load model and tools
model = load_model("weather_nn_model.h5")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🌦 Weather Prediction App")

st.write("Enter weather parameters to predict weather type")

# Numeric Inputs
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=60.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
precipitation = st.number_input("Precipitation (%)", value=20.0)
pressure = st.number_input("Atmospheric Pressure", value=1012.0)
uv_index = st.number_input("UV Index", value=5.0)
visibility = st.number_input("Visibility (km)", value=10.0)

# Categorical Inputs
cloud_cover = st.selectbox(
    "Cloud Cover",
    encoders["Cloud Cover"].classes_
)

season = st.selectbox(
    "Season",
    encoders["Season"].classes_
)

location = st.selectbox(
    "Location",
    encoders["Location"].classes_
)

# Predict Button
if st.button("Predict Weather"):

    # Encode categorical values
    cloud_encoded = encoders["Cloud Cover"].transform([cloud_cover])[0]
    season_encoded = encoders["Season"].transform([season])[0]
    location_encoded = encoders["Location"].transform([location])[0]

    # Create feature array
    features = np.array([[
        temperature,
        humidity,
        wind_speed,
        precipitation,
        cloud_encoded,  # ✅ moved here
        pressure,
        uv_index,
        season_encoded,
        visibility,
        location_encoded
    ]], dtype=np.float32)

    # Scale input
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction)

    # Decode result
    weather = encoders["Weather Type"].inverse_transform([predicted_class])[0]

    st.success(f"🌤 Predicted Weather: {weather}")

