
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Traffic Predictor",
    page_icon="🚦"
)

st.title("🚦 Smart City Traffic Predictor")

st.write(
    "Predict traffic congestion using Machine Learning."
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("traffic_model.pkl")

model = load_model()

st.subheader("Enter Traffic Details")

traffic_volume = st.number_input(
    "Traffic Volume",
    min_value=0,
    value=900
)

avg_speed = st.number_input(
    "Average Speed (km/h)",
    min_value=0.0,
    value=35.0
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    value=2.0
)

temperature = st.number_input(
    "Temperature",
    value=28.0
)

hour = st.slider("Hour of Day", 0, 23, 18)

day = st.selectbox(
    "Day of Week",
    range(7)
)

road_type = st.selectbox(
    "Road Type",
    ["highway", "arterial", "residential"]
)

location = st.selectbox(
    "Location",
    ["CBD", "industrial", "residential_zone", "school_zone"]
)

if st.button("Predict Traffic"):

    input_data = pd.DataFrame([{
        "traffic_volume": traffic_volume,
        "avg_speed": avg_speed,
        "rainfall_mm": rainfall,
        "temperature_c": temperature,
        "hour": hour,
        "day_of_week": day,
        "road_type": road_type,
        "location": location
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.metric(
        "Congestion Score",
        f"{prediction:.2f}/100"
    )

    if prediction >= 70:
        st.error("High Traffic Congestion")
    elif prediction >= 35:
        st.warning("Moderate Traffic Congestion")
    else:
        st.success("Low Traffic Congestion")
