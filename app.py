import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("model/depression_model.pkl")

st.set_page_config(
    page_title="Teen Depression Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Teen Depression Prediction System")
st.write("Predict Depression Risk using Social Media & Lifestyle Factors")

# Inputs
age = st.number_input("Age", min_value=13, max_value=19, value=16)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

daily_social_media_hours = st.slider(
    "Daily Social Media Hours",
    0.0, 15.0, 5.0
)

platform_usage = st.selectbox(
    "Platform Usage",
    ["Instagram", "TikTok", "Both"]
)

sleep_hours = st.slider(
    "Sleep Hours",
    0.0, 12.0, 7.0
)

screen_time_before_sleep = st.slider(
    "Screen Time Before Sleep",
    0.0, 5.0, 2.0
)

academic_performance = st.slider(
    "Academic Performance",
    0.0, 100.0, 70.0
)

physical_activity = st.slider(
    "Physical Activity",
    0.0, 10.0, 3.0
)

social_interaction_level = st.selectbox(
    "Social Interaction Level",
    ["low", "medium", "high"]
)

stress_level = st.slider(
    "Stress Level",
    1, 10, 5
)

anxiety_level = st.slider(
    "Anxiety Level",
    1, 10, 5
)

addiction_level = st.slider(
    "Addiction Level",
    1, 10, 5
)

# Prediction
if st.button("Predict Depression Risk"):

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "daily_social_media_hours": [daily_social_media_hours],
        "platform_usage": [platform_usage],
        "sleep_hours": [sleep_hours],
        "screen_time_before_sleep": [screen_time_before_sleep],
        "academic_performance": [academic_performance],
        "physical_activity": [physical_activity],
        "social_interaction_level": [social_interaction_level],
        "stress_level": [stress_level],
        "anxiety_level": [anxiety_level],
        "addiction_level": [addiction_level]
    })

    try:
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Prediction Result")

        st.metric(
            label="Depression Probability",
            value=f"{probability * 100:.2f}%"
        )

        # Custom Risk Levels
        if probability >= 0.45:
            st.error(
                f"🔴 High Depression Risk ({probability * 100:.2f}%)"
            )

        elif probability >= 0.20:
            st.warning(
                f"🟠 Medium Depression Risk ({probability * 100:.2f}%)"
            )

        else:
            st.success(
                f"🟢 Low Depression Risk ({probability * 100:.2f}%)"
            )

        st.progress(float(probability))

        st.subheader("📋 Input Summary")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")