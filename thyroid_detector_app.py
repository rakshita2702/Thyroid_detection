# Import necessary libraries
import streamlit as st
import numpy as np
import joblib

# Function to load the trained model
def load_model(file_name='best_model.pkl'):
    try:
        model = joblib.load(file_name)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

if model is None:
    st.stop()  # Stop execution if the model couldn't be loaded

# Streamlit page configuration
st.set_page_config(page_title="Thyroid Detection", page_icon="🩺", layout="wide")

# Display app title
st.title("Thyroid Detection")

# Create input form
age = st.number_input("Age", min_value=1, max_value=100, step=1)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
TSH = st.number_input("TSH Level", min_value=0.0, max_value=100.0, step=0.1)
T3 = st.number_input("T3 Level", min_value=0.05, max_value=10.6, step=0.01)
TT4 = st.number_input("TT4 Level", min_value=3.0, max_value=431.0, step=0.1)
T4U = st.number_input("T4U Level", min_value=0.0, max_value=100.0, step=0.1)
FTI = st.number_input("Free Thyroxine Index (FTI)", min_value=2.0, max_value=395.0, step=0.1)

# Collect binary feature inputs
binary_features = [st.selectbox(f"Feature {i}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes") for i in range(20)]

# Predict button
if st.button("Predict"):
    features = np.array([[age, sex, TSH, T3, TT4, T4U, FTI] + binary_features])
    try:
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("Thyroid disease detected")
        else:
            st.success("No thyroid disease detected")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
