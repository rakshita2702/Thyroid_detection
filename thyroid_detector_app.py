# Import necessary libraries
import streamlit as st
import numpy as np
import joblib

# Function to load the trained model
def load_model(file_name='best_model.pkl'):
    try:
        model = joblib.load(file_name)
        return model
    except FileNotFoundError:
        st.error("The model file was not found. Ensure the model is trained and saved as 'best_model.pkl'.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

# Stop execution if the model couldn't be loaded
if model is None:
    st.stop()

# Streamlit page configuration
st.set_page_config(page_title="Thyroid Detection", page_icon="🩺", layout="wide")

# Title and description
st.title("Thyroid Detection System")
st.markdown("Predict thyroid disease using a machine learning model.")

# User input form
st.sidebar.header("Enter Patient Details")

# Collect input data
def collect_input():
    user_data = {
        "Age": st.sidebar.number_input("Age", min_value=1, max_value=100, step=1),
        "Sex": st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male"),
        "TSH": st.sidebar.number_input("TSH Level", min_value=0.0, max_value=100.0, step=0.1),
        "T3": st.sidebar.number_input("T3 Level", min_value=0.05, max_value=10.6, step=0.01),
        "TT4": st.sidebar.number_input("TT4 Level", min_value=3.0, max_value=431.0, step=0.1),
        "T4U": st.sidebar.number_input("T4U Level", min_value=0.0, max_value=100.0, step=0.1),
        "FTI": st.sidebar.number_input("Free Thyroxine Index", min_value=2.0, max_value=395.0, step=0.1),
    }

    binary_features = {f"Binary Feature {i}": st.sidebar.selectbox(f"Binary Feature {i}", [0, 1]) for i in range(1, 11)}

    input_array = np.array(list(user_data.values()) + list(binary_features.values())).reshape(1, -1)
    return input_array

input_features = collect_input()

# Prediction
if st.sidebar.button("Predict"):
    try:
        prediction = model.predict(input_features)[0]
        if prediction == 1:
            st.error("Thyroid disease detected.")
        else:
            st.success("No thyroid disease detected.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
