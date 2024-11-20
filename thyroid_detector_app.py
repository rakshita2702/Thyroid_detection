import streamlit as st
import pickle
import numpy as np

# Load the saved model
@st.cache_resource
def load_model():
    with open('best_model_1.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Thyroid Detection", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: rgb(541, 129, 123);
    color: rgb(10, 5, 5);
    font-family: 'Roboto', sans-serif;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQoGu43I5JdP_y2YTXS_QobRhRPeymZSjnqDA&usqp=CAU", use_column_width=True)

st.title("Thyroid Detection")

# Create form inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    on_thyroxine = st.selectbox("On Thyroxine", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    query_on_thyroxine = st.selectbox("Query on Thyroxine", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    on_antithyroid_medication = st.selectbox("On Antithyroid Medication", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    sick = st.selectbox("Sick", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    pregnant = st.selectbox("Pregnant", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    thyroid_surgery = st.selectbox("Thyroid Surgery", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    I131_treatment = st.selectbox("I131 Treatment", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    query_hypothyroid = st.selectbox("Query Hypothyroid", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    query_hyperthyroid = st.selectbox("Query Hyperthyroid", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    lithium = st.selectbox("Lithium", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    goitre = st.selectbox("Goitre", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    tumor = st.selectbox("Tumor", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    hypopituitary = st.selectbox("Hypopituitary", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    psych = st.selectbox("Psychological Symptoms", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    TSH_measured = st.selectbox("TSH Measured", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    T3_measured = st.selectbox("T3 Measured", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col3:
    TT4_measured = st.selectbox("TT4 Measured", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    T4U_measured = st.selectbox("T4U Measured", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    FTI_measured = st.selectbox("FTI Measured", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    TBG_measured = st.selectbox("TBG Measured", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    TSH = st.number_input("TSH Level", min_value=0.0, max_value=100.0, step=0.1)
    T3 = st.number_input("T3 Level", min_value=0.05, max_value=10.6, step=0.01)
    TT4 = st.number_input("TT4 Level", min_value=3.0, max_value=431.0, step=0.1)
    T4U = st.number_input("T4U Level", min_value=0.0, max_value=100.0, step=0.1)
    FTI = st.number_input("Free Thyroxine Index (FTI)", min_value=2.0, max_value=395.0, step=0.1)
    TBG = st.number_input("TBG Level", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict"):
    features = np.array([[
        age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, 
        sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, 
        query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, 
        TSH_measured, T3_measured, TT4_measured, T4U_measured, FTI_measured, 
        TBG_measured, TSH, T3, TT4, T4U, FTI, TBG
    ]])
    
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.error("Thyroid disease detected")
    else:
        st.success("No thyroid disease detected")

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    font-size: 16pt;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)
