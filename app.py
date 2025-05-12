import streamlit as st
import pandas as pd
from predictor import predict

st.set_page_config(layout="wide")
st.title("MDR AI Clinical Dashboard")

uploaded_file = st.file_uploader("Upload Patient CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    features = df.drop(columns=['MDR_Present', 'Dose_Error', 'Timing_Error', 'Route_Error', 'Resistance_Type'], errors='ignore')

    st.subheader("Model Predictions")
    st.write("MDR Present:", predict("MDR_Present", features))
    st.write("Dose Error:", predict("Dose_Error", features))
    st.write("Timing Error:", predict("Timing_Error", features))
    st.write("Route Error:", predict("Route_Error", features))