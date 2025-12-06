import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb 

st.title("Prediksi Kasus Covid-19 (XGBoost)")
st.write("Kevin Augusta Mahartadi - 000000115470")

try:
    model = joblib.load('xgb_model.pkl')
except FileNotFoundError:
    st.error("File 'xgb_model.pkl' tidak ditemukan di folder ini")
    st.stop()

st.subheader("Masukkan Data Perubahan Mobilitas (%)")

col1, col2 = st.columns(2)

with col1:
    retail = st.number_input("Retail & Recreation", value=0.0)
    grocery = st.number_input("Grocery & Pharmacy", value=0.0)
    transit = st.number_input("Transit Stations", value=0.0)

with col2:
    workplace = st.number_input("Workplaces", value=0.0)
    residential = st.number_input("Residential", value=0.0)


if st.button("Prediksi New Cases"):
    input_data = pd.DataFrame([[retail, grocery, transit, workplace, residential]],
                              columns=['retail_and_recreation_percent_change_from_baseline',
                                       'grocery_and_pharmacy_percent_change_from_baseline',
                                       'transit_stations_percent_change_from_baseline',
                                       'workplaces_percent_change_from_baseline',
                                       'residential_percent_change_from_baseline'])

    prediction = model.predict(input_data)

    st.success(f"Estimasi Kasus Baru: {int(prediction[0])}")