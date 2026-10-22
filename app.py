import joblib
import numpy as np
import pandas as pd
import streamlit as st

model = joblib.load('house_price_pipeline3.pkl')

st.title("House Price Prediction")

df = pd.read_csv('sample.csv')

st.write("Values badlo aur price dekho:")

inputs = {}
cols = st.columns(2)
for i, col in enumerate(df.columns):
    with cols[i % 2]:
        val = df[col].iloc[0]
        if not pd.api.types.is_numeric_dtype(df[col]):
            options = sorted(df[col].dropna().unique())
            inputs[col] = st.selectbox(col, options, index=options.index(val))
        else:
            inputs[col] = st.number_input(col, value=float(val))

if st.button("Predict"):
    row = pd.DataFrame([inputs])
    pred = model.predict(row)
    st.success(f"Predicted Price: ${np.expm1(pred[0]):,.0f}")