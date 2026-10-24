import joblib
import numpy as np
import pandas as pd
import streamlit as st

model = joblib.load('house_price_pipeline3.pkl')

st.set_page_config(page_title="House Price Studio", page_icon="⌂", layout="wide")

df = pd.read_csv('sample.csv')

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(28, 118, 104, 0.16), transparent 36%),
            #f4f7f6;
        color: #172b2a;
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1050px;
        padding-top: 3.2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .hero {
        padding: 2.6rem 2.8rem;
        border-radius: 24px;
        background: linear-gradient(120deg, #103d39 0%, #17675d 58%, #238675 100%);
        color: #f7fffc;
        box-shadow: 0 18px 44px rgba(20, 77, 68, 0.17);
        margin-bottom: 1.8rem;
    }
    .eyebrow {
        color: #a8e0d2;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero h1 {
        color: #ffffff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.1rem, 5vw, 3.25rem);
        font-weight: 800;
        letter-spacing: -0.055em;
        line-height: 1.08;
        margin: 0;
    }
    .hero p {
        color: #d0e8e2;
        font-size: 1.03rem;
        margin: 0.9rem 0 0;
        max-width: 600px;
    }
    .section-heading {
        color: #183b37;
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0 0 0.25rem;
    }
    .section-copy {
        color: #637774;
        font-size: 0.94rem;
        margin: 0 0 1.2rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #e2ebe8;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(28, 65, 59, 0.055);
    }
    [data-testid="stWidgetLabel"] p {
        color: #334d49;
        font-weight: 600;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #fbfdfc;
        border-color: #dce7e3;
        border-radius: 10px;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #278675;
        box-shadow: 0 0 0 1px #278675;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #278675;
        box-shadow: 0 0 0 1px #278675;
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input {
        color: #172b2a !important;
        -webkit-text-fill-color: #172b2a !important;
    }
    .stButton > button {
        min-height: 3rem;
        border: 0;
        border-radius: 11px;
        background: linear-gradient(100deg, #17675d, #238675);
        color: white;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 18px rgba(23, 103, 93, 0.2);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .stButton > button:hover {
        border: 0;
        color: white;
        box-shadow: 0 11px 22px rgba(23, 103, 93, 0.27);
        transform: translateY(-1px);
    }
    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid #b9e3d2;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.5rem;
        }
        .hero {
            padding: 1.8rem 1.5rem;
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">A smarter move starts here</div>
        <h1>Find your home’s<br>next value.</h1>
        <p>Adjust the property details below to get a price estimate tailored to your home.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

inputs = {}
with st.container(border=True):
    st.markdown('<div class="section-heading">Property details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Fine-tune the details to explore your estimate.</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, col in enumerate(df.columns):
        with cols[i % 2]:
            val = df[col].iloc[0]
            if not pd.api.types.is_numeric_dtype(df[col]):
                options = sorted(df[col].dropna().unique())
                inputs[col] = st.selectbox(col, options, index=options.index(val))
            else:
                inputs[col] = st.number_input(col, value=float(val))

if st.button("Estimate my home’s value", use_container_width=True):
    row = pd.DataFrame([inputs])
    pred = model.predict(row)
    st.success(f"Predicted Price: ${np.expm1(pred[0]):,.0f}")
