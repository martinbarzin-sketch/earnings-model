import streamlit as st
import requests
import pandas as pd

API_KEY = "G2EQrF4VuLpXNlyREllntjszF0fzcBj8"

st.title("Earnings Beat Predictor")

ticker = st.text_input("Enter ticker:", "XYZ").upper()

def fmp(endpoint):
    url = f"https://financialmodelingprep.com/stable/{endpoint}&apikey={API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

if st.button("Analyze"):
    earnings = fmp(f"earnings?symbol={ticker}")
    earnings_df = pd.DataFrame(earnings)

    quote = fmp(f"quote?symbol={ticker}")

    history = earnings_df.dropna(subset=["epsActual", "epsEstimated"]).head(8).copy()

    history["eps_beat"] = history["epsActual"] > history["epsEstimated"]
    history["revenue_beat"] = history["revenueActual"] > history["revenueEstimated"]

    eps_beat_rate = history["eps_beat"].mean()
    revenue_beat_rate = history["revenue_beat"].mean()

    score = 50

    if eps_beat_rate >= 0.60:
        score += 10
    elif eps_beat_rate < 0.45:
        score -= 10

    if revenue_beat_rate >= 0.60:
        score += 10
    elif revenue_beat_rate < 0.45:
        score -= 10

    st.subheader(f"{ticker} Result")
    st.metric("Estimated EPS Beat Chance", f"{score}%")
    st.metric("Recent EPS Beat Rate", f"{eps_beat_rate * 100:.1f}%")
    st.metric("Recent Revenue Beat Rate", f"{revenue_beat_rate * 100:.1f}%")

    if quote:
        st.metric("Current Price", quote[0].get("price"))

    st.dataframe(history)
