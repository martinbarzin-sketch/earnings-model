import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
import numpy as np

# -------------------------------
# Load API key from Streamlit Cloud secrets
api_key = st.secrets["mp"]["api_key"]
# -------------------------------

st.set_page_config(
    page_title="Ron's Earnings Terminal",
    page_icon="📊",
    layout="wide"
)

# ---------- API ----------
def fmp_get(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ---------- CORE HELPERS ----------
def get_next_earnings_date(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/earning_calendar?symbol={ticker}&limit=1&apikey={api_key}"
    data = fmp_get(url)
    if not data:
        return None
    return pd.to_datetime(data[0]["date"]).date()

def get_implied_move(ticker, api_key):
    earn_date = get_next_earnings_date(ticker, api_key)
    url = f"https://financialmodelingprep.com/api/v3/options/{ticker}?apikey={api_key}"
    data = fmp_get(url)
    if not data:
        return None, None

    stock_price = data[0].get("stockPrice")
    if stock_price is None:
        return None, None

    expirations = []
    for exp_block in data:
        exp_date = pd.to_datetime(exp_block["expirationDate"]).date()
        expirations.append((exp_date, exp_block))

    if not expirations:
        return None, None

    if earn_date:
        after = [e for e in expirations if e[0] >= earn_date]
        if after:
            target_exp = min(after, key=lambda x: x[0])
        else:
            target_exp = min(expirations, key=lambda x: abs(x[0] - earn_date))
    else:
        target_exp = min(expirations, key=lambda x: x[0])

    _, exp_block = target_exp
    chain = exp_block.get("options", [])
    if not chain:
        return None, None

    calls = [o for o in chain if o.get("type") == "CALL"]
    puts  = [o for o in chain if o.get("type") == "PUT"]
    if not calls or not puts:
        return None, None

    atm_call = min(calls, key=lambda x: abs(x["strike"] - stock_price))
    atm_put  = min(puts,  key=lambda x: abs(x["strike"] - stock_price))

    def mid(opt):
        bid = opt.get("bid")
        ask = opt.get("ask")
        last = opt.get("lastPrice")
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2
        return last or 0.0

    call_mid = mid(atm_call)
    put_mid  = mid(atm_put)
    if call_mid <= 0 or put_mid <= 0:
        return None, None

    implied_move = (call_mid + put_mid) / stock_price
    return implied_move, stock_price

def get_simple_sentiment(ticker, api_key, limit=20):
    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={api_key}"
    data = fmp_get(url)
    if not data:
        return 0.0
    scores = []
    for item in data:
        txt = item.get("title", "") + " " + item.get("text", "")
        if not txt.strip():
            continue
        s = TextBlob(txt).sentiment.polarity
        scores.append(s)
    if not scores:
        return 0.0
    return float(np.mean(scores))

def get_price_history(ticker, api_key, days=5):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries={days}&apikey={api_key}"
    data = fmp_get(url)
    if not data or "historical" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df

def get_post_earnings_moves_simple(ticker, api_key, limit=10):
    url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}?limit={limit}&apikey={api_key}"
    data = fmp_get(url)
    if not data:
        return pd.DataFrame()
    rows = []
    prices = get_price_history(ticker, api_key, days=60)
    if prices.empty:
        return pd.DataFrame()
    for e in data:
        d0 = pd.to_datetime(e["date"])
        d1 = d0 + pd.Timedelta(days=1)
        try:
            p0 = prices.loc[:d0].iloc[-1]["close"]
            p1 = prices.loc[:d1].iloc[-1]["close"]
        except Exception:
            continue
        move = (p1 - p0) / p0
        rows.append({"date": d0.date(), "post_earnings_move_pct": move})
    return pd.DataFrame(rows)

# ---------- MODEL ----------
def train_dummy_model():
    X = np.array([
        [0.05,  0.2,  0.1],
        [0.03, -0.3, -0.2],
        [0.02,  0.1,  0.0],
        [0.07,  0.4,  0.3],
        [0.01, -0.2, -0.1],
    ])
    y = np.array([1, 0, 0, 1, 0])
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

def predict_direction(clf, implied_move, sentiment, hist_move):
    x = np.array([[implied_move or 0.0, sentiment, hist_move or 0.0]])
    prob = clf.predict_proba(x)[0, 1]
    return prob

# ---------- UI ----------
def main():
    st.markdown(
        "<h1 style='text-align: center;'>Ron's Earnings Terminal</h1>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        st.success("API key loaded from Streamlit Cloud.")
        if "clf" not in st.session_state:
            st.session_state["clf"] = train_dummy_model()
        if st.button("Reset model"):
            st.session_state["clf"] = train_dummy_model()
            st.success("Model reset.")

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Backtest", "🛰 Scanner"])

    # ---------- PREDICT TAB ----------
    with tab1:
        st.subheader("Next Earnings Move")
        col_in, col_info = st.columns([2, 1])
        with col_in:
            ticker = st.text_input("Ticker", value="AAPL").upper().strip()
            run = st.button("Run Prediction")
        with col_info:
            st.markdown("**Tip:** Try liquid names with active options (AAPL, MSFT, AMZN, NVDA, etc.).")

        if run and ticker:
            implied_move, spot = get_implied_move(ticker, api_key)
            sentiment = get_simple_sentiment(ticker, api_key, limit=20)
            hist_df = get_post_earnings_moves_simple(ticker, api_key, limit=5)
            hist_move = hist_df["post_earnings_move_pct"].mean() if not hist_df.empty else 0.0

            clf = st.session_state["clf"]
            up_prob = predict_direction(clf, implied_move, sentiment, hist_move)
            predicted_move = (implied_move or 0.0) * (2 * up_prob - 1)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Spot", f"{spot:.2f}" if spot else "N/A")
            with m2:
                st.metric("Implied Move (±%)", f"{(implied_move or 0.0)*100:.2f}")
            with m3:
                st.metric("Up Probability", f"{up_prob*100:.1f}%")
            with m4:
                st.metric("Avg Past 1D Move", f"{hist_move*100:.2f}%")

            if predicted_move > 0.01:
                st.success("📈 **UP bias into earnings**")
            elif predicted_move < -0.01:
                st.error("📉 **DOWN bias into earnings**")
            else:
                st.info("⚪ **FLAT / uncertain**")

            with st.expander("News Sentiment (raw)"):
                st.write(f"Sentiment score: {sentiment:.3f}")

            if not hist_df.empty:
                st.subheader("Historical 1-day Post-Earnings Moves")
                st.line_chart(hist_df.set_index("date")["post_earnings_move_pct"])

    # ---------- BACKTEST TAB ----------
    with tab2:
        st.subheader("Historical Implied vs Actual (simple view)")
        c1, c2 = st.columns([2, 1])
        with c1:
            ticker_bt = st.text_input("Backtest Ticker", value="AAPL", key="bt_ticker").upper().strip()
        with c2:
            run_bt = st.button("Run Backtest")

        if run_bt and ticker_bt:
            hist_df = get_post_earnings_moves_simple(ticker_bt, api_key, limit=10)
            if hist_df.empty:
                st.warning("No historical earnings data.")
            else:
                st.dataframe(hist_df)
                st.line_chart(hist_df.set_index("date")["post_earnings_move_pct"])
                csv = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, file_name=f"{ticker_bt}_post_earnings.csv")

    # ---------- SCANNER TAB ----------
    with tab3:
        st.subheader("Simple Universe Scanner")
        universe = st.multiselect(
            "Universe",
            ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"],
            default=["AAPL", "MSFT", "AMZN"],
        )
        run_scan = st.button("Run Scan")
        if run_scan and universe:
            rows = []
            for t in universe:
                implied_move, spot = get_implied_move(t, api_key)
                sentiment = get_simple_sentiment(t, api_key, limit=10)
                clf = st.session_state["clf"]
                up_prob = predict_direction(clf, implied_move, sentiment, 0.0)
                score = (implied_move or 0.0) * abs(2*up_prob - 1) * (1 + sentiment)
                rows.append({
                    "ticker": t,
                    "spot": spot,
                    "implied_move_pct": (implied_move or 0.0)*100,
                    "up_prob": up_prob,
                    "sentiment": sentiment,
                    "scan_score": score,
                })
            if not rows:
                st.warning("No data.")
            else:
                df = pd.DataFrame(rows).sort_values("scan_score", ascending=False)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Scan CSV", csv, file_name="scan_results.csv")

if __name__ == "__main__":
    main()
