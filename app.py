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
    """
    Advanced multi-strike implied move using FMP v4 options chain.
    - Pulls full chain
    - Picks expiration around earnings
    - Uses multiple near-the-money strikes to stabilize the estimate
    """
    earn_date = get_next_earnings_date(ticker, api_key)

    url = f"https://financialmodelingprep.com/api/v4/options/chain?symbol={ticker}&apikey={api_key}"
    data = fmp_get(url)
    if not data:
        return None, None

    df = pd.DataFrame(data)
    if df.empty:
        return None, None

    # Try to get stock price from payload
    stock_price = None
    if "stockPrice" in df.columns:
        try:
            stock_price = float(df["stockPrice"].dropna().iloc[0])
        except Exception:
            stock_price = None

    if stock_price is None:
        return None, None

    # Clean and parse expiration
    if "expirationDate" not in df.columns:
        return None, None

    df["expirationDate"] = pd.to_datetime(df["expirationDate"]).dt.date

    # Choose expiration closest AFTER earnings if possible
    if earn_date:
        after = df[df["expirationDate"] >= earn_date]
        if not after.empty:
            target_exp = after["expirationDate"].min()
        else:
            # fallback: closest in absolute time
            df["diff"] = (pd.to_datetime(df["expirationDate"]) - pd.to_datetime(earn_date)).abs()
            target_exp = df.loc[df["diff"].idxmin(), "expirationDate"]
    else:
        target_exp = df["expirationDate"].min()

    chain = df[df["expirationDate"] == target_exp].copy()
    if chain.empty:
        return None, None

    # Normalize option type column name
    opt_col = None
    for c in chain.columns:
        if c.lower() in ["optiontype", "type", "side"]:
            opt_col = c
            break
    if opt_col is None:
        return None, None

    # Separate calls/puts
    calls = chain[chain[opt_col].str.upper().isin(["CALL", "C"])]
    puts  = chain[chain[opt_col].str.upper().isin(["PUT", "P"])]
    if calls.empty or puts.empty:
        return None, None

    # Focus on strikes near the money (within ±10%)
    calls = calls.copy()
    puts = puts.copy()
    calls = calls[(calls["strike"] > 0) & (calls["strike"].between(0.9*stock_price, 1.1*stock_price))]
    puts  = puts[(puts["strike"] > 0) & (puts["strike"].between(0.9*stock_price, 1.1*stock_price))]
    if calls.empty or puts.empty:
        return None, None

    def mid_series(df_part):
        bid_col = None
        ask_col = None
        last_col = None
        for c in df_part.columns:
            cl = c.lower()
            if cl == "bid":
                bid_col = c
            elif cl == "ask":
                ask_col = c
            elif cl in ["last", "lastprice"]:
                last_col = c

        mids = []
        for _, row in df_part.iterrows():
            bid = row.get(bid_col) if bid_col else None
            ask = row.get(ask_col) if ask_col else None
            last = row.get(last_col) if last_col else None
            val = 0.0
            try:
                if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                    val = (bid + ask) / 2
                elif pd.notna(last) and last > 0:
                    val = last
            except Exception:
                pass
            mids.append(val)
        return np.array(mids)

    call_mids = mid_series(calls)
    put_mids  = mid_series(puts)

    if call_mids.size == 0 or put_mids.size == 0:
        return None, None

    # Match calls/puts by closest strike
    call_strikes = calls["strike"].values
    put_strikes  = puts["strike"].values

    pairs = []
    for i, cs in enumerate(call_strikes):
        j = np.argmin(np.abs(put_strikes - cs))
        pairs.append((cs, call_mids[i], put_mids[j]))

    if not pairs:
        return None, None

    # Use multiple near-the-money pairs to stabilize estimate
    pairs = sorted(pairs, key=lambda x: abs(x[0] - stock_price))
    top_pairs = pairs[:5]  # up to 5 closest strikes

    straddle_costs = [(c + p) for _, c, p in top_pairs]
    avg_straddle = np.mean(straddle_costs)

    implied_move = avg_straddle / stock_price
    return float(implied_move), float(stock_price)

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

def get_price_history(ticker, api_key, days=60):
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

    prices = get_price_history(ticker, api_key, days=365)
    if prices.empty:
        return pd.DataFrame()

    rows = []
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
            st.markdown("**Tip:** Try liquid names with active options (AAPL, MSFT, AMZN, NVDA, TSLA).")

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

            if implied_move is None or spot is None:
                st.warning("No usable options data for this ticker/expiration. Try another symbol.")
            else:
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
