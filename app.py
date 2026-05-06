import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import date

# -------------------------------
# Load API key from Streamlit Cloud secrets
api_key = st.secrets["mp"]["api_key"]
# -------------------------------

st.set_page_config(
    page_title="Barzin Financial Earnings Terminal",
    page_icon="📊",
    layout="wide"
)

# ---------- API ----------
BASE = "https://financialmodelingprep.com/stable"

def fmp_get(path, params=None):
    if params is None:
        params = {}
    params["apikey"] = api_key
    url = f"{BASE}/{path}"
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ---------- CORE HELPERS ----------
def get_next_earnings_date(ticker):
    data = fmp_get("earning_calendar", {"symbol": ticker, "limit": 1})
    if not data:
        return None
    try:
        return pd.to_datetime(data[0]["date"]).date()
    except Exception:
        return None

def get_price_history(ticker, days=365):
    data = fmp_get("historical-price-eod/full", {"symbol": ticker})
    if not data or "historical" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df.tail(days)

def get_spot_from_history(ticker):
    prices = get_price_history(ticker, days=5)
    if prices.empty:
        return None
    try:
        return float(prices["close"].iloc[-1])
    except Exception:
        return None

def get_options_chain(ticker):
    data = fmp_get("options/chain", {"symbol": ticker})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # Normalize columns
    if "expirationDate" in df.columns:
        df["expirationDate"] = pd.to_datetime(df["expirationDate"]).dt.date
    if "openInterest" not in df.columns:
        df["openInterest"] = 0.0
    return df

def pick_best_expiration_by_oi(df, spot):
    if df.empty or "expirationDate" not in df.columns:
        return None
    # Sum OI per expiration
    grp = df.groupby("expirationDate")["openInterest"].sum().reset_index()
    if grp.empty:
        return None
    # Sort by total OI desc, then by nearest expiration asc
    grp = grp.sort_values(["openInterest", "expirationDate"], ascending=[False, True])
    return grp["expirationDate"].iloc[0]

def compute_implied_move_from_options(ticker):
    """
    Multi-strike, highest-OI expiration, nearest-exp tie-breaker.
    """
    chain = get_options_chain(ticker)
    if chain.empty:
        return None, None

    # Spot: try from chain, else from history
    spot = None
    for col in ["stockPrice", "underlyingPrice", "underlying"]:
        if col in chain.columns:
            try:
                val = chain[col].dropna().iloc[0]
                spot = float(val)
                break
            except Exception:
                pass
    if spot is None:
        spot = get_spot_from_history(ticker)
    if spot is None or spot <= 0:
        return None, None

    # Pick expiration
    best_exp = pick_best_expiration_by_oi(chain, spot)
    if best_exp is None:
        return None, spot

    sub = chain[chain["expirationDate"] == best_exp].copy()
    if sub.empty:
        return None, spot

    # Identify option type column
    opt_col = None
    for c in sub.columns:
        if c.lower() in ["optiontype", "type", "side"]:
            opt_col = c
            break
    if opt_col is None:
        return None, spot

    # Filter calls/puts
    calls = sub[sub[opt_col].str.upper().isin(["CALL", "C"])].copy()
    puts = sub[sub[opt_col].str.upper().isin(["PUT", "P"])].copy()
    if calls.empty or puts.empty:
        return None, spot

    # Ensure strike column
    if "strike" not in calls.columns or "strike" not in puts.columns:
        return None, spot
    calls = calls[calls["strike"] > 0]
    puts = puts[puts["strike"] > 0]
    if calls.empty or puts.empty:
        return None, spot

    # Focus near-the-money strikes
    calls = calls[calls["strike"].between(0.9 * spot, 1.1 * spot)]
    puts = puts[puts["strike"].between(0.9 * spot, 1.1 * spot)]
    if calls.empty or puts.empty:
        return None, spot

    def mid_price(row):
        bid = None
        ask = None
        last = None
        for c in row.index:
            cl = c.lower()
            if cl == "bid":
                bid = row[c]
            elif cl == "ask":
                ask = row[c]
            elif cl in ["last", "lastprice"]:
                last = row[c]
        val = 0.0
        try:
            if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                val = (bid + ask) / 2
            elif pd.notna(last) and last > 0:
                val = last
        except Exception:
            pass
        return float(val)

    calls["mid"] = calls.apply(mid_price, axis=1)
    puts["mid"] = puts.apply(mid_price, axis=1)
    calls = calls[calls["mid"] > 0]
    puts = puts[puts["mid"] > 0]
    if calls.empty or puts.empty:
        return None, spot

    call_strikes = calls["strike"].values
    put_strikes = puts["strike"].values
    call_mids = calls["mid"].values
    put_mids = puts["mid"].values

    pairs = []
    for i, cs in enumerate(call_strikes):
        j = np.argmin(np.abs(put_strikes - cs))
        pairs.append((cs, call_mids[i], put_mids[j]))

    if not pairs:
        return None, spot

    # Sort by closeness to spot, take up to 5
    pairs = sorted(pairs, key=lambda x: abs(x[0] - spot))[:5]
    straddle_costs = [c + p for _, c, p in pairs]
    if not straddle_costs:
        return None, spot

    avg_straddle = float(np.mean(straddle_costs))
    implied_move = avg_straddle / spot
    return implied_move, spot

def get_simple_sentiment(ticker, limit=20):
    data = fmp_get("stock_news", {"tickers": ticker, "limit": limit})
    if not data:
        return 0.0
    scores = []
    for item in data:
        txt = (item.get("title", "") or "") + " " + (item.get("text", "") or "")
        if not txt.strip():
            continue
        s = TextBlob(txt).sentiment.polarity
        scores.append(s)
    if not scores:
        return 0.0
    return float(np.mean(scores))

def get_post_earnings_moves_simple(ticker, limit=10):
    data = fmp_get("historical/earning_calendar", {"symbol": ticker, "limit": limit})
    if not data:
        return pd.DataFrame()

    prices = get_price_history(ticker, days=365)
    if prices.empty:
        return pd.DataFrame()

    rows = []
    for e in data:
        try:
            d0 = pd.to_datetime(e["date"])
        except Exception:
            continue
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
        "<h1 style='text-align: center;'>Barzin Financial Earnings Terminal</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 16px; color: gray;'>Advanced Earnings Prediction Engine</p>",
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
            st.markdown("**Tip:** Try liquid names (AAPL, MSFT, AMZN, NVDA, TSLA).")

        if run and ticker:
            implied_move, spot = compute_implied_move_from_options(ticker)
            sentiment = get_simple_sentiment(ticker, limit=20)
            hist_df = get_post_earnings_moves_simple(ticker, limit=5)
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
            hist_df = get_post_earnings_moves_simple(ticker_bt, limit=10)
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
                implied_move, spot = compute_implied_move_from_options(t)
                sentiment = get_simple_sentiment(t, limit=10)
                clf = st.session_state["clf"]
                up_prob = predict_direction(clf, implied_move, sentiment, 0.0)
                score = (implied_move or 0.0) * abs(2 * up_prob - 1) * (1 + sentiment)
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
