import re, collections
import numpy as np, pandas as pd, streamlit as st, yfinance as yf, plotly.express as px

st.set_page_config(page_title="Market Scanner", layout="wide")

NIFTY50 = """RELIANCE TCS HDFCBANK INFY ICICIBANK HINDUNILVR ITC SBIN BHARTIARTL KOTAKBANK LT AXISBANK ASIANPAINT MARUTI
SUNPHARMA TITAN ULTRACEMCO BAJFINANCE NESTLEIND WIPRO HCLTECH M&M NTPC POWERGRID ONGC JSWSTEEL TATASTEEL ADANIENT
ADANIPORTS COALINDIA TECHM INDUSINDBK BAJAJFINSV HINDALCO DRREDDY CIPLA EICHERMOT GRASIM BRITANNIA APOLLOHOSP DIVISLAB
HEROMOTOCO BPCL SBILIFE HDFCLIFE TATACONSUM BAJAJ-AUTO SHRIRAMFIN""".split()
DEFAULT = "\n".join(s + ".NS" for s in NIFTY50)

SECTORS = {"Bank": "^NSEBANK", "IT": "^CNXIT", "Pharma": "^CNXPHARMA", "Auto": "^CNXAUTO", "FMCG": "^CNXFMCG",
           "Metal": "^CNXMETAL", "Energy": "^CNXENERGY", "Realty": "^CNXREALTY", "Infra": "^CNXINFRA",
           "PSU Bank": "^CNXPSUBANK", "Media": "^CNXMEDIA", "Fin Services": "^CNXFIN"}
BENCH = "^NSEI"

st.title("📈 Market Scanner (live)")
with st.sidebar:
    st.header("Settings")
    tickers = [t.strip() for t in st.text_area("Tickers (Yahoo format, one per line)\nIndia: .NS  US: AAPL", DEFAULT, height=250).split() if t.strip()]
    tol = st.slider("Proximity tolerance % (near 50 DMA / highs)", 0.5, 5.0, 1.5) / 100
    cross_days = st.slider("Crossover lookback (days)", 1, 30, 10)
    if st.button("🔄 Refresh live data"):
        st.cache_data.clear()
    st.caption("Data: Yahoo Finance (free, ~15 min delay for some exchanges).")

@st.cache_data(ttl=300, show_spinner="Downloading prices...")
def get_prices(tk):
    raw = yf.download(list(tk), period="max", interval="1d", auto_adjust=True, group_by="ticker", progress=False, threads=True)
    out = {}
    for t in tk:
        try:
            d = raw[t] if len(tk) > 1 else raw
            d = d.dropna(subset=["Close"])
            if len(d) > 30: out[t] = d
        except Exception:
            pass
    return out

@st.cache_data(ttl=300)
def sector_prices():
    syms = [BENCH] + list(SECTORS.values())
    return yf.download(syms, period="2y", auto_adjust=True, progress=False)["Close"].ffill()

@st.cache_data(ttl=3600, show_spinner="Fetching earnings data...")
def get_earnings(t):
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=12)
        return e.dropna(subset=["Reported EPS"]) if e is not None else None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner="Fetching financials...")
def get_revenue(t):
    try:
        q = yf.Ticker(t).quarterly_income_stmt
        r = q.loc["Total Revenue"].dropna().sort_index(ascending=False)
        return r
    except Exception:
        return None

prices = get_prices(tuple(tickers))
tabs = st.tabs(["Technical scans", "PEAD", "Sector rotation", "Revenue growth YoY", "Earnings call compare"])

# ---------- Technical scans ----------
with tabs[0]:
    rows = []
    for t, d in prices.items():
        c = d["Close"]; last = c.iloc[-1]
        ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
        diff = (ma50 - ma200).dropna()
        cross = ""
        if len(diff) > cross_days:
            w = np.sign(diff.iloc[-cross_days - 1:])
            if w.iloc[0] < 0 < w.iloc[-1]: cross = "Golden cross"
            elif w.iloc[0] > 0 > w.iloc[-1]: cross = "Death cross"
        hi52 = d["High"].iloc[-252:].max(); ath = d["High"].max()
        rows.append({"Ticker": t, "Price": round(last, 2), "50 DMA": round(ma50.iloc[-1], 2) if len(c) >= 50 else None,
                     "200 DMA": round(ma200.iloc[-1], 2) if len(c) >= 200 else None,
                     "% from 50DMA": round((last / ma50.iloc[-1] - 1) * 100, 2) if len(c) >= 50 else None,
                     "Cross": cross, "52W High": round(hi52, 2), "% from 52W High": round((last / hi52 - 1) * 100, 2),
                     "ATH": round(ath, 2), "% from ATH": round((last / ath - 1) * 100, 2)})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No data loaded.")
    else:
        a, b = st.columns(2)
        with a:
            st.subheader("At / near 50 DMA")
            st.dataframe(df[df["% from 50DMA"].abs() <= tol * 100][["Ticker", "Price", "50 DMA", "% from 50DMA"]], hide_index=True)
            st.subheader("52-week highs")
            st.dataframe(df[df["% from 52W High"] >= -tol * 100][["Ticker", "Price", "52W High", "% from 52W High"]], hide_index=True)
        with b:
            st.subheader("50/200 DMA crossovers")
            st.dataframe(df[df["Cross"] != ""][["Ticker", "Price", "50 DMA", "200 DMA", "Cross"]], hide_index=True)
            st.subheader("All-time highs")
            st.dataframe(df[df["% from ATH"] >= -tol * 100][["Ticker", "Price", "ATH", "% from ATH"]], hide_index=True)
        with st.expander("Full table"):
            st.dataframe(df, hide_index=True)

# ---------- PEAD ----------
with tabs[1]:
    st.subheader("Post-Earnings Announcement Drift")
    st.caption("Latest reported quarter: EPS surprise, day-0 reaction, and drift since. Surprise % relies on analyst estimates, which Yahoo has patchily for Indian stocks.")
    if st.button("Run PEAD scan (slow: one call per stock)"):
        out = []
        bar = st.progress(0.0)
        for i, t in enumerate(prices):
            bar.progress((i + 1) / len(prices))
            e = get_earnings(t)
            if e is None or e.empty: continue
            row = e.iloc[0]; dt = pd.Timestamp(e.index[0]).tz_localize(None).normalize()
            c = prices[t]["Close"]; c.index = c.index.tz_localize(None) if c.index.tz else c.index
            after = c[c.index >= dt]; before = c[c.index < dt]
            if after.empty or before.empty: continue
            gap = after.iloc[0] / before.iloc[-1] - 1
            drift = c.iloc[-1] / after.iloc[0] - 1
            sur = row.get("Surprise(%)")
            sig = "Bullish PEAD" if (pd.notna(sur) and sur > 5 and gap > 0 and drift > 0) else \
                  "Bearish PEAD" if (pd.notna(sur) and sur < -5 and gap < 0 and drift < 0) else ""
            out.append({"Ticker": t, "Earnings date": dt.date(), "EPS est": row.get("EPS Estimate"), "EPS actual": row.get("Reported EPS"),
                        "Surprise %": sur, "Day-0 gap %": round(gap * 100, 2), "Drift since %": round(drift * 100, 2),
                        "Days since": len(after) - 1, "Signal": sig})
        st.dataframe(pd.DataFrame(out).sort_values("Drift since %", ascending=False), hide_index=True)

# ---------- Sector rotation ----------
with tabs[2]:
    st.subheader("Sector rotation (RRG-style vs Nifty 50)")
    sp = sector_prices()
    rs = sp.drop(columns=BENCH).div(sp[BENCH], axis=0)
    ratio = 100 * rs / rs.rolling(65).mean()
    mom = 100 * ratio / ratio.shift(10)
    inv = {v: k for k, v in SECTORS.items()}
    res = pd.DataFrame({"RS-Ratio": ratio.iloc[-1], "RS-Momentum": mom.iloc[-1],
                        "1M %": (sp.iloc[-1] / sp.iloc[-22] - 1) * 100, "3M %": (sp.iloc[-1] / sp.iloc[-66] - 1) * 100}).dropna()
    res = res.drop(index=BENCH, errors="ignore")
    res["Sector"] = [inv.get(i, i) for i in res.index]
    res["Quadrant"] = np.select([(res["RS-Ratio"] > 100) & (res["RS-Momentum"] > 100), (res["RS-Ratio"] > 100),
                                 (res["RS-Momentum"] > 100)], ["Leading", "Weakening", "Improving"], "Lagging")
    fig = px.scatter(res, x="RS-Ratio", y="RS-Momentum", text="Sector", color="Quadrant", size_max=12,
                     color_discrete_map={"Leading": "green", "Weakening": "orange", "Improving": "blue", "Lagging": "red"})
    fig.add_hline(y=100); fig.add_vline(x=100); fig.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(res.round(2).set_index("Sector").sort_values("RS-Ratio", ascending=False))

# ---------- Revenue ----------
with tabs[3]:
    st.subheader("Quarterly revenue growth")
    if st.button("Run revenue scan"):
        out = []
        for t in prices:
            r = get_revenue(t)
            if r is None or len(r) < 5: continue
            yoy0 = r.iloc[0] / r.iloc[4] - 1
            yoy1 = r.iloc[1] / r.iloc[5] - 1 if len(r) > 5 else np.nan
            out.append({"Ticker": t, "Latest qtr": r.index[0].date(), "Revenue": r.iloc[0], "YoY %": round(yoy0 * 100, 1),
                        "Prev qtr YoY %": round(yoy1 * 100, 1) if pd.notna(yoy1) else None,
                        "QoQ %": round((r.iloc[0] / r.iloc[1] - 1) * 100, 1),
                        "Accelerating": bool(pd.notna(yoy1) and yoy0 > yoy1)})
        st.dataframe(pd.DataFrame(out).sort_values("YoY %", ascending=False), hide_index=True)
        st.caption("Yahoo only exposes ~5-6 recent quarters, so YoY history is short.")

# ---------- Earnings call ----------
POS = set("growth strong improved improvement record robust momentum confident opportunity expansion accelerate beat upside healthy gains".split())
NEG = set("decline weak pressure headwind slowdown challenging loss uncertainty impact delay risk softness cautious inflation decrease".split())
GUIDE = set("guidance outlook expect expected target forecast anticipate pipeline visibility".split())
STOP = set("the and for that with this have from are was our will you which been they their about were into than also more over what when there can not but its all one would year quarter company".split())

def read_file(f):
    if f is None: return ""
    if f.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return " ".join(p.extract_text() or "" for p in PdfReader(f).pages)
    return f.read().decode("utf-8", errors="ignore")

def words(t): return re.findall(r"[a-z']+", t.lower())

with tabs[4]:
    st.subheader("Earnings call: previous quarter vs latest quarter")
    st.caption("Yahoo has no transcripts. Download them from the company's investor page / BSE / NSE and upload (PDF or TXT).")
    c1, c2 = st.columns(2)
    f_old = c1.file_uploader("Previous quarter call", type=["pdf", "txt"], key="o")
    f_new = c2.file_uploader("Latest quarter call", type=["pdf", "txt"], key="n")
    if f_old and f_new:
        t_old, t_new = read_file(f_old), read_file(f_new)
        w_old, w_new = words(t_old), words(t_new)
        def rate(w, s): return round(1000 * sum(x in s for x in w) / max(len(w), 1), 2)
        st.dataframe(pd.DataFrame({"Metric (per 1000 words)": ["Positive tone", "Negative tone", "Guidance language"],
                                   "Previous": [rate(w_old, POS), rate(w_old, NEG), rate(w_old, GUIDE)],
                                   "Latest": [rate(w_new, POS), rate(w_new, NEG), rate(w_new, GUIDE)]}), hide_index=True)
        co, cn = collections.Counter(x for x in w_old if len(x) > 4 and x not in STOP), collections.Counter(x for x in w_new if len(x) > 4 and x not in STOP)
        chg = {k: cn[k] / max(len(w_new), 1) * 1000 - co[k] / max(len(w_old), 1) * 1000 for k in set(co) | set(cn)}
        s = sorted(chg.items(), key=lambda x: x[1])
        a, b = st.columns(2)
        a.write("**Topics rising**"); a.write(", ".join(k for k, _ in s[::-1][:25]))
        b.write("**Topics fading**"); b.write(", ".join(k for k, _ in s[:25]))
        key = st.text_input("Optional: Anthropic API key for an AI summary of changes", type="password")
        if key and st.button("Summarise with Claude"):
            import anthropic
            msg = anthropic.Anthropic(api_key=key).messages.create(
                model="claude-sonnet-5-5", max_tokens=1500,
                messages=[{"role": "user", "content": "Compare these two earnings calls. List changes in guidance, demand, margins, risks, capex, management tone, and give a bullish/bearish read.\n\nPREVIOUS:\n" + t_old[:60000] + "\n\nLATEST:\n" + t_new[:60000]}])
            st.markdown(msg.content[0].text)
