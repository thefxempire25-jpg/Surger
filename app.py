"""Market Beats Scanner - single-file Streamlit app.

Run:  streamlit run app.py
Needs: streamlit, yfinance, pandas, numpy  (see requirements.txt)
"""
import copy
import html

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Market Beats Scanner", page_icon="\U0001FA82", layout="centered")

# =====================================================================
# LOGIC: universe, 4H institutional scan, timeframe activity (stars)
# =====================================================================
UNIVERSE = {
    "Forex (CFD)": {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
        "NZDUSD": "NZDUSD=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
        "EURGBP": "EURGBP=X",
    },
    "Crypto": {
        "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
        "XRPUSD": "XRP-USD", "BNBUSD": "BNB-USD", "ADAUSD": "ADA-USD",
        "DOGEUSD": "DOGE-USD", "AVAXUSD": "AVAX-USD", "LINKUSD": "LINK-USD",
    },
    "Commodities": {
        "Gold": "GC=F", "Silver": "SI=F", "Crude Oil WTI": "CL=F",
        "Brent Oil": "BZ=F", "Natural Gas": "NG=F", "Copper": "HG=F",
        "Platinum": "PL=F",
    },
    "Index Futures": {
        "S&P 500 (ES)": "ES=F", "Nasdaq 100 (NQ)": "NQ=F",
        "Dow (YM)": "YM=F", "Russell 2000 (RTY)": "RTY=F",
        "10Y Note (ZN)": "ZN=F",
    },
}

DEFAULTS = dict(rvol=2.0, z=1.5, dom=0.25, body=0.8, depth=0.5, lookback=6)


def extract(raw, ticker):
    """Pull one ticker's OHLCV out of a yfinance download frame."""
    if raw is None or len(raw) == 0:
        return None
    df = raw
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0)
        if ticker in set(lv0):
            df = raw[ticker]
        else:
            df = raw.copy()
            df.columns = df.columns.get_level_values(0)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        return None
    return df[need].dropna(subset=["Close"])


def to_4h(df):
    """Resample 1h candles to 4h candles aligned to UTC (00,04,08,12,16,20)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")
    out = d.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                "Close": "last", "Volume": "sum"})
    return out.dropna(subset=["Close"])


def analyze(df4, price, p=None, now=None):
    """Score one instrument on its last CLOSED 4h candle. Returns a dict or None."""
    p = {**DEFAULTS, **(p or {})}
    if df4 is None or len(df4) < 70:
        return None
    d = df4.copy()
    now = now or pd.Timestamp.now(tz="UTC")
    if d.index[-1] + pd.Timedelta(hours=4) > now:   # drop the still-forming candle
        d = d.iloc[:-1]
    if len(d) < 60:
        return None

    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rng = h - l

    has_vol = bool((v.tail(60) > 0).mean() > 0.5)
    rvol = v / v.rolling(20).mean() if has_vol else pd.Series(np.nan, index=d.index)

    rng_atr = rng / atr
    body_atr = (c - o).abs() / atr
    delta = pd.Series(np.where(rng > 0, v * (2 * c - h - l) / rng.replace(0, np.nan), 0.0), index=d.index)
    dstd = delta.rolling(50).std()
    dz = (delta - delta.rolling(50).mean()) / dstd.replace(0, np.nan)
    dom = delta / v.replace(0, np.nan)
    near_low = l.rolling(3).min() <= l.rolling(20).min()
    near_high = h.rolling(3).max() >= h.rolling(20).max()

    dem = (c > o) & (body_atr >= p["body"]) & (rvol >= p["rvol"]) & (dz >= p["z"]) & (dom >= p["dom"]) & near_low
    sup = (c < o) & (body_atr >= p["body"]) & (rvol >= p["rvol"]) & (dz <= -p["z"]) & (dom <= -p["dom"]) & near_high

    # ----- activity score (0-100) -----
    parts = [min(max(rng_atr.iloc[-1] / 2.0, 0), 1)]                    # range expansion
    atr_pct = (atr / c).tail(100)
    parts.append(float(atr_pct.rank(pct=True).iloc[-1]))                # volatility vs own history
    if has_vol:
        parts.append(min(max(rvol.iloc[-1] / 3.0, 0), 1))               # relative volume
        parts.append(min(max(abs(dz.iloc[-1]) / 3.0, 0), 1))            # delta aggression
    parts = [x for x in parts if x == x]
    score = float(np.mean(parts) * 100) if parts else 0.0

    # ----- most recent live (unbroken) signal inside the lookback window -----
    signal, bars_ago, zlo, zhi, dist = "-", None, np.nan, np.nan, np.nan
    n = len(d)
    for k in range(1, int(p["lookback"]) + 1):
        i = n - k
        if i < 0:
            break
        if not (dem.iloc[i] or sup.iloc[i]):
            continue
        is_dem = bool(dem.iloc[i])
        if is_dem:
            lo, hi = float(l.iloc[i]), float(l.iloc[i] + rng.iloc[i] * p["depth"])
            broken = (c.iloc[i + 1:] < lo).any()
        else:
            lo, hi = float(h.iloc[i] - rng.iloc[i] * p["depth"]), float(h.iloc[i])
            broken = (c.iloc[i + 1:] > hi).any()
        if broken:
            continue
        signal = "DEMAND" if is_dem else "SUPPLY"
        bars_ago, zlo, zhi = k - 1, lo, hi
        a = float(atr.iloc[-1]) or np.nan
        if price > zhi:
            dist = (price - zhi) / a
        elif price < zlo:
            dist = (zlo - price) / a
        else:
            dist = 0.0
        break

    return {
        "Price": price,
        "Activity": round(score, 0),
        "RVOL": float(rvol.iloc[-1]) if has_vol else np.nan,
        "Range/ATR": float(rng_atr.iloc[-1]),
        "ATR %": float(atr.iloc[-1] / c.iloc[-1] * 100),
        "Delta z": float(dz.iloc[-1]) if has_vol else np.nan,
        "Signal": signal,
        "Bars ago": bars_ago,
        "Zone low": zlo,
        "Zone high": zhi,
        "Dist (ATR)": dist,
        "Volume data": "yes" if has_vol else "no",
    }


def fetch_1h(tickers, period="60d"):
    import yfinance as yf
    return yf.download(list(tickers), period=period, interval="1h", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)


def scan(categories, params=None):
    tickers, meta = [], {}
    for cat in categories:
        for name, t in UNIVERSE[cat].items():
            tickers.append(t)
            meta[t] = (name, cat)
    raw = fetch_1h(tickers)
    rows = []
    for t in tickers:
        try:
            df1 = extract(raw, t)
            if df1 is None or df1.empty:
                continue
            res = analyze(to_4h(df1), float(df1["Close"].iloc[-1]), params)
            if res:
                name, cat = meta[t]
                rows.append({"Symbol": name, "Category": cat, **res})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Activity", ascending=False).reset_index(drop=True)
    return df



TIMEFRAMES = {
    "M3":  dict(fetch="1m",  period="5d",  rule="3min"),
    "M5":  dict(fetch="5m",  period="5d",  rule=None),
    "M15": dict(fetch="15m", period="10d", rule=None),
    "M30": dict(fetch="30m", period="20d", rule=None),
    "H1":  dict(fetch="60m", period="30d", rule=None),
    "H2":  dict(fetch="60m", period="30d", rule="2h"),
    "H4":  dict(fetch="60m", period="60d", rule="4h"),
}
TF_MINUTES = {"M3": 3, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H2": 120, "H4": 240}
CAT_ICON = {"Forex (CFD)": "\U0001F4B1", "Crypto": "\u20BF", "Commodities": "\U0001F6E2", "Index Futures": "\U0001F4C8"}


DEFAULT_SETTINGS = {
    "trackers": [{"tf": "M5"}, {"tf": "M15"}, {"tf": "H1"}, {"tf": "H4"}],
    "categories": list(UNIVERSE.keys()),
    "min_stars_sky": 2,
    "tz": "UTC",
    "thresholds": [1.6, 2.2, 3.0],
    "log_hours": 24,
}
TZS = ["UTC", "Europe/London", "Europe/Paris", "America/New_York", "Asia/Dubai",
       "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney", "Africa/Lagos"]


def fetch_tf(tickers, tf):
    import yfinance as yf
    cfg = TIMEFRAMES[tf]
    return yf.download(list(tickers), period=cfg["period"], interval=cfg["fetch"], group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)


def to_tf(df, tf):
    """Normalise to UTC and resample if the timeframe is not natively available."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")
    rule = TIMEFRAMES[tf]["rule"]
    if rule:
        d = d.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"})
    return d.dropna(subset=["Close"])


def tf_events(d, tf, thresholds=(1.6, 2.2, 3.0), max_bars=300, now=None):
    """Closed candles whose range is unusually large vs ATR. 1-3 stars = intensity."""
    cols = ["time_close", "stars", "dir", "rng_atr", "rvol"]
    if d is None or len(d) < 40:
        return pd.DataFrame(columns=cols)
    mins = TF_MINUTES[tf]
    now = now or pd.Timestamp.now(tz="UTC")
    d = d.copy()
    if d.index[-1] + pd.Timedelta(minutes=mins) > now:      # drop the forming candle
        d = d.iloc[:-1]
    if len(d) < 30:
        return pd.DataFrame(columns=cols)

    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rng_atr = (h - l) / atr

    t1, t2, t3 = thresholds
    stars = (rng_atr >= t1).astype(int) + (rng_atr >= t2).astype(int) + (rng_atr >= t3).astype(int)
    has_vol = bool((v.tail(60) > 0).mean() > 0.5)
    rvol = v / v.rolling(20).mean() if has_vol else pd.Series(np.nan, index=d.index)
    if has_vol:
        bump = (stars > 0) & (rvol >= 2.0)
        stars = (stars + bump.astype(int)).clip(upper=3)
    stars.iloc[:20] = 0

    direction = np.sign(c - o)
    m = (stars > 0) & (direction != 0)
    m = m & (pd.Series(np.arange(len(d)), index=d.index) >= len(d) - max_bars)
    out = pd.DataFrame({
        "time_close": d.index[m.values] + pd.Timedelta(minutes=mins),
        "stars": stars[m].astype(int).values,
        "dir": direction[m].astype(int).values,
        "rng_atr": rng_atr[m].values,
        "rvol": rvol[m].values,
    })
    return out


def all_events(categories, tf, thresholds=(1.6, 2.2, 3.0), max_bars=300):
    tickers, meta = [], {}
    for cat in categories:
        for name, t in UNIVERSE[cat].items():
            tickers.append(t)
            meta[t] = (name, cat)
    raw = fetch_tf(tickers, tf)
    frames = []
    for t in tickers:
        try:
            df1 = extract(raw, t)
            if df1 is None or df1.empty:
                continue
            ev = tf_events(to_tf(df1, tf), tf, thresholds, max_bars)
            if ev.empty:
                continue
            ev.insert(0, "Symbol", meta[t][0])
            ev.insert(1, "Category", meta[t][1])
            frames.append(ev)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Category", "time_close", "stars", "dir", "rng_atr", "rvol"])
    return pd.concat(frames, ignore_index=True)


def ago_parts(minutes):
    """(number, unit) for the parachute canopy, e.g. ('10', 'minutes ago')."""
    if minutes < 1:
        return "now", ""
    if minutes < 60:
        n = int(minutes)
        return str(n), "minute ago" if n == 1 else "minutes ago"
    hrs = minutes / 60.0
    if hrs < 24:
        n = int(hrs)
        return str(n), "hr ago" if n == 1 else "hrs ago"
    n = int(hrs / 24)
    return str(n), "day ago" if n == 1 else "days ago"


def freshness(minutes, tf):
    step = TF_MINUTES[tf]
    if minutes <= 2 * step:
        return "Fresh"
    if minutes <= 8 * step:
        return "Active"
    return "Fading"


# =====================================================================
# STYLE + HTML HELPERS
# =====================================================================
CSS = """<style>
.sky{background:linear-gradient(180deg,#38bdf8 0%,#7dd3fc 55%,#bae6fd 100%);border-radius:28px;padding:14px 8px 18px;text-align:center;box-shadow:0 6px 18px rgba(20,33,61,.18);margin-bottom:12px}
.sky.fading{background:linear-gradient(180deg,#1e5a78 0%,#2b7a8c 60%,#3b8f98 100%)}
.sky.mini{border-radius:20px;padding:8px 4px 12px}
.flag{display:inline-block;background:#fff;color:#14213D;font-weight:800;padding:5px 14px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.2);margin-top:-4px}
.mini .flag{font-size:12px;padding:3px 8px}
.pill{display:inline-block;color:#fff;font-weight:800;padding:6px 20px;border-radius:12px;margin:8px 0 2px;font-size:18px}
.mini .pill{font-size:12px;padding:3px 10px}
.pill.bull{background:#2e9e5b}.pill.bear{background:#b5523b}
.stars{color:#f7c948;font-size:22px;letter-spacing:4px;text-shadow:0 1px 2px rgba(0,0,0,.35)}
.mini .stars{font-size:15px;letter-spacing:2px}
.cat{font-size:12px;color:#14213D;opacity:.75;margin-top:4px}
.fading .cat{color:#e2f1f5}
.logbook{background:linear-gradient(180deg,#fff8bf,#ffe98a);border-radius:22px;padding:14px 18px 8px;color:#7a4b1f;box-shadow:0 6px 18px rgba(20,33,61,.15)}
.logtitle{font-size:26px;font-weight:900;color:#d9965b;text-align:center;border-bottom:3px solid #e3a76f;padding-bottom:8px;margin-bottom:6px}
.lgrow{border-bottom:1px solid #e6cf7a;padding:8px 0}
.lgh{display:flex;justify-content:space-between;font-weight:800;color:#c08a4e}
.lgline{display:flex;gap:12px;margin-top:3px}
.lgline .st{color:#e0a53a;min-width:54px;font-size:15px}
.lgline .sy{font-weight:800;color:#7a4b1f}
.up{color:#1b8a4b}.dn{color:#b5472f}
.empty{background:#F3E9CC;border-radius:16px;padding:18px;text-align:center}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def parachute(num, unit, bullish, faded, size):
    col = "#3ddc84" if bullish else "#f4895f"
    if faded:
        col = "#8c5a3c"
    svg = (
        '<svg viewBox="0 0 200 215" width="%d" xmlns="http://www.w3.org/2000/svg">'
        '<g stroke="#5b2a1a" stroke-width="2" opacity=".85">'
        '<line x1="14" y1="92" x2="100" y2="150"/><line x1="56" y1="92" x2="100" y2="150"/>'
        '<line x1="144" y1="92" x2="100" y2="150"/><line x1="186" y1="92" x2="100" y2="150"/></g>'
        '<path d="M12 92 Q100 -34 188 92 Q166 72 144 92 Q122 72 100 92 Q78 72 56 92 Q34 72 12 92Z" '
        'fill="%s" stroke="#5b2a1a" stroke-width="3"/>'
        '<text x="100" y="56" text-anchor="middle" font-size="36" font-weight="900" fill="#14213D">%s</text>'
        '<text x="100" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#14213D">%s</text>'
        '<text x="100" y="190" text-anchor="middle" font-size="46">\U0001F43B</text></svg>'
    ) % (size, col, html.escape(num), html.escape(unit))
    return svg


def card(row, tf, now, mini=False):
    mins = (now - row.time_close).total_seconds() / 60.0
    num, unit = ago_parts(mins)
    fresh = freshness(mins, tf)
    bull = row.dir > 0
    faded = fresh == "Fading"
    side = "Bull" if bull else "Bear"
    cls = "sky" + (" mini" if mini else "") + (" fading" if faded else "")
    size = 130 if mini else 250
    stars = "\u2605" * int(row.stars)
    return (
        '<div class="%s">%s<div><span class="flag">%s</span></div>'
        '<div><span class="pill %s">%s %s</span></div><div class="stars">%s</div>'
        '<div class="cat">%s %s</div></div>'
    ) % (cls, parachute(num, unit, bull, faded, size), html.escape(row.Symbol),
         "bull" if bull else "bear", fresh, side, stars,
         CAT_ICON.get(row.Category, ""), html.escape(row.Category))


def log_html(ev, tf, tz, hours, now):
    e = ev[(now - ev.time_close) <= pd.Timedelta(hours=hours)]
    parts = ['<div class="logbook"><div class="logtitle">LOG %s</div>' % tf]
    if e.empty:
        parts.append('<div class="lgrow">Nothing in the last %d hours.</div>' % hours)
    for t, g in sorted(e.groupby("time_close"), key=lambda x: x[0], reverse=True):
        mins = (now - t).total_seconds() / 60.0
        num, unit = ago_parts(mins)
        parts.append('<div class="lgrow"><div class="lgh"><span>%s</span><span>%s %s</span></div>'
                     % (t.tz_convert(tz).strftime("%H:%M"), num, unit))
        for n in (3, 2, 1):
            gg = g[g.stars == n]
            if gg.empty:
                continue
            syms = " &nbsp;".join(
                '<span class="%s">%s%s</span>' % ("up" if r.dir > 0 else "dn",
                                                  "\u25B2" if r.dir > 0 else "\u25BC", html.escape(r.Symbol))
                for r in gg.itertuples())
            parts.append('<div class="lgline"><span class="st">%s</span><span class="sy">%s</span></div>'
                         % ("\u2605" * n, syms))
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


@st.cache_data(ttl=60, show_spinner=False)
def get_events(categories: tuple, tf: str, thresholds: tuple):
    return all_events(list(categories), tf, thresholds)


@st.cache_data(ttl=600, show_spinner=False)
def run_inst_scan(categories: tuple):
    return scan(list(categories))



# =====================================================================
# UI
# =====================================================================
if "settings" not in st.session_state:
    st.session_state["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
S = st.session_state["settings"]

st.title("\U0001FA82 Market Beats Scanner")
top = st.container()
tab_sky, tab_log, tab_inst, tab_set = st.tabs(["\U0001FA82 Sky", "\U0001F4D2 Log", "\U0001F3DB 4H Institutional", "\u2699\uFE0F Trackers"])

# The settings tab is built first so every other tab sees the latest values.
with tab_set:
    st.subheader("Trackers")
    st.caption("Pick up to four timeframes to follow. Switch between them with the chips at the top.")
    tfs = list(TIMEFRAMES.keys())
    for i in range(4):
        S["trackers"][i]["tf"] = st.selectbox("Tracker %d timeframe" % (i + 1), tfs,
                                              index=tfs.index(S["trackers"][i]["tf"]), key="tf%d" % i)

    st.subheader("Markets")
    S["categories"] = st.multiselect("Scan these markets", list(UNIVERSE.keys()), default=S["categories"]) or list(UNIVERSE.keys())

    st.subheader("Stars and display")
    S["min_stars_sky"] = st.select_slider("Show on Sky from", options=[1, 2, 3], value=S["min_stars_sky"],
                                          format_func=lambda x: "\u2605" * x)
    S["log_hours"] = st.slider("Log window (hours)", 6, 72, S["log_hours"])
    S["tz"] = st.selectbox("Time zone", TZS, index=TZS.index(S["tz"]))
    with st.expander("Star thresholds (candle range / ATR)"):
        a = st.number_input("1 star at", 0.8, 5.0, float(S["thresholds"][0]), 0.1)
        b = st.number_input("2 stars at", 1.0, 6.0, float(S["thresholds"][1]), 0.1)
        c = st.number_input("3 stars at", 1.5, 8.0, float(S["thresholds"][2]), 0.1)
        S["thresholds"] = sorted([a, b, c])
        st.caption("Where real volume exists (crypto, futures), RVOL of 2 or more adds a star.")

with top:
    labels = ["T%d \u00B7 %s" % (i + 1, S["trackers"][i]["tf"]) for i in range(4)]
    pick = st.radio("Tracker", labels, horizontal=True, label_visibility="collapsed")
    tf = S["trackers"][labels.index(pick)]["tf"]
    if st.button("\U0001F504 Refresh", use_container_width=True):
        st.cache_data.clear()

now = pd.Timestamp.now(tz="UTC")
with st.spinner("Scanning %s..." % tf):
    ev = get_events(tuple(S["categories"]), tf, tuple(S["thresholds"]))

with tab_sky:
    sky = ev[ev.stars >= S["min_stars_sky"]].sort_values("time_close", ascending=False).drop_duplicates("Symbol")
    if sky.empty:
        st.markdown('<div class="empty">Quiet skies on %s. Lower the star filter in Trackers, or check back soon.</div>' % tf,
                    unsafe_allow_html=True)
    else:
        rows = list(sky.head(10).itertuples())
        st.markdown(card(rows[0], tf, now), unsafe_allow_html=True)
        rest_rows = rows[1:]
        for i in range(0, len(rest_rows), 2):
            cols = st.columns(2)
            for col, r in zip(cols, rest_rows[i:i + 2]):
                col.markdown(card(r, tf, now, mini=True), unsafe_allow_html=True)
    st.caption("Most recent active market first. Fresh = within 2 candles, Active = within 8, Fading = older. "
               "Stars = how big the candle is versus ATR. Educational use, not financial advice.")

with tab_log:
    st.markdown(log_html(ev, tf, S["tz"], S["log_hours"], now), unsafe_allow_html=True)

with tab_inst:
    st.caption("Institutional demand/supply on the last closed 4H candle: displacement candle at a swing extreme "
               "with heavy one-sided delta (estimated from candle shape).")
    if st.button("Run 4H institutional scan", use_container_width=True):
        with st.spinner("Scanning 4H..."):
            st.session_state["inst"] = run_inst_scan(tuple(S["categories"]))
    inst = st.session_state.get("inst")
    if inst is None:
        st.info("Tap the button to run the 4H scan (it is not run automatically to keep the app fast).")
        inst = pd.DataFrame()
    elif inst.empty:
        st.info("No data returned. Try again in a minute.")
    if not inst.empty:
        only = st.checkbox("Only markets with an active signal", value=False)
        v = inst[inst["Signal"] != "-"] if only else inst
        st.dataframe(
            v[["Symbol", "Category", "Activity", "RVOL", "Range/ATR", "Delta z", "Signal", "Bars ago", "Zone low", "Zone high", "Dist (ATR)"]].head(25),
            use_container_width=True, hide_index=True,
            column_config={"Activity": st.column_config.ProgressColumn("Activity", min_value=0, max_value=100, format="%d")})
        st.caption("Yahoo Finance has no real volume for forex, so forex is ranked on range and volatility only and gets no signals.")
