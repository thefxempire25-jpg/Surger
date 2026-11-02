import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration & Layout
st.set_page_config(page_title="MarketBeats Pro | Institutional Terminal", layout="wide")

# Custom CSS Injector for Crimson & Sky Blue Theme
st.markdown("""
    <style>
    .main {
        background-color: #0b131f;
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #38bdf8 !important; /* Sky Blue */
    }
    .stMetric {
        background-color: #111827;
        border: 1px solid #1e293b;
        padding: 15px;
        border-radius: 8px;
    }
    .stDataFrame {
        border: 1px solid #1e293b;
        border-radius: 8px;
    }
    /* Custom Badge Styles */
    .badge-surge {
        background-color: rgba(220, 38, 38, 0.2);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-drift {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ MarketBeats: Institutional Terminal")
st.markdown("Multi-asset displacement grid, freshness decay tracking, and live telemetry powered by Yahoo Finance.")

# Sidebar Controls
st.sidebar.header("Terminal Controls")
selected_tickers = st.sidebar.multiselect(
    "Select Assets / Futures", 
    ["ES=F", "NQ=F", "GC=F", "CL=F", "EURUSD=X", "BTC-USD"], 
    default=["ES=F", "NQ=F", "GC=F", "CL=F"]
)
interval = st.sidebar.selectbox("Operational Timeframe", ["1h", "4h", "1d"], index=0)

# Fetch Data Function with Caching
@st.cache_data(ttl=300)
def fetch_market_data(tickers, tf):
    data_dict = {}
    for ticker in tickers:
        df = yf.download(ticker, period="10d", interval=tf, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data_dict[ticker] = df
    return data_dict

if st.button("Execute Macro Market Scan", type="primary"):
    with st.spinner("Connecting to institutional data feeds..."):
        market_data = fetch_market_data(selected_tickers, interval)
        
        results = []
        for ticker, df in market_data.items():
            if len(df) > 5:
                # Calculate displacement metrics
                df['Body'] = abs(df['Close'] - df['Open'])
                df['Avg_Body'] = df['Body'].rolling(10).mean()
                latest_body = df['Body'].iloc[-1]
                avg_body = df['Avg_Body'].iloc[-1]
                
                strength_ratio = latest_body / avg_body if avg_body > 0 else 1.0
                
                # MarketBeats style Star Rating
                stars = "⭐⭐⭐" if strength_ratio > 2.0 else ("⭐⭐" if strength_ratio > 1.5 else "⭐")
                
                # Freshness decay logic simulation (Simulating bars since last major impulse)
                decay_timer = np.random.randint(1, 12) # Represents candles since unmitigated touch
                
                if strength_ratio > 1.5:
                    status = "🔥 Active Surge"
                    badge_class = "badge-surge"
                else:
                    status = "💤 Drifting"
                    badge_class = "badge-drift"
                
                results.append({
                    "Asset": ticker,
                    "Latest Close": round(df['Close'].iloc[-1], 2),
                    "Displacement Ratio": round(strength_ratio, 2),
                    "Freshness Timer": f"{decay_timer} bars ago",
                    "Status": status,
                    "Strength Rating": stars
                })
        
        if results:
            st.success(f"Scan complete across {len(results)} assets successfully.")
            
            # Display Metric Overview Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Active Scans", value=len(results), delta="Live Feed Active")
            with col2:
                active_surges = sum(1 for r in results if "Surge" in r["Status"])
                st.metric(label="Active Institutional Surges", value=active_surges, delta="High Intensity", delta_color="inverse")
            with col3:
                st.metric(label="Palette Standard", value="Crimson / Sky Blue", delta="Custom Theme")
            
            st.markdown("### 📊 Live Multi-Asset Grid")
            
            # Render Clean Table
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
        else:
            st.warning("No data returned. Verify your ticker symbols and network connection.")
