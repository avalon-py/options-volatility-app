import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import plotly.graph_objects as go
from scipy.interpolate import griddata

# --- APP CONFIG ---
st.set_page_config(page_title="^SPX Vol Surface", layout="wide")

# --- 1. VECTORIZED MATH ENGINE ---
def bs_price(S, K, T, r, sigma):
    """Vectorized Black-Scholes Price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def find_iv(price, S, K, T, r):
    """Root-finder for Implied Volatility"""
    try:
        # Using Brent's method for stability
        return brentq(lambda sigma: bs_price(S, K, T, r, sigma) - price, 1e-6, 5.0)
    except:
        return np.nan

# Vectorize the solver for array-speed
find_iv_vec = np.vectorize(find_iv)

# --- 2. DATA INGESTION (CACHED) ---
@st.cache_data(ttl=3600) # Only fetch from Yahoo once per hour
def get_market_data(ticker_symbol):
    tk = yf.Ticker(ticker_symbol)
    # Get current price
    hist = tk.history(period='1d')
    if hist.empty: return None, None
    S = hist['Close'].iloc[-1]
    
    # Fetch first 15 expiration dates for density
    exps = tk.options[:15]
    all_options = []
    
    for date in exps:
        opt = tk.option_chain(date)
        calls = opt.calls
        calls['expirationDate'] = date
        all_options.append(calls)
    
    df = pd.concat(all_options).reset_index(drop=True)
    return df, S

# --- 3. MAIN UI ---
st.title("📈 Real-Time ^SPX Volatility Surface")
st.markdown("This dashboard solves for **Implied Volatility** across the S&P 500 option chain to visualize the market's 'Geometry of Fear'.")

with st.sidebar:
    st.header("Model Parameters")
    r_input = st.slider("Risk-Free Rate (r)", 0.0, 0.10, 0.045, 0.005)
    moneyness_range = st.slider("Moneyness Filter", 0.7, 1.3, (0.85, 1.15))
    st.info("Filtering for liquidity (Volume > 1, OI > 2) to reduce noise.")

# --- 4. THE PIPELINE ---
df_raw, S_current = get_market_data("^SPX")

if df_raw is not None:
    # --- CLEANING ---
    df = df_raw.copy()
    df['expirationDate'] = pd.to_datetime(df['expirationDate'])
    df['T'] = (df['expirationDate'] - datetime.now()).dt.days / 365.0
    
    # Filter for quality
    df = df[(df['bid'] > 0) & (df['volume'] > 1) & (df['openInterest'] > 2)]
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['moneyness'] = df['strike'] / S_current
    
    # Apply user filters
    df = df[(df['moneyness'] >= moneyness_range[0]) & (df['moneyness'] <= moneyness_range[1])]
    df = df[df['T'] > (5/365.0)] # Min 5 days to expiry
    
    # --- CALCULATION ---
    with st.spinner("Baking the surface..."):
        df['iv'] = find_iv_vec(df['mid_price'], S_current, df['strike'], df['T'], r_input)
        df = df.dropna(subset=['iv'])
    
    # --- INTERPOLATION & VIZ ---
    # Create grid for the 3D 'Sheet'
    strike_range = np.linspace(df['strike'].min(), df['strike'].max(), 50)
    time_range = np.linspace(df['T'].min(), df['T'].max(), 50)
    strike_grid, time_grid = np.meshgrid(strike_range, time_range)
    
    # Cubic interpolation for smoothness
    iv_grid = griddata((df['strike'], df['T']), df['iv'], (strike_grid, time_grid), method='cubic')

    # Plotly Surface
    fig = go.Figure(data=[go.Surface(
        x=strike_grid, 
        y=time_grid, 
        z=iv_grid, 
        colorscale='Plasma',
        hovertemplate='Strike: %{x}<br>T (Years): %{y}<br>IV: %{z:.2%}<extra></extra>'
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Expiration (Years)',
            zaxis_title='Implied Volatility'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # --- FOOTER STATS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Underlying Price (^SPX)", f"{S_current:.2f}")
    col2.metric("Data Points Processed", len(df))
    col3.metric("Avg Implied Vol", f"{df['iv'].mean():.2%}")

else:
    st.error("Could not fetch market data. Check your internet connection or Yahoo Finance status.")