import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import plotly.graph_objects as go
from scipy.interpolate import griddata
import requests
import time

# --- APP CONFIG ---
st.set_page_config(page_title="^SPX Vol Surface", layout="wide")

# --- 1. SESSION SETUP (To prevent Rate Limits) ---
# Using a custom User-Agent makes Yahoo think we are a browser, not a bot
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

# --- 2. VECTORIZED MATH ENGINE ---
def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def find_iv(price, S, K, T, r):
    try:
        return brentq(lambda sigma: bs_price(S, K, T, r, sigma) - price, 1e-6, 5.0)
    except:
        return np.nan

find_iv_vec = np.vectorize(find_iv)

# --- 3. DATA INGESTION (With Cooldowns) ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_symbol):
    tk = yf.Ticker(ticker_symbol, session=session)
    
    # Get current price
    hist = tk.history(period='1d')
    if hist.empty: return None, None
    S = hist['Close'].iloc[-1]
    
    # Reduced to 10 expirations to stay under Yahoo's radar
    exps = tk.options[:10]
    all_options = []
    
    progress_bar = st.progress(0)
    for i, date in enumerate(exps):
        try:
            opt = tk.option_chain(date)
            calls = opt.calls
            calls['expirationDate'] = date
            all_options.append(calls)
            # Add a small delay to prevent rapid-fire requests
            time.sleep(0.3) 
            progress_bar.progress((i + 1) / len(exps))
        except Exception as e:
            st.warning(f"Could not fetch data for {date}: {e}")
            continue
    
    if not all_options: return None, None
    df = pd.concat(all_options).reset_index(drop=True)
    return df, S

# --- 4. MAIN UI ---
st.title("📈 Hardened ^SPX Volatility Surface")

with st.sidebar:
    st.header("Parameters")
    r_input = st.slider("Risk-Free Rate (r)", 0.0, 0.10, 0.045, 0.005)
    moneyness_range = st.slider("Moneyness Filter", 0.7, 1.3, (0.85, 1.15))
    if st.button('Clear Cache & Force Refresh'):
        st.cache_data.clear()
        st.rerun()

# --- 5. THE PIPELINE ---
try:
    df_raw, S_current = get_market_data("^SPX")

    if df_raw is not None:
        df = df_raw.copy()
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
        df['T'] = (df['expirationDate'] - datetime.now()).dt.days / 365.0
        
        # Quality Filters
        df = df[(df['bid'] > 0) & (df['volume'] > 0)] 
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['moneyness'] = df['strike'] / S_current
        df = df[(df['moneyness'] >= moneyness_range[0]) & (df['moneyness'] <= moneyness_range[1])]
        df = df[df['T'] > (2/365.0)] # Focus on > 2 days to expiry

        # Math Engine
        df['iv'] = find_iv_vec(df['mid_price'], S_current, df['strike'], df['T'], r_input)
        df = df.dropna(subset=['iv'])

        # Interpolation
        grid_x, grid_y = np.mgrid[df['strike'].min():df['strike'].max():50j, 
                                 df['T'].min():df['T'].max():50j]
        grid_z = griddata((df['strike'], df['T']), df['iv'], (grid_x, grid_y), method='cubic')

        # Viz
        fig = go.Figure(data=[go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Viridis')])
        fig.update_layout(scene=dict(xaxis_title='Strike', yaxis_title='Time', zaxis_title='IV'),
                          height=700)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Yahoo Finance is blocking requests. Please wait a few minutes and refresh.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Try refreshing the page or checking the sidebar to clear the cache.")
