"""
^SPX Volatility Surface — Firebase-cached, Plotly.js rendered
──────────────────────────────────────────────────────────────
Yahoo Finance is fetched directly via curl_cffi (Chrome TLS impersonation)
with manual crumb auth — bypassing yfinance entirely, which doesn't pass
the curl session through its crumb/CSRF flow on Streamlit Cloud IPs.
"""

import json
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from curl_cffi import requests as curl_requests
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.stats import norm

import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(page_title="SPX Vol Surface", layout="wide", initial_sidebar_state="expanded")

FIRESTORE_COLLECTION = "vol_surfaces"
TICKER = "%5EGSPC"  # ^GSPC URL-encoded


# ── FIREBASE ──────────────────────────────────────────────────────────────────
@st.cache_resource
def init_firebase():
    if firebase_admin._apps:
        return firestore.client()
    fb = st.secrets["firebase"]
    cred = credentials.Certificate({
        "type": fb["type"],
        "project_id": fb["project_id"],
        "private_key_id": fb["private_key_id"],
        "private_key": fb["private_key"].replace("\\n", "\n"),
        "client_email": fb["client_email"],
        "client_id": fb["client_id"],
        "auth_uri": fb["auth_uri"],
        "token_uri": fb["token_uri"],
        "auth_provider_x509_cert_url": fb["auth_provider_x509_cert_url"],
        "client_x509_cert_url": fb["client_x509_cert_url"],
    })
    firebase_admin.initialize_app(cred)
    return firestore.client()


def firestore_read(db, date_key):
    doc = db.collection(FIRESTORE_COLLECTION).document(date_key).get()
    return doc.to_dict() if doc.exists else None


def firestore_write(db, date_key, payload):
    db.collection(FIRESTORE_COLLECTION).document(date_key).set(payload)


# ── DIRECT YAHOO FINANCE CLIENT ───────────────────────────────────────────────
class YahooClient:
    """
    Thin Yahoo Finance client using curl_cffi Chrome impersonation.
    Handles cookie + crumb auth manually so nothing leaks back to
    a plain requests session.
    """

    BASE  = "https://query1.finance.yahoo.com"
    BASE2 = "https://query2.finance.yahoo.com"

    def __init__(self):
        self.session = curl_requests.Session(impersonate="chrome110")
        self._crumb = None
        self._authenticate()

    def _authenticate(self):
        # Step 1: hit the main page to get a valid cookie jar
        self.session.get("https://finance.yahoo.com", timeout=15)
        time.sleep(0.5)
        # Step 2: fetch crumb (requires the cookie from step 1)
        r = self.session.get(
            f"{self.BASE}/v1/test/getcrumb",
            headers={"Accept": "*/*"},
            timeout=15,
        )
        if r.status_code != 200 or not r.text.strip():
            raise RuntimeError(f"Crumb fetch failed (HTTP {r.status_code}). Yahoo may be blocking this IP.")
        self._crumb = r.text.strip()

    def _get(self, url, params=None):
        p = dict(params or {})
        p["crumb"] = self._crumb
        r = self.session.get(url, params=p, timeout=20)
        r.raise_for_status()
        return r.json()

    def spot_price(self):
        data = self._get(
            f"{self.BASE}/v8/finance/chart/{TICKER}",
            params={"interval": "1d", "range": "1d"},
        )
        return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])

    def option_expiries(self):
        data = self._get(f"{self.BASE2}/v7/finance/options/{TICKER}")
        return data["optionChain"]["result"][0]["expirationDates"]  # list of unix timestamps

    def option_chain(self, expiry_ts):
        data = self._get(
            f"{self.BASE2}/v7/finance/options/{TICKER}",
            params={"date": expiry_ts},
        )
        result = data["optionChain"]["result"][0]
        calls = result["options"][0]["calls"]
        expiry_date = datetime.utcfromtimestamp(expiry_ts).strftime("%Y-%m-%d")
        rows = []
        for c in calls:
            rows.append({
                "strike":         c.get("strike"),
                "bid":            c.get("bid", 0),
                "ask":            c.get("ask", 0),
                "volume":         c.get("volume", 0),
                "expirationDate": expiry_date,
            })
        return pd.DataFrame(rows)


# ── BLACK-SCHOLES / IV ────────────────────────────────────────────────────────
def bs_call_price(S, K, T, r, sigma):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _iv_scalar(price, S, K, T, r):
    try:
        return brentq(lambda s: bs_call_price(S, K, T, r, s) - price, 1e-6, 5.0)
    except Exception:
        return np.nan


find_iv_vec = np.vectorize(_iv_scalar)


# ── SURFACE BUILD ─────────────────────────────────────────────────────────────
def build_surface(r_input, moneyness_range):
    client = YahooClient()

    S = client.spot_price()
    expiry_timestamps = client.option_expiries()[:10]

    progress = st.progress(0, text="Fetching option chains…")
    status   = st.empty()
    all_calls = []

    for i, ts in enumerate(expiry_timestamps):
        exp_label = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        try:
            status.text(f"📡  Loading expiry {exp_label}  ({i+1}/{len(expiry_timestamps)})")
            df_chain = client.option_chain(ts)
            all_calls.append(df_chain)
            time.sleep(0.35)
        except Exception as exc:
            st.warning(f"Skipping {exp_label}: {exc}")
        progress.progress((i + 1) / len(expiry_timestamps))

    status.empty()
    progress.empty()

    if not all_calls:
        raise RuntimeError("No option data returned.")

    df = pd.concat(all_calls, ignore_index=True)
    df["expirationDate"] = pd.to_datetime(df["expirationDate"])
    df["T"] = (df["expirationDate"] - datetime.now()).dt.days / 365.0
    df = df[(df["bid"] > 0) & (df["volume"] > 0)]
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["moneyness"] = df["strike"] / S
    df = df[
        (df["moneyness"] >= moneyness_range[0])
        & (df["moneyness"] <= moneyness_range[1])
        & (df["T"] > 2 / 365.0)
    ]

    df["iv"] = find_iv_vec(df["mid_price"], S, df["strike"], df["T"], r_input)
    df = df.dropna(subset=["iv"])

    if df.empty:
        raise RuntimeError("IV computation yielded no valid points.")

    grid_x, grid_y = np.mgrid[
        df["strike"].min(): df["strike"].max(): 50j,
        df["T"].min(): df["T"].max(): 50j,
    ]
    grid_z = griddata((df["strike"], df["T"]), df["iv"], (grid_x, grid_y), method="cubic")
    grid_z_clean = np.where(np.isnan(grid_z), None, grid_z)

    return {
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "grid_z": grid_z_clean.tolist(),
        "S": S,
        "r": r_input,
        "moneyness_low":  moneyness_range[0],
        "moneyness_high": moneyness_range[1],
        "timestamp": datetime.utcnow().isoformat(),
        "source": "yahoo_finance",
    }


# ── PLOTLY.JS RENDERER ────────────────────────────────────────────────────────
def render_plotly_js(payload):
    S  = payload["S"]
    ts = payload.get("timestamp", "")[:10]
    source_label = (
        "🔄 Live (just computed)"
        if payload.get("source") == "yahoo_finance"
        else f"⚡ Cached from Firestore ({ts})"
    )
    js_payload = json.dumps({"x": payload["grid_x"], "y": payload["grid_y"],
                              "z": payload["grid_z"], "S": S})

    html = f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0a0e1a;font-family:'SF Mono','Fira Code',monospace;color:#c8d8f0}}
  #header{{display:flex;align-items:baseline;gap:18px;padding:16px 24px 8px;border-bottom:1px solid #1e2d4a}}
  #header h1{{font-size:1.15rem;font-weight:600;letter-spacing:.12em;color:#e8f0ff;text-transform:uppercase}}
  #header .meta{{font-size:.72rem;color:#5a7aaa;letter-spacing:.05em}}
  #source-badge{{margin-left:auto;font-size:.68rem;padding:3px 10px;border-radius:20px;border:1px solid #2a4070;color:#7aaae8;background:#101828}}
  #plot{{width:100%;height:680px}}
  #tooltip-box{{position:absolute;top:14px;right:14px;background:rgba(10,18,35,.9);border:1px solid #1e3055;border-radius:6px;padding:10px 14px;font-size:.72rem;line-height:1.7;pointer-events:none;display:none;min-width:160px;color:#a8c0e0}}
  #tooltip-box .val{{color:#60c8ff;font-weight:600}}
  #container{{position:relative}}
</style></head><body>
<div id="header">
  <h1>^SPX · Implied Volatility Surface</h1>
  <span class="meta">SPX = {S:,.2f}</span>
  <span id="source-badge">{source_label}</span>
</div>
<div id="container">
  <div id="plot"></div>
  <div id="tooltip-box">
    <div>Strike <span class="val" id="tt-strike">—</span></div>
    <div>Expiry (yrs) <span class="val" id="tt-t">—</span></div>
    <div>IV <span class="val" id="tt-iv">—</span></div>
    <div>Moneyness <span class="val" id="tt-m">—</span></div>
  </div>
</div>
<script>
(function(){{
  const raw={js_payload}, S=raw.S;
  const surface={{
    type:'surface', x:raw.x, y:raw.y, z:raw.z,
    colorscale:[[0,'#0d1f3c'],[.15,'#0a3d62'],[.3,'#1565c0'],[.45,'#1976d2'],
                [.55,'#26c6da'],[.7,'#80deea'],[.82,'#ffe082'],[.92,'#ff8f00'],[1,'#bf360c']],
    showscale:true,
    colorbar:{{title:{{text:'IV',side:'right',font:{{color:'#7090b0',size:11}}}},
               tickfont:{{color:'#7090b0',size:10}},thickness:14,len:.75}},
    contours:{{z:{{show:true,usecolormap:true,highlightcolor:'#60c8ff',project:{{z:false}}}}}},
    hovertemplate:'Strike: <b>%{{x:.0f}}</b><br>T (yrs): <b>%{{y:.3f}}</b><br>IV: <b>%{{z:.4f}}</b><extra></extra>',
    opacity:.93,
    lighting:{{ambient:.7,diffuse:.6,specular:.15,roughness:.6}},
    lightposition:{{x:100,y:200,z:0}}
  }};
  const layout={{
    paper_bgcolor:'#0a0e1a', plot_bgcolor:'#0a0e1a',
    margin:{{l:0,r:0,t:0,b:0}},
    scene:{{
      bgcolor:'#0a0e1a',
      xaxis:{{title:{{text:'Strike',font:{{color:'#5a7aaa',size:11}}}},tickfont:{{color:'#4a6a9a',size:9}},gridcolor:'#132040',zerolinecolor:'#132040'}},
      yaxis:{{title:{{text:'Time to Expiry (yrs)',font:{{color:'#5a7aaa',size:11}}}},tickfont:{{color:'#4a6a9a',size:9}},gridcolor:'#132040',zerolinecolor:'#132040'}},
      zaxis:{{title:{{text:'Implied Volatility',font:{{color:'#5a7aaa',size:11}}}},tickfont:{{color:'#4a6a9a',size:9}},gridcolor:'#132040',zerolinecolor:'#132040',tickformat:'.1%'}},
      camera:{{eye:{{x:1.6,y:-1.6,z:.9}}}}
    }}
  }};
  Plotly.newPlot('plot',[surface],layout,{{responsive:true,displaylogo:false}});
  document.getElementById('plot').on('plotly_hover',function(d){{
    const pt=d.points[0]; if(!pt) return;
    document.getElementById('tooltip-box').style.display='block';
    document.getElementById('tt-strike').textContent=pt.x.toFixed(0);
    document.getElementById('tt-t').textContent=pt.y.toFixed(4);
    document.getElementById('tt-iv').textContent=(pt.z*100).toFixed(2)+'%';
    document.getElementById('tt-m').textContent=(pt.x/S).toFixed(4);
  }});
  document.getElementById('plot').on('plotly_unhover',function(){{
    document.getElementById('tooltip-box').style.display='none';
  }});
}})();
</script></body></html>
"""
    components.html(html, height=740, scrolling=False)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️  Parameters")
    r_input        = st.slider("Risk-Free Rate (r)", 0.0, 0.10, 0.045, 0.005, format="%.3f")
    moneyness_range = st.slider("Moneyness Filter", 0.70, 1.30, (0.85, 1.15), 0.01)
    st.markdown("---")
    st.caption("Surface computed once per day, cached in Firestore. All users that day load instantly.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<style>section[data-testid='stAppViewContainer']{background:#0a0e1a}"
    " header[data-testid='stHeader']{background:#0a0e1a}</style>",
    unsafe_allow_html=True,
)

try:
    db        = init_firebase()
    today_key = date.today().isoformat()

    with st.spinner("Checking Firestore for today's surface…"):
        cached = firestore_read(db, today_key)

    if cached:
        st.toast(f"⚡ Loaded from Firestore (computed at {cached.get('timestamp','?')[:16]} UTC)", icon="✅")
        render_plotly_js(cached)
    else:
        st.info("🔄  No cached surface for today — fetching live data. This takes ~30s and only happens once per day.")
        payload = build_surface(r_input, moneyness_range)
        with st.spinner("Saving to Firestore…"):
            firestore_write(db, today_key, payload)
        st.toast("✅  Saved to Firestore — future loads will be instant!", icon="🔥")
        render_plotly_js(payload)

except Exception as exc:
    st.error(f"**Pipeline error:** {exc}")
    st.info(
        "Common causes:\n"
        "- Missing Firebase credentials in `.streamlit/secrets.toml`\n"
        "- Yahoo Finance blocking this IP — try redeploying (gets a new IP)\n"
        "- No option data survived the quality filters"
    )
    st.exception(exc)
