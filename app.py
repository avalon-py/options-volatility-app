"""
^SPX Volatility Surface — Firebase-cached, Plotly.js rendered
──────────────────────────────────────────────────────────────
Flow:
  1. On any page load, check Firestore for today's date key.
  2. Cache HIT  → deserialize JSON grid from Firestore, render immediately.
  3. Cache MISS → fetch Yahoo Finance, compute IV surface, persist to
                  Firestore, then render. All subsequent users that day
                  get the cached version instantly.

Frontend rendering is done entirely client-side via Plotly.js so the
heavy numpy interpolation only runs once per calendar day globally.
"""

import json
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.stats import norm

# ── Firebase ──────────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore

# ── APP CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPX Vol Surface",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIRESTORE_COLLECTION = "vol_surfaces"


# ── FIREBASE INIT (singleton) ─────────────────────────────────────────────────
@st.cache_resource
def init_firebase():
    """
    Initialise the Firebase Admin SDK exactly once per server process.

    Credentials are read from st.secrets["firebase"] which maps to the
    [firebase] section of .streamlit/secrets.toml.  See secrets.toml.template
    in this repo for the required keys.
    """
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


# ── FIRESTORE HELPERS ─────────────────────────────────────────────────────────
def firestore_read(db, date_key: str) -> dict | None:
    """Return today's surface dict from Firestore, or None if absent."""
    doc = db.collection(FIRESTORE_COLLECTION).document(date_key).get()
    return doc.to_dict() if doc.exists else None


def firestore_write(db, date_key: str, payload: dict) -> None:
    """Persist the surface payload to Firestore under today's date key."""
    db.collection(FIRESTORE_COLLECTION).document(date_key).set(payload)


# ── BLACK-SCHOLES / IV ENGINE ─────────────────────────────────────────────────
def bs_call_price(S: float, K, T, r: float, sigma) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _find_iv_scalar(price: float, S: float, K: float, T: float, r: float) -> float:
    try:
        return brentq(lambda s: bs_call_price(S, K, T, r, s) - price, 1e-6, 5.0)
    except Exception:
        return np.nan


find_iv_vec = np.vectorize(_find_iv_scalar)


# ── YAHOO FETCH + IV COMPUTE ──────────────────────────────────────────────────
def build_surface(r_input: float, moneyness_range: tuple) -> dict:
    """
    Pull live option chain data from Yahoo Finance, compute the IV surface,
    and return a JSON-serialisable dict ready for Firestore + Plotly.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    tk = yf.Ticker("^SPX")
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not retrieve SPX spot price from Yahoo Finance.")

    S = float(hist["Close"].iloc[-1])
    exps = tk.options[:10]

    progress = st.progress(0, text="Fetching option chains…")
    status = st.empty()
    all_calls = []

    for i, exp_date in enumerate(exps):
        try:
            status.text(f"📡  Loading expiry {exp_date}  ({i + 1}/{len(exps)})")
            chain = tk.option_chain(exp_date)
            calls = chain.calls.copy()
            calls["expirationDate"] = exp_date
            all_calls.append(calls)
            time.sleep(0.5)          # respectful throttle
        except Exception as exc:
            st.warning(f"Skipping {exp_date}: {exc}")
        progress.progress((i + 1) / len(exps))

    status.empty()
    progress.empty()

    if not all_calls:
        raise RuntimeError("No option data returned — Yahoo may be throttling.")

    df = pd.concat(all_calls, ignore_index=True)
    df["expirationDate"] = pd.to_datetime(df["expirationDate"])
    df["T"] = (df["expirationDate"] - datetime.now()).dt.days / 365.0

    # ── Quality filters ──
    df = df[(df["bid"] > 0) & (df["volume"] > 0)]
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["moneyness"] = df["strike"] / S
    df = df[
        (df["moneyness"] >= moneyness_range[0])
        & (df["moneyness"] <= moneyness_range[1])
        & (df["T"] > 2 / 365.0)
    ]

    # ── Implied Volatility ──
    df["iv"] = find_iv_vec(df["mid_price"], S, df["strike"], df["T"], r_input)
    df = df.dropna(subset=["iv"])

    if df.empty:
        raise RuntimeError("IV computation yielded no valid points after filtering.")

    # ── Interpolation grid (50×50) ──
    grid_x, grid_y = np.mgrid[
        df["strike"].min(): df["strike"].max(): 50j,
        df["T"].min(): df["T"].max(): 50j,
    ]
    grid_z = griddata(
        (df["strike"], df["T"]),
        df["iv"],
        (grid_x, grid_y),
        method="cubic",
    )

    # Replace NaN with None so JSON is happy
    grid_z_clean = np.where(np.isnan(grid_z), None, grid_z)

    return {
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "grid_z": grid_z_clean.tolist(),
        "S": S,
        "r": r_input,
        "moneyness_low": moneyness_range[0],
        "moneyness_high": moneyness_range[1],
        "timestamp": datetime.utcnow().isoformat(),
        "source": "yahoo_finance",
    }


# ── PLOTLY.JS RENDERER ────────────────────────────────────────────────────────
def render_plotly_js(payload: dict) -> None:
    """
    Embed a self-contained Plotly.js surface plot inside a Streamlit HTML
    component.  All rendering happens in the browser — no server-side Plotly.
    """
    S = payload["S"]
    ts = payload.get("timestamp", "")[:10]
    source_label = "🔄 Live (just computed)" if payload.get("source") == "yahoo_finance" else f"⚡ Cached from Firestore ({ts})"

    # Serialise only what JS needs — keep the blob small
    js_payload = json.dumps({
        "x": payload["grid_x"],
        "y": payload["grid_y"],
        "z": payload["grid_z"],
        "S": S,
    })

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0e1a;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #c8d8f0;
  }}
  #header {{
    display: flex;
    align-items: baseline;
    gap: 18px;
    padding: 16px 24px 8px;
    border-bottom: 1px solid #1e2d4a;
  }}
  #header h1 {{
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #e8f0ff;
    text-transform: uppercase;
  }}
  #header .meta {{
    font-size: 0.72rem;
    color: #5a7aaa;
    letter-spacing: 0.05em;
  }}
  #source-badge {{
    margin-left: auto;
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #2a4070;
    color: #7aaae8;
    background: #101828;
  }}
  #plot {{ width: 100%; height: 680px; }}
  #tooltip-box {{
    position: absolute;
    top: 14px; right: 14px;
    background: rgba(10,18,35,0.9);
    border: 1px solid #1e3055;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.72rem;
    line-height: 1.7;
    pointer-events: none;
    display: none;
    min-width: 160px;
    color: #a8c0e0;
  }}
  #tooltip-box .val {{ color: #60c8ff; font-weight: 600; }}
  #container {{ position: relative; }}
</style>
</head>
<body>

<div id="header">
  <h1>^SPX  ·  Implied Volatility Surface</h1>
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
(function() {{
  const raw = {js_payload};
  const S   = raw.S;

  // Build readable x-axis labels (strike → moneyness %)
  const x0 = raw.x.map(row => row[0]);          // first-column strikes per row
  const y0 = raw.x[0].map((_, ci) => raw.y[0][ci]); // T values

  const surface = {{
    type: 'surface',
    x: raw.x,
    y: raw.y,
    z: raw.z,
    colorscale: [
      [0.00, '#0d1f3c'],
      [0.15, '#0a3d62'],
      [0.30, '#1565c0'],
      [0.45, '#1976d2'],
      [0.55, '#26c6da'],
      [0.70, '#80deea'],
      [0.82, '#ffe082'],
      [0.92, '#ff8f00'],
      [1.00, '#bf360c'],
    ],
    showscale: true,
    colorbar: {{
      title: {{ text: 'IV', side: 'right', font: {{ color: '#7090b0', size: 11 }} }},
      tickfont: {{ color: '#7090b0', size: 10 }},
      thickness: 14,
      len: 0.75,
    }},
    contours: {{
      z: {{ show: true, usecolormap: true, highlightcolor: '#60c8ff', project: {{ z: false }} }},
    }},
    hovertemplate:
      'Strike: <b>%{{x:.0f}}</b><br>' +
      'T (yrs): <b>%{{y:.3f}}</b><br>' +
      'IV: <b>%{{z:.4f}}</b><extra></extra>',
    opacity: 0.93,
    lighting: {{ ambient: 0.7, diffuse: 0.6, specular: 0.15, roughness: 0.6 }},
    lightposition: {{ x: 100, y: 200, z: 0 }},
  }};

  const layout = {{
    paper_bgcolor: '#0a0e1a',
    plot_bgcolor: '#0a0e1a',
    margin: {{ l: 0, r: 0, t: 0, b: 0 }},
    scene: {{
      bgcolor: '#0a0e1a',
      xaxis: {{
        title: {{ text: 'Strike', font: {{ color: '#5a7aaa', size: 11 }} }},
        tickfont: {{ color: '#4a6a9a', size: 9 }},
        gridcolor: '#132040',
        zerolinecolor: '#132040',
      }},
      yaxis: {{
        title: {{ text: 'Time to Expiry (yrs)', font: {{ color: '#5a7aaa', size: 11 }} }},
        tickfont: {{ color: '#4a6a9a', size: 9 }},
        gridcolor: '#132040',
        zerolinecolor: '#132040',
      }},
      zaxis: {{
        title: {{ text: 'Implied Volatility', font: {{ color: '#5a7aaa', size: 11 }} }},
        tickfont: {{ color: '#4a6a9a', size: 9 }},
        gridcolor: '#132040',
        zerolinecolor: '#132040',
        tickformat: '.1%',
      }},
      camera: {{ eye: {{ x: 1.6, y: -1.6, z: 0.9 }} }},
    }},
  }};

  const config = {{
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['toImage'],
    toImageButtonOptions: {{ format: 'svg', filename: 'spx_vol_surface' }},
  }};

  Plotly.newPlot('plot', [surface], layout, config);

  // Live tooltip enrichment
  document.getElementById('plot').on('plotly_hover', function(data) {{
    const pt = data.points[0];
    if (!pt) return;
    const box = document.getElementById('tooltip-box');
    box.style.display = 'block';
    document.getElementById('tt-strike').textContent = pt.x.toFixed(0);
    document.getElementById('tt-t').textContent      = pt.y.toFixed(4);
    document.getElementById('tt-iv').textContent     = (pt.z * 100).toFixed(2) + '%';
    document.getElementById('tt-m').textContent      = (pt.x / S).toFixed(4);
  }});
  document.getElementById('plot').on('plotly_unhover', function() {{
    document.getElementById('tooltip-box').style.display = 'none';
  }});
}})();
</script>
</body>
</html>
"""
    components.html(html, height=740, scrolling=False)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️  Parameters")
    r_input = st.slider("Risk-Free Rate (r)", 0.0, 0.10, 0.045, 0.005, format="%.3f")
    moneyness_range = st.slider("Moneyness Filter", 0.70, 1.30, (0.85, 1.15), 0.01)

    st.markdown("---")
    st.caption(
        "The surface is computed once per day and stored in Firestore.  "
        "All users that day see the cached version instantly."
    )


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
st.markdown(
    "<style>section[data-testid='stAppViewContainer']{background:#0a0e1a}"
    " header[data-testid='stHeader']{background:#0a0e1a}</style>",
    unsafe_allow_html=True,
)

try:
    db = init_firebase()
    today_key = date.today().isoformat()

    # ── 1. Try Firestore first ──────────────────────────────────────────────
    with st.spinner("Checking Firestore for today's surface…"):
        cached = firestore_read(db, today_key)

    if cached:
        # ── Cache HIT: render immediately, no Yahoo call needed ──
        st.toast(f"⚡ Loaded from Firestore (computed at {cached.get('timestamp','?')[:16]} UTC)", icon="✅")
        render_plotly_js(cached)

    else:
        # ── Cache MISS: compute, persist, then render ──
        st.info(
            "🔄  No cached surface for today — fetching live data from Yahoo Finance.  "
            "This takes ~30 seconds and only happens once per day."
        )
        payload = build_surface(r_input, moneyness_range)

        with st.spinner("Saving to Firestore…"):
            firestore_write(db, today_key, payload)

        st.toast("✅  Surface saved to Firestore — future loads will be instant!", icon="🔥")
        render_plotly_js(payload)

except Exception as exc:
    st.error(f"**Pipeline error:** {exc}")
    st.info(
        "Common causes:\n"
        "- Missing or incorrect Firebase credentials in `.streamlit/secrets.toml`\n"
        "- Yahoo Finance rate-limiting (try again in a few minutes)\n"
        "- No option data survived the quality filters"
    )
    st.exception(exc)
