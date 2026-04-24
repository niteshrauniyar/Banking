"""
app.py
NEPSE Institutional Intelligence System — Main Streamlit Application
Production-grade dark institutional trading dashboard.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config MUST be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="NEPSE Institutional IQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ─────────────────────────────────────────────────────────────
from utils import (
    logger, NEPSE_SYMBOLS, SECTOR_MAP, now_npt, is_market_open,
    color_for_signal, score_to_label, clamp
)
from data_fetcher import get_market_overview, get_stock_history, get_broker_data
from data_cleaner import clean_ohlcv, clean_broker_data, clean_market_overview, add_technical_indicators
from analysis import run_full_analysis
from signals import generate_signal, generate_market_signals
from charts import (
    candlestick_chart, volume_profile_chart, broker_flow_chart,
    market_heatmap, signal_summary_chart, smart_money_radar,
    macd_chart, sector_performance_chart, broker_dominance_pie, ofi_chart,
)


# ══════════════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════════════

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
  --bg:       #0a0e1a;
  --bg2:      #111827;
  --bg3:      #1a2035;
  --border:   #1e2d45;
  --green:    #00ff88;
  --red:      #ff4466;
  --yellow:   #ffd700;
  --blue:     #3b82f6;
  --purple:   #a855f7;
  --teal:     #06b6d4;
  --text:     #e2e8f0;
  --text-dim: #64748b;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

.stSidebar { background-color: var(--bg2) !important; border-right: 1px solid var(--border); }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--text) !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 6px 6px 0 0;
  color: var(--text-dim) !important;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  padding: 8px 18px;
  transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
  background: var(--bg3) !important;
  border-bottom-color: var(--bg3) !important;
  color: var(--teal) !important;
}

div[data-testid="metric-container"] {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: var(--text-dim) !important; font-size: 11px; font-family: 'JetBrains Mono', monospace; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 22px; font-weight: 700; }

.signal-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 12px;
  position: relative;
  overflow: hidden;
}
.signal-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  border-radius: 12px 0 0 12px;
}
.signal-buy::before  { background: #00ff88; }
.signal-sell::before { background: #ff4466; }
.signal-neutral::before { background: #ffd700; }

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.08em;
}
.badge-buy     { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid #00ff88; }
.badge-sell    { background: rgba(255,68,102,0.15); color: #ff4466; border: 1px solid #ff4466; }
.badge-neutral { background: rgba(255,215,0,0.15); color: #ffd700; border: 1px solid #ffd700; }

.info-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 8px;
}
.info-pill {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--text-dim);
}

.level-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-top: 10px;
}
.level-box {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  text-align: center;
}
.level-box .lvl-label { font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }
.level-box .lvl-val   { font-size: 15px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

.reason-list { margin: 8px 0; padding: 0; list-style: none; }
.reason-list li { font-size: 12px; color: var(--text-dim); padding: 3px 0 3px 16px; position: relative; }
.reason-list li::before { content: '›'; position: absolute; left: 0; color: var(--teal); }

.source-badge {
  display: inline-block;
  padding: 2px 8px;
  background: rgba(6,182,212,0.1);
  border: 1px solid rgba(6,182,212,0.3);
  border-radius: 4px;
  font-size: 10px;
  color: var(--teal);
  font-family: 'JetBrains Mono', monospace;
}

.stSelectbox > div { background: var(--bg3) !important; border: 1px solid var(--border) !important; }
.stButton > button {
  background: linear-gradient(135deg, var(--teal), var(--blue)) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHING LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def cached_market_overview() -> Tuple[pd.DataFrame, str]:
    raw, source = get_market_overview()
    df = clean_market_overview(raw)
    return df, source


@st.cache_data(ttl=600, show_spinner=False)
def cached_stock_data(symbol: str) -> Tuple[pd.DataFrame, str]:
    raw, source = get_stock_history(symbol, days=90)
    df = clean_ohlcv(raw, symbol)
    df = add_technical_indicators(df)
    return df, source


@st.cache_data(ttl=600, show_spinner=False)
def cached_broker_data(symbol: str) -> Tuple[pd.DataFrame, str]:
    raw, source = get_broker_data(symbol)
    df = clean_broker_data(raw)
    return df, source


@st.cache_data(ttl=600, show_spinner=False)
def cached_analysis(symbol: str) -> Dict:
    df, _    = cached_stock_data(symbol)
    brk, _   = cached_broker_data(symbol)
    return run_full_analysis(df, brk, symbol)


@st.cache_data(ttl=600, show_spinner=False)
def cached_signal(symbol: str) -> Dict:
    df, _    = cached_stock_data(symbol)
    brk, _   = cached_broker_data(symbol)
    analysis = cached_analysis(symbol)
    return generate_signal(symbol, df, brk, analysis)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER RENDERS
# ══════════════════════════════════════════════════════════════════════════════

def render_signal_card(sig: Dict):
    signal   = sig.get("signal", "NEUTRAL")
    conf     = sig.get("confidence", 0)
    symbol   = sig.get("symbol", "")
    reasons  = sig.get("reasoning", [])
    levels   = sig.get("levels", {})
    regime   = sig.get("regime", "UNKNOWN")
    risk     = sig.get("risk_rating", "MEDIUM")
    rr       = sig.get("rr_ratio", 0)

    cls_map  = {"BUY": "signal-buy", "SELL": "signal-sell", "NEUTRAL": "signal-neutral"}
    bdg_map  = {"BUY": "badge-buy",  "SELL": "badge-sell",  "NEUTRAL": "badge-neutral"}
    cls  = cls_map.get(signal, "signal-neutral")
    bdg  = bdg_map.get(signal, "badge-neutral")

    conf_color = "#00ff88" if conf >= 65 else "#ff4466" if conf <= 35 else "#ffd700"
    reason_html = "".join(f"<li>{r}</li>" for r in reasons[:6])

    lvl_entry = levels.get("entry", 0)
    lvl_sl    = levels.get("stop_loss", 0)
    lvl_t1    = levels.get("target1", 0)
    lvl_t2    = levels.get("target2", 0)

    html = f"""
    <div class="signal-card {cls}">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
          <span style="font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;">{symbol}</span>
          &nbsp;&nbsp;<span class="badge {bdg}">{signal}</span>
        </div>
        <div style="text-align:right;">
          <div style="font-size:28px;font-weight:700;color:{conf_color};font-family:'JetBrains Mono',monospace;">{conf:.0f}</div>
          <div style="font-size:10px;color:#64748b;">CONFIDENCE</div>
        </div>
      </div>
      <div class="info-row">
        <span class="info-pill">Regime: {regime}</span>
        <span class="info-pill">Risk: {risk}</span>
        <span class="info-pill">R:R = 1:{rr:.1f}</span>
        <span class="info-pill">LTP: {sig.get('last_close', 0):,.2f}</span>
      </div>
      <div class="level-grid">
        <div class="level-box"><div class="lvl-label">ENTRY</div><div class="lvl-val" style="color:#00ff88;">{lvl_entry:,.2f}</div></div>
        <div class="level-box"><div class="lvl-label">STOP</div><div class="lvl-val" style="color:#ff4466;">{lvl_sl:,.2f}</div></div>
        <div class="level-box"><div class="lvl-label">TARGET 1</div><div class="lvl-val" style="color:#ffd700;">{lvl_t1:,.2f}</div></div>
        <div class="level-box"><div class="lvl-label">TARGET 2</div><div class="lvl-val" style="color:#f97316;">{lvl_t2:,.2f}</div></div>
      </div>
      <ul class="reason-list">{reason_html}</ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:16px 0 8px;">
          <div style="font-size:28px;">📊</div>
          <div style="font-size:16px;font-weight:700;color:#06b6d4;letter-spacing:0.05em;">NEPSE IQ</div>
          <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;">INSTITUTIONAL INTELLIGENCE</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        now      = now_npt()
        mkt_open = is_market_open()
        status   = "🟢 MARKET OPEN" if mkt_open else "🔴 MARKET CLOSED"
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;padding:10px 12px;margin-bottom:12px;">
          <div style="font-size:12px;font-weight:700;color:{'#00ff88' if mkt_open else '#ff4466'}">{status}</div>
          <div style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;">{now.strftime('%a %d %b %Y %H:%M NPT')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**📌 Select Symbol**")
        selected = st.selectbox("Symbol", NEPSE_SYMBOLS, label_visibility="collapsed")

        st.markdown("**🏭 Sector Filter**")
        sectors   = ["All"] + sorted(set(SECTOR_MAP.values()))
        sel_sector = st.selectbox("Sector", sectors, label_visibility="collapsed")

        st.divider()
        st.markdown("**⚙️ Analysis Settings**")
        lookback = st.slider("Lookback Days", 30, 120, 60, 10)

        st.divider()
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("""
        <div style="margin-top:20px;padding:10px;background:#111827;border-radius:8px;border:1px solid #1e2d45;">
          <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;line-height:1.6;">
            ⚠️ For educational purposes only.<br>
            Not financial advice.<br>
            Always do your own research.
          </div>
        </div>
        """, unsafe_allow_html=True)

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_market_overview():
    st.markdown("### 🗺️ Market Overview")

    with st.spinner("Loading market data…"):
        market_df, source = cached_market_overview()

    if market_df.empty:
        st.error("Could not load market data. Please refresh.")
        return

    st.markdown(f'<span class="source-badge">📡 {source}</span>', unsafe_allow_html=True)
    st.markdown("")

    # Top metrics
    if "change_pct" in market_df.columns:
        gainers  = market_df[market_df["change_pct"] > 0]
        losers   = market_df[market_df["change_pct"] < 0]
        unchanged = market_df[market_df["change_pct"] == 0]
        total_to = market_df.get("turnover", pd.Series([0]*len(market_df))).sum()
        avg_chg  = market_df["change_pct"].mean()
        metric_row([
            ("📈 Gainers",     len(gainers),           None),
            ("📉 Losers",      len(losers),            None),
            ("➡️ Unchanged",   len(unchanged),          None),
            ("💰 Turnover (M)", f"NPR {total_to:,.1f}", None),
            ("📊 Avg Change",  f"{avg_chg:+.2f}%",     None),
        ])

    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(market_heatmap(market_df), use_container_width=True)
    with col2:
        st.plotly_chart(sector_performance_chart(market_df), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 All Stocks")
    display_cols = [c for c in ["symbol", "close", "change", "change_pct", "volume", "turnover", "sector"]
                    if c in market_df.columns]

    styled = market_df[display_cols].copy() if display_cols else market_df.copy()
    if "change_pct" in styled.columns:
        styled = styled.sort_values("change_pct", ascending=False)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "change_pct": st.column_config.NumberColumn("Chg %", format="%.2f%%"),
            "volume": st.column_config.NumberColumn("Volume", format="%d"),
            "turnover": st.column_config.NumberColumn("Turnover (M)", format="%.2f"),
            "close": st.column_config.NumberColumn("LTP", format="%.2f"),
            "change": st.column_config.NumberColumn("Change", format="%.2f"),
        },
        height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – INSTITUTIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def tab_institutional_analysis(symbol: str):
    st.markdown(f"### 🏦 Institutional Analysis — {symbol}")

    with st.spinner(f"Running deep analysis for {symbol}…"):
        df, src_h       = cached_stock_data(symbol)
        brk, src_b      = cached_broker_data(symbol)
        analysis        = cached_analysis(symbol)
        signal          = cached_signal(symbol)

    if df.empty:
        st.error("No data available.")
        return

    sources_html = f'<span class="source-badge">Price: {src_h}</span> &nbsp; <span class="source-badge">Broker: {src_b}</span>'
    st.markdown(sources_html, unsafe_allow_html=True)
    st.markdown("")

    # Key metrics row
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg_pct = float((last["close"] - prev["close"]) / (prev["close"] + 0.01) * 100)
    sm  = analysis.get("smart_money", {})
    acc = analysis.get("accumulation", {})
    inst_vol = analysis.get("inst_vol_pct", 0)

    metric_row([
        ("💰 LTP",                f"NPR {last['close']:,.2f}",   f"{chg_pct:+.2f}%"),
        ("🧠 Smart Money Score", f"{sm.get('composite', 0):.0f}/100", None),
        ("🏦 Broker Acc. Score", f"{acc.get('accumulation_score', 0):.0f}/100", None),
        ("🏛️ Inst. Volume %",    f"{inst_vol:.1f}%",             None),
        ("⚠️ Anomaly Score",     f"{analysis.get('anomaly', {}).get('anomaly_score', 0):.0f}/100", None),
    ])
    st.markdown("---")

    # Main chart
    levels = signal.get("levels", {})
    st.plotly_chart(candlestick_chart(df, signal, levels, symbol), use_container_width=True)
    st.markdown("")

    # Second row: volume profile + smart money radar
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(volume_profile_chart(analysis.get("vol_profile", {}), symbol), use_container_width=True)
    with c2:
        st.plotly_chart(smart_money_radar(sm, symbol), use_container_width=True)

    # MACD + OFI
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(macd_chart(df, symbol), use_container_width=True)
    with c4:
        st.plotly_chart(ofi_chart(df, symbol), use_container_width=True)

    # Detailed analysis panels
    st.markdown("---")
    st.markdown("#### 🔬 Deep Analysis Results")

    with st.expander("📊 Smart Money Components", expanded=True):
        comps = sm.get("components", {})
        if comps:
            comp_df = pd.DataFrame([
                {"Indicator": k.replace("_", " ").title(), "Score": f"{v:.1f}/100",
                 "Status": "🟢 Bullish" if v > 60 else "🔴 Bearish" if v < 40 else "🟡 Neutral"}
                for k, v in comps.items()
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        acc_status = "🟢 ACCUMULATING" if sm.get("is_smart_money_accumulating") else "🔴 DISTRIBUTING / NEUTRAL"
        st.markdown(f"**Smart Money Phase:** {acc_status}")

    with st.expander("🔍 Order Flow & Metaorder Detection"):
        meta = analysis.get("metaorder", {})
        sweep = analysis.get("liq_sweep", {})
        trap  = analysis.get("trap", {})
        dist  = analysis.get("distribution", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Metaorder Splitting**")
            st.markdown(f"- Splitting Score: `{meta.get('splitting_score', 0):.1f}/100`")
            st.markdown(f"- OFI Mean: `{meta.get('ofi_mean', 0):.4f}`")
            st.markdown(f"- Autocorr: `{meta.get('autocorr_mean', 0):.4f}`")
            st.markdown(f"- Detected: `{'⚠️ YES' if meta.get('is_splitting') else 'No'}`")

            st.markdown("**Wyckoff Phase**")
            st.markdown(f"- Phase: `{dist.get('phase', 'UNKNOWN')}`")
            st.markdown(f"- Score: `{dist.get('score', 0):.1f}/100`")

        with col2:
            st.markdown("**Liquidity Sweep**")
            if sweep.get("sweep"):
                st.markdown(f"- Type: `{sweep.get('type')}`")
                st.markdown(f"- Bias: `{sweep.get('bias')}`")
                st.markdown(f"- Level: `{sweep.get('level', 0):,.2f}`")
            else:
                st.markdown("- No sweep detected")

            st.markdown("**Trap / Inducement**")
            if trap.get("trap") != "NONE":
                st.markdown(f"- Pattern: `{trap.get('trap')}`")
                st.markdown(f"- Direction: `{trap.get('direction')}`")
                st.markdown(f"- Confidence: `{trap.get('confidence', 0)}%`")
            else:
                st.markdown("- No trap detected")

    with st.expander("🤖 ML Regime Detection"):
        regimes = analysis.get("regimes")
        if isinstance(regimes, pd.DataFrame) and "regime" in regimes.columns:
            regime_counts = regimes["regime"].value_counts().reset_index()
            regime_counts.columns = ["Regime", "Days"]
            c1, c2 = st.columns(2)
            c1.dataframe(regime_counts, use_container_width=True, hide_index=True)
            c2.markdown(f"""
            **Current Regime:** `{regimes['regime'].iloc[-1]}`  
            **Last 5 Days:**
            """)
            for reg in regimes["regime"].tail(5).values:
                emoji = "🟢" if "ACCUM" in str(reg) else "🔴" if "DISTRIB" in str(reg) else "🟡"
                c2.markdown(f"- {emoji} {reg}")

    with st.expander("⚠️ Anomaly Detection"):
        anom = analysis.get("anomaly", {})
        score = anom.get("anomaly_score", 0)
        sigs  = anom.get("signals", [])
        status = "🔴 ANOMALOUS" if score >= 33 else "🟢 NORMAL"
        st.markdown(f"**Status:** {status} (Score: {score:.0f}/100)")
        if sigs:
            for s in sigs:
                st.markdown(f"- ⚠️ {s}")
        else:
            st.markdown("- No anomalies detected")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def tab_signals():
    st.markdown("### 📡 Trade Signal Board")

    symbols_to_scan = NEPSE_SYMBOLS[:20]  # Top 20 for performance

    with st.spinner("Scanning all symbols for high-conviction setups…"):
        progress = st.progress(0, text="Scanning symbols…")
        signals_list = []
        for i, sym in enumerate(symbols_to_scan):
            try:
                sig = cached_signal(sym)
                signals_list.append(sig)
            except Exception as e:
                logger.warning(f"Signal failed for {sym}: {e}")
            progress.progress((i + 1) / len(symbols_to_scan), text=f"Scanning {sym}…")
        progress.empty()

    if not signals_list:
        st.warning("No signals generated.")
        return

    # Summary DataFrame
    rows = []
    for s in signals_list:
        rows.append({
            "symbol":     s.get("symbol", ""),
            "signal":     s.get("signal", "NEUTRAL"),
            "confidence": s.get("confidence", 0),
            "tech":       s.get("tech_score", 0),
            "flow":       s.get("flow_score", 0),
            "broker":     s.get("broker_score", 0),
            "regime":     s.get("regime", ""),
            "risk":       s.get("risk_rating", ""),
            "entry":      s.get("levels", {}).get("entry", 0),
            "stop":       s.get("levels", {}).get("stop_loss", 0),
            "t1":         s.get("levels", {}).get("target1", 0),
            "t2":         s.get("levels", {}).get("target2", 0),
            "rr":         s.get("rr_ratio", 0),
            "last_close": s.get("last_close", 0),
        })
    sig_df = pd.DataFrame(rows).sort_values("confidence", ascending=False)

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        sig_filter = st.multiselect("Signal Type", ["BUY", "SELL", "NEUTRAL"],
                                    default=["BUY", "SELL"])
    with col2:
        min_conf = st.slider("Min Confidence", 0, 100, 55)
    with col3:
        max_risk = st.selectbox("Max Risk", ["ANY", "LOW", "MEDIUM"])

    filtered = sig_df[sig_df["signal"].isin(sig_filter) & (sig_df["confidence"] >= min_conf)]
    if max_risk != "ANY":
        filtered = filtered[filtered["risk"] == max_risk]

    st.markdown(f"**{len(filtered)} signals match your filters**")
    st.markdown("")

    # Summary chart
    st.plotly_chart(signal_summary_chart(filtered), use_container_width=True)
    st.markdown("---")

    # Signal cards
    buys    = filtered[filtered["signal"] == "BUY"]
    sells   = filtered[filtered["signal"] == "SELL"]
    neutral = filtered[filtered["signal"] == "NEUTRAL"]

    if not buys.empty:
        st.markdown("#### 🟢 BUY Signals")
        cols = st.columns(min(len(buys), 2))
        for i, (_, row) in enumerate(buys.head(4).iterrows()):
            with cols[i % 2]:
                sig = next((s for s in signals_list if s.get("symbol") == row["symbol"]), {})
                render_signal_card(sig)

    if not sells.empty:
        st.markdown("#### 🔴 SELL Signals")
        cols = st.columns(min(len(sells), 2))
        for i, (_, row) in enumerate(sells.head(4).iterrows()):
            with cols[i % 2]:
                sig = next((s for s in signals_list if s.get("symbol") == row["symbol"]), {})
                render_signal_card(sig)

    # Full table
    st.markdown("---")
    st.markdown("#### 📋 Full Signal Table")
    st.dataframe(
        filtered,
        use_container_width=True, hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f"),
            "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "stop":  st.column_config.NumberColumn("Stop",  format="%.2f"),
            "t1":    st.column_config.NumberColumn("T1",    format="%.2f"),
            "t2":    st.column_config.NumberColumn("T2",    format="%.2f"),
            "rr":    st.column_config.NumberColumn("R:R",   format="%.2f"),
        },
        height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – BROKER ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════

def tab_broker_activity(symbol: str):
    st.markdown(f"### 🏦 Broker Intelligence — {symbol}")

    with st.spinner("Loading broker data…"):
        df, src_h     = cached_stock_data(symbol)
        brk, src_b    = cached_broker_data(symbol)
        analysis      = cached_analysis(symbol)

    if brk.empty:
        st.warning("No broker data available.")
        return

    st.markdown(f'<span class="source-badge">Broker: {src_b}</span>', unsafe_allow_html=True)
    st.markdown("")

    acc = analysis.get("accumulation", {})
    net = analysis.get("network", pd.DataFrame())
    dom = analysis.get("dominant", pd.DataFrame())

    metric_row([
        ("📊 Total Transactions", len(brk), None),
        ("💰 Total Turnover",    f"NPR {brk['amount'].sum()/1e6:,.1f}M" if "amount" in brk.columns else "N/A", None),
        ("🏦 Active Brokers",    brk["buyer_broker"].nunique() if "buyer_broker" in brk.columns else 0, None),
        ("🏛️ Inst. Volume %",   f"{analysis.get('inst_vol_pct', 0):.1f}%", None),
        ("📈 Acc. Score",        f"{acc.get('accumulation_score', 0):.0f}/100", None),
    ])
    st.markdown("---")

    # Net flow + dominance charts
    broker_flow = analysis.get("broker_flow", pd.DataFrame())
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(broker_flow_chart(broker_flow, symbol), use_container_width=True)
    with c2:
        st.plotly_chart(broker_dominance_pie(dom, symbol), use_container_width=True)

    st.markdown("---")

    # Top accumulators / distributors
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Top Accumulators")
        top_acc = acc.get("top_accumulators", [])
        if top_acc:
            acc_df = pd.DataFrame(top_acc)
            acc_df["net_val_M"] = acc_df["net_val"] / 1e6
            st.dataframe(acc_df[["broker_name", "net_qty", "net_val_M"]]
                         .rename(columns={"broker_name": "Broker", "net_qty": "Net Qty",
                                          "net_val_M": "Net Value (M)"}),
                         use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 🔴 Top Distributors")
        top_dis = acc.get("top_distributors", [])
        if top_dis:
            dis_df = pd.DataFrame(top_dis)
            dis_df["net_val_M"] = dis_df["net_val"] / 1e6
            st.dataframe(dis_df[["broker_name", "net_qty", "net_val_M"]]
                         .rename(columns={"broker_name": "Broker", "net_qty": "Net Qty",
                                          "net_val_M": "Net Value (M)"}),
                         use_container_width=True, hide_index=True)

    # Network centrality
    st.markdown("---")
    st.markdown("#### 🕸️ Broker Network Centrality")
    if isinstance(net, pd.DataFrame) and not net.empty:
        display_net = net[["broker_name", "degree_centrality", "volume_centrality", "centrality_score"]].head(15)
        display_net["volume_centrality"] = (display_net["volume_centrality"] / 1e6).round(2)
        st.dataframe(
            display_net.rename(columns={
                "broker_name": "Broker", "degree_centrality": "Counterparties",
                "volume_centrality": "Volume (M)", "centrality_score": "Centrality Score"
            }),
            use_container_width=True, hide_index=True,
            column_config={
                "Centrality Score": st.column_config.ProgressColumn("Centrality Score", min_value=0, max_value=100, format="%.1f"),
            },
        )

    # Large trade clusters
    st.markdown("---")
    st.markdown("#### 🐋 Large Trade Clusters (Block Trades)")
    from analysis import large_trade_clusters
    clusters = large_trade_clusters(brk)
    if not clusters.empty:
        clusters["total_value_M"] = clusters["total_value"] / 1e6
        st.dataframe(
            clusters[["avg_price", "total_qty", "total_value_M", "n_trades", "buyer_brokers", "seller_brokers"]]
            .rename(columns={
                "avg_price": "Avg Price", "total_qty": "Total Qty",
                "total_value_M": "Value (M)", "n_trades": "Trades",
                "buyer_brokers": "Buy Brokers", "seller_brokers": "Sell Brokers"
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No significant block trade clusters detected.")

    # Raw floorsheet
    st.markdown("---")
    with st.expander("📋 Raw Floorsheet Data"):
        display_cols = [c for c in ["transaction_no", "buyer_broker", "seller_broker", "quantity", "rate", "amount", "trade_class"]
                        if c in brk.columns]
        st.dataframe(brk[display_cols].head(200), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;padding:8px 0 16px;">
      <div>
        <h1 style="margin:0;font-size:28px;font-family:'Space Grotesk',sans-serif;font-weight:700;
                   background:linear-gradient(90deg,#06b6d4,#3b82f6);-webkit-background-clip:text;
                   -webkit-text-fill-color:transparent;">
          NEPSE Institutional Intelligence System
        </h1>
        <p style="margin:2px 0 0;color:#64748b;font-size:12px;font-family:'JetBrains Mono',monospace;">
          Real-time smart money detection · Order flow analytics · Institutional signals
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    selected_symbol = render_sidebar()

    tabs = st.tabs([
        "🗺️  Market Overview",
        "🏦  Institutional Analysis",
        "📡  Signals",
        "🏦  Broker Activity",
    ])

    with tabs[0]:
        tab_market_overview()

    with tabs[1]:
        tab_institutional_analysis(selected_symbol)

    with tabs[2]:
        tab_signals()

    with tabs[3]:
        tab_broker_activity(selected_symbol)


if __name__ == "__main__":
    main()
