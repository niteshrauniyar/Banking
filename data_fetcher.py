"""
data_fetcher.py
Multi-source data fetcher for NEPSE Institutional Intelligence System.
Priority: NEPSE Official API → ShareSansar → NepseAlpha → Simulated fallback
"""

import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils import (
    retry, timed, random_headers, json_headers,
    nepal_trading_days, NEPSE_SYMBOLS, BROKER_NAMES, logger, now_npt
)

# ── Source constants ───────────────────────────────────────────────────────────
NEPSE_BASE       = "https://nepalstock.com.np/api/nots"
SHARESANSAR_BASE = "https://www.sharesansar.com"
NEPSEALPHA_BASE  = "https://nepsealpha.com"
TIMEOUT          = 12


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – NEPSE Official API
# ══════════════════════════════════════════════════════════════════════════════

@retry(max_attempts=3, delay=1.5)
def _nepse_get(endpoint: str) -> Optional[dict]:
    url = f"{NEPSE_BASE}/{endpoint}"
    resp = requests.get(url, headers=json_headers(), timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_nepse_market_summary() -> Optional[pd.DataFrame]:
    """Fetch today's summary from NEPSE official API."""
    try:
        data = _nepse_get("market/turnover")
        if data and isinstance(data, dict):
            records = data.get("content") or data.get("data") or []
            if records:
                df = pd.DataFrame(records)
                logger.info(f"NEPSE API: {len(df)} records")
                return df
    except Exception as exc:
        logger.warning(f"NEPSE API summary failed: {exc}")
    return None


@retry(max_attempts=3, delay=2.0)
def fetch_nepse_stock_detail(symbol: str) -> Optional[dict]:
    try:
        data = _nepse_get(f"company/symbol/{symbol}")
        return data
    except Exception as exc:
        logger.debug(f"NEPSE stock detail failed for {symbol}: {exc}")
        return None


@retry(max_attempts=3, delay=1.5)
def fetch_nepse_price_history(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    try:
        data = _nepse_get(f"market/history/price/{symbol}")
        if data:
            content = data.get("content") or data.get("data") or data if isinstance(data, list) else []
            if content:
                df = pd.DataFrame(content)
                logger.info(f"NEPSE price history: {symbol} – {len(df)} rows")
                return df
    except Exception as exc:
        logger.debug(f"NEPSE price history failed for {symbol}: {exc}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – ShareSansar Scraper
# ══════════════════════════════════════════════════════════════════════════════

@retry(max_attempts=3, delay=2.0)
def _sharesansar_html(path: str) -> Optional[BeautifulSoup]:
    url = f"{SHARESANSAR_BASE}/{path}"
    resp = requests.get(url, headers=random_headers(), timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def fetch_sharesansar_floorsheet(symbol: str = None) -> Optional[pd.DataFrame]:
    """Scrape floorsheet from ShareSansar."""
    try:
        path = f"stock/{symbol}" if symbol else "today-share-price"
        soup = _sharesansar_html(path)
        if soup is None:
            return None
        table = soup.find("table", {"class": lambda c: c and "table" in c})
        if table is None:
            table = soup.find("table")
        if table is None:
            return None
        rows = []
        headers_row = table.find("thead")
        hdrs = [th.get_text(strip=True) for th in (headers_row.find_all("th") if headers_row else [])]
        for tr in table.find("tbody").find_all("tr") if table.find("tbody") else []:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=hdrs[:len(rows[0])] if hdrs else None)
        logger.info(f"ShareSansar scrape: {len(df)} rows")
        return df
    except Exception as exc:
        logger.warning(f"ShareSansar scrape failed: {exc}")
        return None


def fetch_sharesansar_today_price() -> Optional[pd.DataFrame]:
    """Fetch today's price table from ShareSansar."""
    try:
        soup = _sharesansar_html("today-share-price")
        if soup is None:
            return None
        table = soup.find("table", id="headFixed")
        if table is None:
            table = soup.find("table")
        if table is None:
            return None
        df = _parse_html_table(table)
        logger.info(f"ShareSansar today price: {len(df)} rows")
        return df
    except Exception as exc:
        logger.warning(f"ShareSansar today price failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – NepseAlpha Scraper
# ══════════════════════════════════════════════════════════════════════════════

@retry(max_attempts=3, delay=2.0)
def fetch_nepsealpha_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        url = f"{NEPSEALPHA_BASE}/company/{symbol}/chart-data"
        headers = json_headers()
        headers["Referer"] = f"{NEPSEALPHA_BASE}/company/{symbol}"
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            logger.info(f"NepseAlpha: {symbol} – {len(df)} rows")
            return df
        elif isinstance(data, dict):
            content = data.get("data") or data.get("content") or []
            if content:
                return pd.DataFrame(content)
    except Exception as exc:
        logger.debug(f"NepseAlpha failed for {symbol}: {exc}")
    return None


@retry(max_attempts=3, delay=2.0)
def fetch_nepsealpha_broker_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        url = f"{NEPSEALPHA_BASE}/floorsheet/{symbol}"
        resp = requests.get(url, headers=random_headers(), timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if table:
            return _parse_html_table(table)
    except Exception as exc:
        logger.debug(f"NepseAlpha broker data failed: {exc}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Simulation Fallback (always succeeds)
# ══════════════════════════════════════════════════════════════════════════════

def _seed_for(symbol: str) -> int:
    return sum(ord(c) for c in symbol) % (2**31)


def simulate_ohlcv(symbol: str, days: int = 90) -> pd.DataFrame:
    """
    Simulate realistic OHLCV data anchored on plausible NEPSE price ranges.
    Uses GBM + mean-reversion + volume regime model.
    """
    rng = np.random.default_rng(_seed_for(symbol))
    dates = nepal_trading_days(days)

    # Anchor prices per sector
    base_prices = {
        "Banking": 350, "Insurance": 650, "Hydropower": 180,
        "Finance": 120, "Telecom": 800, "Infrastructure": 200,
        "Microfinance": 1500,
    }
    sector = {
        "NABIL": "Banking", "NICA": "Banking", "SCB": "Banking", "SBI": "Banking",
        "EBL": "Banking", "ADBL": "Banking", "GBIME": "Banking",
        "NBL": "Banking", "PCBL": "Banking", "SRBL": "Banking",
        "KBL": "Banking", "MBL": "Banking", "PRVU": "Banking",
        "SANIMA": "Banking", "NMB": "Banking",
        "NLIC": "Insurance", "LICN": "Insurance", "ALICL": "Insurance",
        "PRIN": "Insurance", "SICL": "Insurance",
        "CHCL": "Hydropower", "BHPL": "Hydropower", "NHPC": "Hydropower",
        "BPCL": "Hydropower", "GHL": "Hydropower", "UPPER": "Hydropower",
        "AKJCL": "Hydropower", "MHNL": "Hydropower", "HPPL": "Hydropower",
        "BARUN": "Hydropower", "KPCL": "Hydropower", "SHL": "Hydropower",
        "NTC": "Telecom", "NICL": "Finance", "HIDCL": "Finance",
        "NIFRA": "Infrastructure", "API": "Finance",
        "SHIVM": "Microfinance", "RURU": "Microfinance", "HDL": "Finance",
    }
    base = base_prices.get(sector.get(symbol, "Banking"), 300)
    base *= rng.uniform(0.7, 1.5)

    mu     = rng.uniform(-0.0002, 0.0008)   # drift
    sigma  = rng.uniform(0.010, 0.022)      # daily vol
    prices = [base]

    # regime: 0=normal, 1=accumulation, 2=distribution
    regime = 0
    regime_counter = 0
    for i in range(1, days):
        regime_counter += 1
        if regime_counter > rng.integers(8, 20):
            regime = rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            regime_counter = 0
        drift_adj = mu + (0.0015 if regime == 1 else -0.0010 if regime == 2 else 0)
        shock = rng.normal(drift_adj, sigma)
        prices.append(max(10, prices[-1] * (1 + shock)))

    prices = np.array(prices)
    df = pd.DataFrame({"date": dates[:len(prices)]})
    df["close"] = np.round(prices, 2)

    intraday_range = prices * rng.uniform(0.005, 0.025, len(prices))
    df["open"]  = np.round(prices * (1 + rng.normal(0, 0.004, len(prices))), 2)
    df["high"]  = np.round(np.maximum(df["open"], df["close"]) + intraday_range * rng.uniform(0.3, 0.7, len(prices)), 2)
    df["low"]   = np.round(np.minimum(df["open"], df["close"]) - intraday_range * rng.uniform(0.3, 0.7, len(prices)), 2)
    df["low"]   = df["low"].clip(lower=1)

    # Volume with institutional spikes
    base_vol = rng.integers(50_000, 500_000)
    volumes = rng.integers(base_vol // 3, base_vol * 2, len(prices)).astype(float)
    spike_days = rng.choice(len(prices), size=max(3, len(prices) // 15), replace=False)
    volumes[spike_days] *= rng.uniform(3, 8, len(spike_days))
    df["volume"] = np.round(volumes).astype(int)

    df["turnover"] = np.round(df["close"] * df["volume"] / 1e6, 2)  # in millions
    df["symbol"]   = symbol
    df["date"]     = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def simulate_broker_data(symbol: str, n_transactions: int = 200) -> pd.DataFrame:
    """Simulate realistic floorsheet / broker transaction data."""
    rng = np.random.default_rng(_seed_for(symbol) + 1)
    broker_ids = list(BROKER_NAMES.keys())

    # Pick 5–10 dominant brokers per stock
    dominant = rng.choice(broker_ids, size=rng.integers(5, 11), replace=False).tolist()
    dominant_weight = 0.65

    def _pick_broker():
        if rng.random() < dominant_weight:
            return int(rng.choice(dominant))
        return int(rng.choice(broker_ids))

    ohlcv = simulate_ohlcv(symbol, days=30)
    price_range = ohlcv["close"].values

    rows = []
    for i in range(n_transactions):
        price = float(rng.choice(price_range)) * rng.uniform(0.98, 1.02)
        quantity = int(rng.choice([
            rng.integers(100, 1000),
            rng.integers(1000, 10000),
            rng.integers(10000, 100000),
        ], p=[0.65, 0.28, 0.07]))
        rows.append({
            "transaction_no": 1000000 + i,
            "symbol": symbol,
            "buyer_broker": _pick_broker(),
            "seller_broker": _pick_broker(),
            "quantity": quantity,
            "rate": round(price, 2),
            "amount": round(price * quantity, 2),
            "date": (now_npt() - timedelta(days=rng.integers(0, 5))).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def simulate_market_overview() -> pd.DataFrame:
    """Simulate today's market overview for all NEPSE symbols."""
    rows = []
    for sym in NEPSE_SYMBOLS:
        ohlcv = simulate_ohlcv(sym, days=5)
        last = ohlcv.iloc[-1]
        prev = ohlcv.iloc[-2] if len(ohlcv) > 1 else last
        chg = last["close"] - prev["close"]
        rows.append({
            "symbol": sym,
            "ltp": last["close"],
            "open": last["open"],
            "high": last["high"],
            "low": last["low"],
            "close": last["close"],
            "prev_close": prev["close"],
            "change": round(chg, 2),
            "change_pct": round(safe_pct(chg, prev["close"]), 2),
            "volume": last["volume"],
            "turnover": last["turnover"],
        })
    return pd.DataFrame(rows)


def safe_pct(num, den):
    return (num / den * 100) if den else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Orchestrator (try live → fallback to simulation)
# ══════════════════════════════════════════════════════════════════════════════

@timed
def get_market_overview() -> Tuple[pd.DataFrame, str]:
    """Return (dataframe, source_label)."""
    # 1. Try ShareSansar today price
    df = fetch_sharesansar_today_price()
    if df is not None and len(df) > 5:
        return df, "ShareSansar (live)"

    # 2. Try NEPSE API
    df = fetch_nepse_market_summary()
    if df is not None and len(df) > 5:
        return df, "NEPSE API (live)"

    # 3. Simulation fallback
    df = simulate_market_overview()
    return df, "Simulation (fallback)"


@timed
def get_stock_history(symbol: str, days: int = 90) -> Tuple[pd.DataFrame, str]:
    """Return (ohlcv_df, source_label)."""
    # 1. Try NEPSE official
    df = fetch_nepse_price_history(symbol, days)
    if df is not None and len(df) >= 10:
        return df, "NEPSE API (live)"

    # 2. Try NepseAlpha
    df = fetch_nepsealpha_data(symbol)
    if df is not None and len(df) >= 10:
        return df, "NepseAlpha (live)"

    # 3. Simulation
    df = simulate_ohlcv(symbol, days)
    return df, "Simulation (fallback)"


@timed
def get_broker_data(symbol: str) -> Tuple[pd.DataFrame, str]:
    """Return (broker_df, source_label)."""
    df = fetch_nepsealpha_broker_data(symbol)
    if df is not None and len(df) >= 10:
        return df, "NepseAlpha (live)"

    df = simulate_broker_data(symbol, n_transactions=300)
    return df, "Simulation (fallback)"


# ── HTML table parser helper ───────────────────────────────────────────────────
def _parse_html_table(table) -> pd.DataFrame:
    rows = []
    header_cells = table.find("thead")
    hdrs = []
    if header_cells:
        hdrs = [th.get_text(strip=True) for th in header_cells.find_all(["th", "td"])]
    body = table.find("tbody")
    if body is None:
        all_rows = table.find_all("tr")
        for tr in all_rows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
    else:
        for tr in body.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    if hdrs and len(hdrs) == max_cols:
        return pd.DataFrame(rows, columns=hdrs)
    return pd.DataFrame(rows)


logger.info("Data fetcher module loaded ✓")
