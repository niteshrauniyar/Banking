"""
data_cleaner.py
Robust cleaner / normaliser for messy NEPSE data from multiple sources.
"""

import re
import logging
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from utils import ensure_numeric, forward_fill_ohlcv, logger, SECTOR_MAP

# ── Column aliases from various sources ───────────────────────────────────────
OHLCV_ALIASES: Dict[str, List[str]] = {
    "symbol":   ["symbol", "stock", "ticker", "scrip", "company", "Symbol", "Stock", "Scrip"],
    "date":     ["date", "Date", "businessDate", "tradeDate", "trade_date", "businessdate", "published_date"],
    "open":     ["open", "Open", "openPrice", "open_price", "o"],
    "high":     ["high", "High", "highPrice", "high_price", "h"],
    "low":      ["low", "Low", "lowPrice", "low_price", "l"],
    "close":    ["close", "Close", "closePrice", "close_price", "c", "ltp", "LTP", "lastTradedPrice", "last_traded_price"],
    "volume":   ["volume", "Volume", "totalTradedQuantity", "total_volume", "qty", "quantity", "tradedQty", "vol"],
    "turnover": ["turnover", "Turnover", "totalTradedValue", "total_traded_value", "amount", "tradedValue", "Amount"],
    "prev_close": ["prev_close", "previousClose", "previous_close", "prevClose", "prev_ltp"],
    "change":   ["change", "Change", "priceChange", "price_change"],
    "change_pct": ["change_pct", "percentChange", "percent_change", "pctChange", "changePercent"],
}

BROKER_ALIASES: Dict[str, List[str]] = {
    "transaction_no": ["transaction_no", "transactionNo", "sn", "SN", "No", "no"],
    "symbol":       ["symbol", "stock", "scrip", "Symbol", "Stock"],
    "buyer_broker": ["buyer_broker", "buyerBroker", "buyer", "BuyerBroker", "buyerMemberId", "buyer_member"],
    "seller_broker":["seller_broker", "sellerBroker", "seller", "SellerBroker", "sellerMemberId", "seller_member"],
    "quantity":     ["quantity", "Quantity", "qty", "Qty", "tradedQuantity"],
    "rate":         ["rate", "Rate", "price", "Price", "tradeRate"],
    "amount":       ["amount", "Amount", "value", "Value", "tradeValue"],
    "date":         ["date", "Date", "businessDate", "tradeDate"],
}


def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """Return first matching column name (case-insensitive prefix match)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        exact = cols_lower.get(alias.lower())
        if exact:
            return exact
    # partial match
    for alias in aliases:
        for col_lower, col in cols_lower.items():
            if alias.lower() in col_lower:
                return col
    return None


def _rename_columns(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    rename_map = {}
    for canonical, choices in aliases.items():
        found = _find_col(df, choices)
        if found and found != canonical:
            rename_map[found] = canonical
    return df.rename(columns=rename_map)


# ── String → numeric cleaner ──────────────────────────────────────────────────
def _parse_number(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    s = s.replace(",", "").replace(" ", "").replace("Rs.", "").replace("Rs", "")
    s = s.replace("NPR", "").replace("M", "e6").replace("K", "e3").replace("B", "e9")
    try:
        return float(s)
    except (ValueError, OverflowError):
        return np.nan


def _clean_numeric_col(series: pd.Series) -> pd.Series:
    return series.apply(_parse_number).astype(float)


# ── Date normaliser ───────────────────────────────────────────────────────────
_DATE_FMTS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%Y/%m/%d", "%d %b %Y", "%b %d, %Y", "%Y%m%d",
    "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
]


def _parse_date(val) -> pd.Timestamp:
    if pd.isna(val):
        return pd.NaT
    for fmt in _DATE_FMTS:
        try:
            return pd.to_datetime(str(val), format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(str(val), infer_datetime_format=True)
    except Exception:
        return pd.NaT


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def clean_ohlcv(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Normalise a raw OHLCV dataframe from any source."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Rename
    df = _rename_columns(df, OHLCV_ALIASES)

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            if col in ("open", "high", "low") and "close" in df.columns:
                df[col] = df["close"]
            elif col == "volume":
                df[col] = 0
            else:
                df[col] = np.nan

    # Clean numerics
    for col in ["open", "high", "low", "close", "volume", "turnover", "change", "change_pct", "prev_close"]:
        if col in df.columns:
            df[col] = _clean_numeric_col(df[col])

    # Fix OHLC logic violations
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"]  = df[["open", "high", "low", "close"]].min(axis=1)
    df = df[df["close"] > 0].copy()

    # Date
    if "date" in df.columns:
        df["date"] = df["date"].apply(_parse_date)
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
    else:
        from utils import nepal_trading_days
        dates = nepal_trading_days(len(df))
        df["date"] = pd.to_datetime(dates[:len(df)])

    # Symbol
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = symbol or "UNKNOWN"
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    # Turnover
    if "turnover" not in df.columns or df["turnover"].isna().all():
        df["turnover"] = (df["close"] * df["volume"]).round(2)

    # Forward-fill gaps
    df = forward_fill_ohlcv(df)

    # Derived columns
    df["returns"]      = df["close"].pct_change()
    df["log_returns"]  = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"]     = df["high"] - df["low"]
    df["body_size"]    = (df["close"] - df["open"]).abs()
    df["vwap"]         = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)

    # Rolling metrics
    df["vol_ma20"]     = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_ratio"]    = df["volume"] / (df["vol_ma20"] + 1)
    df["sma20"]        = df["close"].rolling(20, min_periods=1).mean()
    df["sma50"]        = df["close"].rolling(50, min_periods=1).mean()
    df["ema9"]         = df["close"].ewm(span=9, adjust=False).mean()

    logger.info(f"clean_ohlcv → {len(df)} rows for {df['symbol'].iloc[0] if len(df) else 'N/A'}")
    return df


def clean_broker_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw broker / floorsheet data."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = _rename_columns(df, BROKER_ALIASES)

    for col in ["quantity", "rate", "amount"]:
        if col in df.columns:
            df[col] = _clean_numeric_col(df[col])

    if "buyer_broker" in df.columns:
        df["buyer_broker"] = pd.to_numeric(df["buyer_broker"], errors="coerce")
    if "seller_broker" in df.columns:
        df["seller_broker"] = pd.to_numeric(df["seller_broker"], errors="coerce")

    df = df.dropna(subset=["quantity", "rate"])
    df = df[df["quantity"] > 0]
    df = df[df["rate"] > 0]

    if "amount" not in df.columns or df["amount"].isna().all():
        df["amount"] = df["quantity"] * df["rate"]

    if "date" in df.columns:
        df["date"] = df["date"].apply(_parse_date)

    logger.info(f"clean_broker_data → {len(df)} rows")
    return df


def clean_market_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise market overview table."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = _rename_columns(df, OHLCV_ALIASES)

    numeric_cols = ["ltp", "open", "high", "low", "close", "prev_close",
                    "change", "change_pct", "volume", "turnover"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _clean_numeric_col(df[col])

    if "close" not in df.columns and "ltp" in df.columns:
        df["close"] = df["ltp"]

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df = df[df["symbol"].str.len() <= 12]
        df = df[df["symbol"].str.isalpha()]

    if "change_pct" not in df.columns:
        if "change" in df.columns and "prev_close" in df.columns:
            df["change_pct"] = (df["change"] / df["prev_close"].replace(0, np.nan) * 100).round(2)
        elif "close" in df.columns and "prev_close" in df.columns:
            df["change_pct"] = ((df["close"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan) * 100).round(2)

    # Add sector
    df["sector"] = df["symbol"].map(SECTOR_MAP).fillna("Other") if "symbol" in df.columns else "Other"

    df = df.dropna(subset=["close"]) if "close" in df.columns else df
    df = df[df.get("close", pd.Series([1])) > 0].reset_index(drop=True)

    logger.info(f"clean_market_overview → {len(df)} rows")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add full technical indicator suite to cleaned OHLCV dataframe."""
    if df.empty or "close" not in df.columns:
        return df

    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs   = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20

    # ATR
    tr1 = h - lo
    tr2 = (h - c.shift(1)).abs()
    tr3 = (lo - c.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # OBV
    direction = np.sign(c.diff().fillna(0))
    df["obv"] = (v * direction).cumsum()

    # Stochastic
    low14  = lo.rolling(14, min_periods=1).min()
    high14 = h.rolling(14, min_periods=1).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=1).mean()

    # Williams %R
    df["williams_r"] = -100 * (high14 - c) / (high14 - low14 + 1e-10)

    # CMF (Chaikin Money Flow)
    mfm = ((c - lo) - (h - c)) / (h - lo + 1e-10)
    mfv = mfm * v
    df["cmf"] = mfv.rolling(20, min_periods=1).sum() / (v.rolling(20, min_periods=1).sum() + 1)

    # Volume indicators
    df["force_index"]  = c.diff() * v
    df["vol_zscore"]   = (v - v.rolling(20, min_periods=1).mean()) / (v.rolling(20, min_periods=1).std() + 1)

    logger.info(f"Technical indicators added: {df['symbol'].iloc[0] if 'symbol' in df.columns else 'N/A'}")
    return df


logger.info("Data cleaner module loaded ✓")
