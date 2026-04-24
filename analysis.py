"""
analysis.py
Institutional-grade quantitative analysis engine for NEPSE.
Implements: liquidity metrics, order flow, broker intelligence,
network analysis, behavioural detection, and ML clustering.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import (
    safe_divide, rolling_zscore, percentile_rank, ewma, winsorise,
    atr, support_resistance, clamp, logger, BROKER_NAMES
)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 – Liquidity & Market Impact
# ══════════════════════════════════════════════════════════════════════════════

def amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio:
        ILLIQ_t = |r_t| / (Volume_t * Price_t)
    Higher = more illiquid (price moves more per unit volume).
    """
    r_abs   = df["returns"].abs()
    dv      = df["volume"] * df["close"]          # dollar volume
    illiq   = r_abs / (dv.replace(0, np.nan) / 1e6)  # scale to millions
    return illiq.rolling(window, min_periods=3).mean().rename("amihud_illiq")


def kyle_lambda(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Kyle's Lambda approximation: price impact per unit signed order flow.
    Using signed-volume proxy: direction = sign(close - open).
    """
    signed_vol = df["volume"] * np.sign(df["close"] - df["open"])
    price_chg  = df["close"].diff()
    lambdas = []
    for i in range(len(df)):
        start = max(0, i - window + 1)
        y = price_chg.iloc[start:i+1].dropna()
        x = signed_vol.iloc[start:i+1].loc[y.index]
        if len(y) >= 3 and x.std() > 0:
            slope, _, _, _, _ = stats.linregress(x, y)
            lambdas.append(abs(slope))
        else:
            lambdas.append(np.nan)
    return pd.Series(lambdas, index=df.index, name="kyle_lambda")


def bid_ask_spread_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Roll (1984) bid-ask spread proxy:
        spread = 2 * sqrt(-Cov(dp_t, dp_{t-1}))  if cov < 0 else 0
    """
    dp  = df["close"].diff()
    cov = dp.rolling(10, min_periods=3).cov(dp.shift(1))
    spread = 2 * np.sqrt((-cov).clip(lower=0))
    return spread.rename("bid_ask_spread")


def volume_spike_score(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
    """
    Normalised volume spike indicator.
    Returns z-score; values > threshold are flagged as spikes.
    """
    z = rolling_zscore(df["volume"], window)
    return z.rename("vol_spike_z")


def price_impact_score(df: pd.DataFrame) -> pd.Series:
    """Composite price-impact score [0–100]."""
    illiq = amihud_illiquidity(df)
    lam   = kyle_lambda(df)
    sprd  = bid_ask_spread_proxy(df)

    illiq_n = percentile_rank(illiq.fillna(illiq.median()))
    lam_n   = percentile_rank(lam.fillna(lam.median()))
    sprd_n  = percentile_rank(sprd.fillna(sprd.median()))
    return ((illiq_n + lam_n + sprd_n) / 3).rename("price_impact_score")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 – Order Flow Intelligence
# ══════════════════════════════════════════════════════════════════════════════

def signed_volume(df: pd.DataFrame) -> pd.Series:
    """
    Tick-rule signed volume: buy-initiated if close > open.
    Positive = buying pressure, negative = selling pressure.
    """
    direction = np.where(df["close"] > df["open"], 1,
                np.where(df["close"] < df["open"], -1, 0))
    return (df["volume"] * direction).rename("signed_volume")


def order_flow_imbalance(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Order flow imbalance OFI = Σ signed_vol / Σ total_vol  over window.
    Range [-1, +1]; positive = buy-side dominated.
    """
    sv   = signed_volume(df)
    net  = sv.rolling(window, min_periods=1).sum()
    tot  = df["volume"].rolling(window, min_periods=1).sum()
    return (net / (tot + 1)).rename("ofi")


def order_flow_autocorr(df: pd.DataFrame, lag: int = 1, window: int = 20) -> pd.Series:
    """Autocorrelation of signed volume – detects metaorder splitting."""
    sv   = signed_volume(df)
    corr = sv.rolling(window, min_periods=5).apply(
        lambda x: x.autocorr(lag=lag) if len(x) > lag + 2 else 0, raw=False
    )
    return corr.rename("of_autocorr")


def detect_metaorder_splitting(df: pd.DataFrame) -> Dict[str, float]:
    """
    Detect institutional order splitting:
    - Persistent positive OFI autocorrelation
    - Compressed intraday range during accumulation
    - Unusual volume-to-price ratio
    """
    ofi      = order_flow_imbalance(df).tail(10)
    autocorr = order_flow_autocorr(df).tail(10)
    ofi_mean = float(ofi.mean())
    ac_mean  = float(autocorr.mean())

    # Conceal-within-range pattern: high volume with tight range
    recent = df.tail(10)
    vol_mean   = float(recent["volume"].mean())
    range_mean = float(recent["hl_range"].mean()) if "hl_range" in recent.columns else float((recent["high"] - recent["low"]).mean())
    close_mean = float(recent["close"].mean())
    vol_per_range = safe_divide(vol_mean, range_mean / (close_mean + 0.01))

    splitting_score = clamp(
        (abs(ac_mean) * 30) + (abs(ofi_mean) * 40) + min(vol_per_range / 10000, 30),
        0, 100
    )
    return {
        "ofi_mean":          round(ofi_mean, 4),
        "autocorr_mean":     round(ac_mean, 4),
        "vol_per_range":     round(vol_per_range, 2),
        "splitting_score":   round(splitting_score, 1),
        "is_splitting":      splitting_score > 45,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 – Trade Size Analysis
# ══════════════════════════════════════════════════════════════════════════════

def classify_trades(broker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each trade into Retail / Semi-institutional / Institutional
    using 75th / 95th percentile cutoffs.
    """
    if broker_df.empty or "quantity" not in broker_df.columns:
        return broker_df

    p75 = broker_df["quantity"].quantile(0.75)
    p95 = broker_df["quantity"].quantile(0.95)

    def _classify(qty):
        if qty >= p95:
            return "Institutional"
        elif qty >= p75:
            return "Semi-Institutional"
        return "Retail"

    df = broker_df.copy()
    df["trade_class"] = df["quantity"].apply(_classify)
    df["trade_value"]  = df["quantity"] * df["rate"]
    return df


def institutional_volume_pct(broker_df: pd.DataFrame) -> float:
    """Percentage of total volume from institutional trades."""
    classified = classify_trades(broker_df)
    if classified.empty:
        return 0.0
    inst = classified[classified["trade_class"] == "Institutional"]["quantity"].sum()
    total = classified["quantity"].sum()
    return round(safe_divide(inst, total) * 100, 2)


def large_trade_clusters(broker_df: pd.DataFrame) -> pd.DataFrame:
    """Identify clusters of large trades (potential block trades)."""
    classified = classify_trades(broker_df)
    large = classified[classified["trade_class"] == "Institutional"].copy()
    if large.empty or len(large) < 3:
        return pd.DataFrame()

    # Group trades within 2% price range
    large = large.sort_values("rate")
    large["price_bucket"] = (large["rate"] // (large["rate"].mean() * 0.02)).astype(int)
    clusters = large.groupby("price_bucket").agg(
        n_trades=("quantity", "count"),
        total_qty=("quantity", "sum"),
        total_value=("trade_value", "sum"),
        avg_price=("rate", "mean"),
        buyer_brokers=("buyer_broker", "nunique"),
        seller_brokers=("seller_broker", "nunique"),
    ).reset_index()
    clusters = clusters[clusters["n_trades"] >= 2].sort_values("total_value", ascending=False)
    return clusters


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 – Broker Intelligence
# ══════════════════════════════════════════════════════════════════════════════

def broker_net_flow(broker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Net flow per broker: buy_qty - sell_qty and buy_val - sell_val.
    Positive = net buyer, negative = net seller.
    """
    if broker_df.empty:
        return pd.DataFrame()

    classified = classify_trades(broker_df)
    buy_side  = classified.groupby("buyer_broker").agg(
        buy_qty=("quantity", "sum"), buy_val=("trade_value", "sum")
    )
    sell_side = classified.groupby("seller_broker").agg(
        sell_qty=("quantity", "sum"), sell_val=("trade_value", "sum")
    )
    flow = buy_side.join(sell_side, how="outer").fillna(0)
    flow["net_qty"] = flow["buy_qty"] - flow["sell_qty"]
    flow["net_val"] = flow["buy_val"] - flow["sell_val"]
    flow["net_qty_pct"] = safe_divide(flow["net_qty"], flow["buy_qty"] + flow["sell_qty"]) * 100
    flow["broker_name"] = flow.index.map(lambda x: BROKER_NAMES.get(int(x) if not np.isnan(x) else -1, f"Broker {x}"))
    flow = flow.reset_index().rename(columns={"index": "broker_id"})
    flow = flow.sort_values("net_val", ascending=False)
    return flow


def aggressive_accumulation(broker_df: pd.DataFrame, top_n: int = 5) -> Dict:
    """
    Detect aggressive accumulation: brokers buying significantly more than selling.
    """
    flow = broker_net_flow(broker_df)
    if flow.empty:
        return {}

    total_turnover = float((broker_df["quantity"] * broker_df["rate"]).sum()) if "quantity" in broker_df else 1
    accumulators   = flow[flow["net_val"] > 0].head(top_n)
    distributors   = flow[flow["net_val"] < 0].tail(top_n)

    acc_val = float(accumulators["net_val"].sum())
    dis_val = float(distributors["net_val"].sum())

    imbalance_score = clamp(
        safe_divide(abs(acc_val - abs(dis_val)), total_turnover + 1) * 200,
        0, 100
    )
    return {
        "top_accumulators":   accumulators[["broker_name", "net_qty", "net_val"]].to_dict("records"),
        "top_distributors":   distributors[["broker_name", "net_qty", "net_val"]].to_dict("records"),
        "accumulation_score": round(imbalance_score, 1),
        "is_accumulating":    acc_val > abs(dis_val) and imbalance_score > 30,
    }


def dominant_brokers(broker_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank brokers by total activity (buy + sell volume)."""
    flow = broker_net_flow(broker_df)
    if flow.empty:
        return pd.DataFrame()
    flow["total_qty"] = flow["buy_qty"] + flow["sell_qty"]
    flow["total_val"] = flow["buy_val"] + flow["sell_val"]
    return flow.nlargest(top_n, "total_val")[
        ["broker_name", "buy_qty", "sell_qty", "net_qty", "total_val", "net_val"]
    ].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 – Network Analysis (Simplified)
# ══════════════════════════════════════════════════════════════════════════════

def broker_network_score(broker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate broker centrality:
    - Degree centrality: number of unique counterparties
    - Volume-weighted centrality: total traded value normalised
    """
    if broker_df.empty:
        return pd.DataFrame()

    classified = classify_trades(broker_df)
    # Build adjacency: buyer–seller pairs
    edges = classified[["buyer_broker", "seller_broker", "trade_value"]].dropna()

    if edges.empty:
        return pd.DataFrame()

    all_brokers = pd.concat([edges["buyer_broker"], edges["seller_broker"]]).dropna().unique()
    scores = []
    for b in all_brokers:
        as_buyer  = edges[edges["buyer_broker"] == b]
        as_seller = edges[edges["seller_broker"] == b]
        counterparties = pd.concat([
            as_buyer["seller_broker"], as_seller["buyer_broker"]
        ]).nunique()
        total_val = float(as_buyer["trade_value"].sum() + as_seller["trade_value"].sum())
        scores.append({
            "broker_id":          b,
            "broker_name":        BROKER_NAMES.get(int(b) if not np.isnan(b) else -1, f"Broker {b}"),
            "degree_centrality":  counterparties,
            "volume_centrality":  total_val,
        })

    df = pd.DataFrame(scores)
    df["centrality_score"] = (
        percentile_rank(df["degree_centrality"]) * 0.4 +
        percentile_rank(df["volume_centrality"]) * 0.6
    ).round(1)
    return df.sort_values("centrality_score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 – Behavioural Detection (Smart Money)
# ══════════════════════════════════════════════════════════════════════════════

def smart_money_score(df: pd.DataFrame, broker_df: pd.DataFrame) -> Dict:
    """
    Composite smart money score [0–100].
    Combines: OFI, volume-price divergence, broker accumulation,
    price-structure, and volatility compression signals.
    """
    scores = {}

    # 1. Order flow imbalance (recent 10 days)
    ofi = order_flow_imbalance(df).tail(10).mean()
    scores["ofi"] = clamp((float(ofi) + 1) / 2 * 100, 0, 100)

    # 2. Volume-price divergence (smart money hides in tight range)
    recent = df.tail(10)
    vol_trend   = stats.linregress(range(len(recent)), recent["volume"].values)[0]
    price_trend = stats.linregress(range(len(recent)), recent["close"].values)[0]
    # High volume + flat/slight price = accumulation
    vp_diverge = clamp(
        50 + (vol_trend / (df["volume"].std() + 1)) * 25 - (price_trend / (df["close"].std() + 0.01)) * 10,
        0, 100
    )
    scores["vp_divergence"] = round(vp_diverge, 1)

    # 3. Broker accumulation
    acc = aggressive_accumulation(broker_df)
    scores["broker_acc"] = float(acc.get("accumulation_score", 50))

    # 4. Volatility compression (Bollinger squeeze proxy)
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_width = ((df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)).tail(10)
        bb_compress = clamp(100 - float(percentile_rank(bb_width).mean()), 0, 100)
    else:
        bb_compress = 50
    scores["bb_compression"] = bb_compress

    # 5. OBV trend (institutional distribution/accumulation)
    if "obv" in df.columns:
        obv_trend = stats.linregress(range(len(df.tail(20))), df["obv"].tail(20).values)[0]
        obv_score = clamp(50 + (obv_trend / (df["obv"].std() + 1)) * 20, 0, 100)
    else:
        obv_score = 50
    scores["obv_trend"] = round(obv_score, 1)

    # Weighted composite
    weights = {"ofi": 0.20, "vp_divergence": 0.20, "broker_acc": 0.30, "bb_compression": 0.15, "obv_trend": 0.15}
    composite = sum(scores[k] * weights[k] for k in weights)

    return {
        "components":   scores,
        "composite":    round(composite, 1),
        "is_smart_money_accumulating": composite > 60,
    }


def detect_distribution_phase(df: pd.DataFrame) -> Dict:
    """
    Wyckoff-style distribution detection:
    - Price near recent highs
    - Volume drying up after spike
    - Bearish divergences
    """
    if len(df) < 20:
        return {"phase": "UNKNOWN", "score": 50}

    recent = df.tail(20)
    price_near_high = float(recent["close"].iloc[-1]) >= float(recent["close"].quantile(0.85))
    vol_declining   = stats.linregress(range(10), recent["volume"].tail(10).values)[0] < 0
    rsi_bearish     = float(recent["rsi"].iloc[-1]) < 55 if "rsi" in recent.columns else False
    macd_cross      = (
        recent["macd"].iloc[-1] < recent["macd_signal"].iloc[-1]
        if "macd" in recent.columns and "macd_signal" in recent.columns else False
    )
    dist_signals = sum([price_near_high, vol_declining, rsi_bearish, macd_cross])
    dist_score   = dist_signals * 25.0

    return {
        "phase":              "DISTRIBUTION" if dist_score > 50 else "ACCUMULATION" if dist_score < 25 else "MARKUP/MARKDOWN",
        "score":              round(dist_score, 1),
        "price_near_high":    price_near_high,
        "volume_declining":   vol_declining,
        "rsi_bearish":        rsi_bearish,
        "macd_bearish_cross": macd_cross,
    }


def detect_inducement_trap(df: pd.DataFrame) -> Dict:
    """
    ICT-style inducement/trap detection:
    - False breakout above recent high (bull trap) → bearish
    - False breakdown below recent low (bear trap) → bullish
    """
    if len(df) < 20:
        return {"trap": "NONE", "direction": None, "confidence": 0}

    recent = df.tail(20)
    recent_high  = float(recent["high"].iloc[:-3].max())
    recent_low   = float(recent["low"].iloc[:-3].min())
    last_3       = df.tail(3)

    bull_trap = (
        float(last_3["high"].max()) > recent_high and
        float(last_3["close"].iloc[-1]) < recent_high and
        float(last_3["volume"].mean()) < float(df["volume"].tail(20).mean())
    )
    bear_trap = (
        float(last_3["low"].min()) < recent_low and
        float(last_3["close"].iloc[-1]) > recent_low and
        float(last_3["volume"].mean()) < float(df["volume"].tail(20).mean())
    )

    if bull_trap:
        return {"trap": "BULL TRAP", "direction": "BEARISH", "confidence": 72,
                "level": round(recent_high, 2)}
    elif bear_trap:
        return {"trap": "BEAR TRAP", "direction": "BULLISH", "confidence": 70,
                "level": round(recent_low, 2)}
    return {"trap": "NONE", "direction": None, "confidence": 0}


def liquidity_sweep_detection(df: pd.DataFrame) -> Dict:
    """
    Detect stop-hunt / liquidity sweep:
    High wick rejection above resistance or low wick rejection below support.
    """
    if len(df) < 5:
        return {"sweep": False}

    last = df.iloc[-1]
    prev = df.tail(20).iloc[:-1]

    body_size  = abs(float(last["close"] - last["open"]))
    upper_wick = float(last["high"]) - max(float(last["open"]), float(last["close"]))
    lower_wick = min(float(last["open"]), float(last["close"])) - float(last["low"])
    atr_val    = float(df["atr"].iloc[-1]) if "atr" in df.columns else body_size

    upper_sweep = upper_wick > atr_val * 1.5 and upper_wick > body_size * 2
    lower_sweep = lower_wick > atr_val * 1.5 and lower_wick > body_size * 2

    # Check if it breached prior highs/lows then reversed
    prior_high = float(prev["high"].max())
    prior_low  = float(prev["low"].min())
    swept_high = float(last["high"]) > prior_high and float(last["close"]) < prior_high
    swept_low  = float(last["low"]) < prior_low and float(last["close"]) > prior_low

    if swept_high or upper_sweep:
        return {"sweep": True, "type": "SELL-SIDE LIQUIDITY SWEEP", "bias": "BEARISH",
                "level": round(prior_high, 2)}
    elif swept_low or lower_sweep:
        return {"sweep": True, "type": "BUY-SIDE LIQUIDITY SWEEP", "bias": "BULLISH",
                "level": round(prior_low, 2)}
    return {"sweep": False, "type": None, "bias": None}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 – Machine Learning (KMeans Clustering)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_market_regimes(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    KMeans clustering on [returns, volatility, volume_ratio, rsi, ofi]
    to identify market regimes: Accumulation / Trending / Distribution
    """
    feature_cols = []
    potential = {
        "returns":    df.get("returns"),
        "vol_ratio":  df.get("vol_ratio"),
        "rsi":        df.get("rsi"),
        "vol_zscore": df.get("vol_spike_z") if "vol_spike_z" in df.columns else df.get("vol_zscore"),
    }
    # Compute OFI
    ofi = order_flow_imbalance(df)
    potential["ofi"] = ofi

    feat_df = pd.DataFrame(potential).dropna()
    if len(feat_df) < n_clusters * 3:
        df["regime"] = "INSUFFICIENT DATA"
        df["regime_id"] = 0
        return df

    scaler  = StandardScaler()
    X       = scaler.fit_transform(feat_df)
    kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels  = kmeans.fit_predict(X)
    feat_df["cluster"] = labels

    # Label clusters by average returns + vol_ratio
    cluster_means = feat_df.groupby("cluster")[["returns", "vol_ratio"]].mean()
    sorted_clusters = cluster_means.sort_values("returns")

    label_map = {}
    for i, idx in enumerate(sorted_clusters.index):
        if i == 0:
            label_map[idx] = "DISTRIBUTION / DOWNTREND"
        elif i == n_clusters - 1:
            label_map[idx] = "ACCUMULATION / UPTREND"
        else:
            label_map[idx] = "CONSOLIDATION"

    feat_df["regime"] = feat_df["cluster"].map(label_map)
    df = df.copy()
    df.loc[feat_df.index, "regime"] = feat_df["regime"]
    df.loc[feat_df.index, "regime_id"] = feat_df["cluster"]
    df["regime"] = df["regime"].fillna("UNKNOWN")
    return df


def detect_anomalous_trading(df: pd.DataFrame, broker_df: pd.DataFrame) -> Dict:
    """
    Detect unusual trading using Isolation Forest-style scoring
    (approximated with z-score + percentile rank combination).
    """
    anomaly_signals = []

    # Volume anomaly
    if "vol_zscore" in df.columns:
        vol_z = float(df["vol_zscore"].tail(5).mean())
        if abs(vol_z) > 2.5:
            anomaly_signals.append(f"Abnormal volume (z={vol_z:.1f})")

    # Price-volume divergence anomaly
    if len(df) >= 5:
        recent = df.tail(5)
        pv_corr = recent["close"].corr(recent["volume"])
        if abs(pv_corr) > 0.85:
            anomaly_signals.append(f"Strong price-volume correlation ({pv_corr:.2f}) – institutional footprint")

    # Broker concentration anomaly
    if not broker_df.empty and "quantity" in broker_df.columns:
        top_broker_share = float(
            broker_df.groupby("buyer_broker")["quantity"].sum().nlargest(1).iloc[0]
            / (broker_df["quantity"].sum() + 1) * 100
        )
        if top_broker_share > 35:
            anomaly_signals.append(f"Single broker controls {top_broker_share:.0f}% of buy volume")

    anomaly_score = min(len(anomaly_signals) * 33, 100)
    return {
        "anomaly_score":   anomaly_score,
        "signals":         anomaly_signals,
        "is_anomalous":    anomaly_score >= 33,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 – Volume Profile
# ══════════════════════════════════════════════════════════════════════════════

def compute_volume_profile(df: pd.DataFrame, n_bins: int = 20) -> Dict:
    """
    Point of Control (POC), Value Area High/Low (VAH/VAL).
    70% of volume falls within VAH–VAL.
    """
    if df.empty or "volume" not in df.columns:
        close = df["close"].iloc[-1] if not df.empty else 100
        return {"poc": close, "vah": close * 1.02, "val": close * 0.98, "profile": pd.DataFrame()}

    lo_price = float(df["low"].min())
    hi_price = float(df["high"].max())
    bins     = np.linspace(lo_price, hi_price, n_bins + 1)
    vol_at_price = np.zeros(n_bins)

    for _, row in df.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        in_range = [(lo <= (bins[i] + bins[i+1]) / 2 <= hi) for i in range(n_bins)]
        if sum(in_range) > 0:
            per_bin = vol / max(sum(in_range), 1)
            for i, flag in enumerate(in_range):
                if flag:
                    vol_at_price[i] += per_bin

    poc_idx   = int(np.argmax(vol_at_price))
    poc_price = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)

    total_vol  = float(vol_at_price.sum())
    target_vol = total_vol * 0.70
    sorted_idx = np.argsort(vol_at_price)[::-1]
    included, acc = [], 0
    for i in sorted_idx:
        acc += vol_at_price[i]
        included.append(i)
        if acc >= target_vol:
            break

    vah = float((bins[max(included)] + bins[max(included) + 1]) / 2)
    val = float((bins[min(included)] + bins[min(included) + 1]) / 2)

    profile = pd.DataFrame({
        "price_level": [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)],
        "volume":      vol_at_price,
        "is_value_area": [i in included for i in range(n_bins)],
    })

    return {"poc": round(poc_price, 2), "vah": round(vah, 2), "val": round(val, 2), "profile": profile}


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ANALYSIS RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(df: pd.DataFrame, broker_df: pd.DataFrame, symbol: str) -> Dict:
    """
    Run the complete institutional analysis pipeline.
    Returns a structured dict with all metrics.
    """
    result = {"symbol": symbol, "error": None}
    try:
        # Price impact & liquidity
        result["amihud"]         = amihud_illiquidity(df).iloc[-1] if len(df) >= 5 else np.nan
        result["kyle_lambda"]    = kyle_lambda(df).iloc[-1] if len(df) >= 5 else np.nan
        result["bid_ask_spread"] = bid_ask_spread_proxy(df).iloc[-1] if len(df) >= 5 else np.nan
        result["vol_spike_z"]    = volume_spike_score(df).iloc[-1] if len(df) >= 5 else 0

        # Order flow
        result["ofi"]             = float(order_flow_imbalance(df).iloc[-1]) if len(df) >= 3 else 0
        result["metaorder"]       = detect_metaorder_splitting(df)

        # Trade size
        result["inst_vol_pct"]    = institutional_volume_pct(broker_df)
        result["large_clusters"]  = large_trade_clusters(broker_df)

        # Broker intelligence
        result["broker_flow"]     = broker_net_flow(broker_df)
        result["accumulation"]    = aggressive_accumulation(broker_df)
        result["dominant"]        = dominant_brokers(broker_df)
        result["network"]         = broker_network_score(broker_df)

        # Behavioural
        result["smart_money"]     = smart_money_score(df, broker_df)
        result["distribution"]    = detect_distribution_phase(df)
        result["trap"]            = detect_inducement_trap(df)
        result["liq_sweep"]       = liquidity_sweep_detection(df)

        # ML
        result["regimes"]         = cluster_market_regimes(df.copy())
        result["anomaly"]         = detect_anomalous_trading(df, broker_df)

        # Volume profile
        result["vol_profile"]     = compute_volume_profile(df)

        # Support / Resistance
        sr = support_resistance(df["close"])
        result["support"]         = sr["support"]
        result["resistance"]      = sr["resistance"]

    except Exception as exc:
        logger.error(f"Analysis error for {symbol}: {exc}", exc_info=True)
        result["error"] = str(exc)

    return result


logger.info("Analysis module loaded ✓")
