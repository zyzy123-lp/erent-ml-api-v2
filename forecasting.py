"""
e-Rent forecasting engine (para sa Admin Analytics).

Ang PHP ay nagpapadala ng buwanang time series (bookings, revenue, bawat equipment).
Para sa BAWAT series, sinusubukan ng engine ang ilang model, tinitingnan kung alin ang
pinakamahusay sa "rolling-origin backtest" (hulaan ang bawat buwan gamit lang ang mga naunang
buwan), at ang panalo ang gagamitin sa totoong forecast. Kaya hindi ito nagpapanggap na
magaling kapag kaunti pa ang data: awtomatiko itong bumabalik sa simpleng baseline.

Mga candidate:
  mean3       - average ng huling 3 buwan (flat)
  wma_trend   - weighted moving average + damped trend (katumbas ng dating PHP formula)
  ridge_ar    - Ridge regression sa lag1..lag3 + trend      (kailangan ng >= 8 buwan)
  gbr_season  - GradientBoosting sa lags + buwan ng taon   (kailangan ng >= 24 buwan)
"""
from __future__ import annotations

import math
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

MIN_BACKTEST_POINTS = 3     # kulang sa backtest -> baseline lang, "Very low" ang confidence
MAX_BACKTEST_POINTS = 8     # ilang huling buwan ang ginagamit sa backtest
LAGS = 3


# ----------------------------------------------------------------- helpers
def next_months(last_key: str, h: int) -> List[str]:
    y, m = int(last_key[:4]), int(last_key[5:7])
    out = []
    for _ in range(h):
        m += 1
        if m > 12:
            y, m = y + 1, 1
        out.append(f"{y:04d}-{m:02d}")
    return out


def _month_of(key: str) -> int:
    return int(key[5:7])


# --------------------------------------------------------- candidate models
# Bawat model: fn(values: List[float], keys: List[str], future_keys: List[str]) -> List[float]
def m_mean3(vals, keys, fkeys):
    base = float(np.mean(vals[-3:])) if vals else 0.0
    return [base] * len(fkeys)


def m_wma_trend(vals, keys, fkeys):
    n = len(vals)
    if n == 0:
        return [0.0] * len(fkeys)
    last = vals[-3:]
    w = np.arange(1, len(last) + 1)
    wma = float(np.dot(last, w) / w.sum())
    win = vals[-6:]
    slope = 0.0
    if len(win) >= 3:
        x = np.arange(len(win))
        slope = float(np.polyfit(x, win, 1)[0])
    return [wma + slope * 0.5 * (i + 1) for i in range(len(fkeys))]


def _lag_row(hist: List[float], t: int, month: int, seasonal: bool) -> List[float]:
    row = [hist[-k] for k in range(1, LAGS + 1)] + [float(t)]
    if seasonal:
        row += [math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)]
    return row


def _recursive(model_factory: Callable, vals, keys, fkeys, seasonal: bool):
    n = len(vals)
    if n <= LAGS + 2:
        return None
    X, y = [], []
    for i in range(LAGS, n):
        X.append(_lag_row(vals[:i], i, _month_of(keys[i]), seasonal))
        y.append(vals[i])
    X, y = np.array(X), np.array(y)
    model = model_factory()
    model.fit(X, y)
    hist = list(vals)
    out = []
    for j, fk in enumerate(fkeys):
        pred = float(model.predict(np.array([_lag_row(hist, n + j, _month_of(fk), seasonal)]))[0])
        pred = max(0.0, pred)
        out.append(pred)
        hist.append(pred)
    return out


def m_ridge_ar(vals, keys, fkeys):
    # Pinaliit ang trend feature (t) para hindi mag-extrapolate nang sobra sa kaunting data
    out = _recursive(lambda: Ridge(alpha=5.0), vals, keys, fkeys, seasonal=False)
    return out if out is not None else m_wma_trend(vals, keys, fkeys)


def m_gbr_season(vals, keys, fkeys):
    out = _recursive(
        lambda: GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.08,
                                          subsample=0.9, random_state=42),
        vals, keys, fkeys, seasonal=True)
    return out if out is not None else m_wma_trend(vals, keys, fkeys)


MODELS: Dict[str, Tuple[Callable, int]] = {      # name -> (fn, minimum na bilang ng buwan)
    "mean3": (m_mean3, 1),
    "wma_trend": (m_wma_trend, 3),
    "ridge_ar": (m_ridge_ar, 8),
    "gbr_season": (m_gbr_season, 24),
}

LABELS = {
    "mean3": "Baseline: current monthly pace (too little history for a trend)",
    "wma_trend": "Weighted moving average with damped trend",
    "ridge_ar": "Machine learning: Ridge regression on recent months and trend",
    "gbr_season": "Machine learning: Gradient boosting with seasonality",
}


# ---------------------------------------------------------------- backtest
def backtest(name: str, vals: List[float], keys: List[str]) -> Optional[List[float]]:
    """Mga error (actual - predicted) ng 1-step-ahead forecast sa mga huling buwan."""
    fn, min_n = MODELS[name]
    n = len(vals)
    start = max(min_n, n - MAX_BACKTEST_POINTS)
    if n - start < MIN_BACKTEST_POINTS:
        return None
    errs = []
    for i in range(start, n):
        try:
            pred = fn(vals[:i], keys[:i], [keys[i]])[0]
        except Exception:
            return None
        errs.append(vals[i] - max(0.0, pred))
    return errs


def confidence_label(n: int, rel_mae: Optional[float]) -> str:
    if n == 0:
        return "None"
    if n < 3 or rel_mae is None:
        return "Very low" if n < 3 else "Low"
    if n >= 24 and rel_mae < 0.35:
        return "High"
    if n >= 12 and rel_mae < 0.6:
        return "Medium"
    return "Low"


# --------------------------------------------------------------- public API
def forecast_series(keys: List[str], vals: List[float], horizon: int = 3,
                    integer: bool = False) -> dict:
    n = len(vals)
    if n == 0 or not keys:
        return {"method": "No data", "model": "none", "confidence": "None", "history_months": 0,
                "backtest_mae": None, "points": []}

    fkeys = next_months(keys[-1], horizon)
    scores: Dict[str, dict] = {}
    for name, (_, min_n) in MODELS.items():
        if n < min_n:
            continue
        errs = backtest(name, vals, keys)
        if errs:
            scores[name] = {"mae": float(np.mean(np.abs(errs))), "sd": float(np.std(errs, ddof=1)) if len(errs) > 1 else float(abs(errs[0]))}

    if scores:
        # Pumili ng may pinakamababang MAE. Kapag halos magkapareho (<5%), piliin ang mas simple.
        order = list(MODELS.keys())
        best = min(scores, key=lambda k: (round(scores[k]["mae"], 4), order.index(k)))
        for k in order:
            if k in scores and scores[k]["mae"] <= scores[best]["mae"] * 1.05:
                best = k
                break
        mae, sd = scores[best]["mae"], scores[best]["sd"]
        chosen = best
    else:
        chosen = "mean3" if n < 3 else "wma_trend"
        mae = sd = None

    preds = MODELS[chosen][0](vals, keys, fkeys)

    mean_level = float(np.mean(vals[-6:])) if vals else 0.0
    rel = (mae / mean_level) if (mae is not None and mean_level > 0) else None
    if sd is None:                      # walang backtest: gamitin ang spread ng history
        sd = float(np.std(vals, ddof=1)) if n > 1 else max(1.0, mean_level * 0.5)
    band0 = max(1.0 if not integer else 1.0, sd)

    points = []
    for i, (k, p) in enumerate(zip(fkeys, preds)):
        p = max(0.0, p)
        band = band0 * math.sqrt(i + 1) * 0.8       # lumalapad ang range habang lumalayo
        points.append({"month": k, "value": round(p, 1),
                       "low": round(max(0.0, p - band), 1), "high": round(p + band, 1)})

    return {
        "method": LABELS[chosen] if scores or n >= 3 else LABELS["mean3"],
        "model": chosen,
        "confidence": confidence_label(n, rel),
        "history_months": n,
        "backtest_mae": None if mae is None else round(mae, 2),
        "backtest_mape_pct": None if rel is None else round(rel * 100, 1),
        "points": points,
    }
