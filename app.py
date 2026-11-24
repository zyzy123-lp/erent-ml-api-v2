"""
e-Rent ML API (v2.1): package ranking (/recommend) + admin analytics forecasting (/forecast)

Ang PHP ay nagpapadala ng LISTAHAN NG CANDIDATE PACKAGES (kasama ang presyo, event_type,
kapasidad ng upuan, items). Ang model ay nagbibigay ng score sa bawat candidate. Hindi na
nito kinakabisa ang package IDs, kaya hindi na ito masisira kapag nagdagdag o nagbura kayo
ng package.

TRAINING DATA
  1. data/real_training.csv  -> totoong data galing sa package_recommendations (kung meron)
  2. synthetic seed data     -> para may mapagsimulan habang kaunti pa ang totoong data
Ilagay sa defense na synthetic ang seed data. Habang dumarami ang totoong rows, mas lumalakas
ang epekto nila (REAL_WEIGHT) at mas kaunti ang bahagi ng synthetic.
"""
import csv
import math
import os
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

from forecasting import forecast_series

app = FastAPI(title="e-Rent ML API", version="2.1")

EVENTS = ["Wedding", "Birthday", "Debut", "Corporate", "Other"]
REAL_CSV = os.path.join(os.path.dirname(__file__), "data", "real_training.csv")
REAL_WEIGHT = 10.0      # bawat totoong row ay katumbas ng 10 synthetic rows
N_SYNTHETIC = 4000
FEATURES = ["event_match", "cand_other", "budget_given", "price_ratio", "within_budget",
            "guests_given", "cap_ratio", "cap_ok", "cart_overlap", "popularity"]


# ------------------------------------------------------------------ features
def norm_event(e: str) -> str:
    return e if e in EVENTS else "Other"


def featurize(event, budget, guests, cart_ids, cand_event, price, chair_cap, cand_eq_ids, times_booked):
    event, cand_event = norm_event(event), norm_event(cand_event)
    budget_given = 1.0 if budget > 0 else 0.0
    price_ratio = min(price / budget, 3.0) if budget > 0 else 0.0
    within_budget = 1.0 if (budget <= 0 or price <= budget) else 0.0
    guests_given = 1.0 if guests > 0 else 0.0
    cap_ratio = min(chair_cap / guests, 3.0) if (guests > 0 and chair_cap > 0) else 0.0
    cap_ok = 1.0 if (guests <= 0 or chair_cap <= 0 or chair_cap >= guests) else 0.0
    cart = set(cart_ids)
    overlap = len(cart & set(cand_eq_ids)) / len(cart) if cart else 0.0
    popularity = min(times_booked, 20) / 20.0
    return [1.0 if event == cand_event else 0.0,
            1.0 if (cand_event == "Other" and event != "Other") else 0.0,
            budget_given, price_ratio, within_budget, guests_given, cap_ratio, cap_ok,
            overlap, popularity]


# ------------------------------------------------------------ training data
def synthetic_data(n: int, seed: int = 42):
    """Synthetic (request, candidate) pairs. Ang label ay galing sa isang may-ingay na
    preference function (event match, budget fit, upuan para sa bisita, popularity)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        event = EVENTS[rng.integers(0, len(EVENTS))]
        guests = int(rng.integers(20, 300)) if rng.random() < 0.75 else 0
        budget = float(np.exp(rng.normal(9.8, 0.7))) if rng.random() < 0.75 else 0.0   # ~5k-60k
        n_cart = int(rng.integers(0, 6))
        cart = [int(v) for v in rng.integers(1, 140, n_cart)]
        cand_event = EVENTS[rng.integers(0, len(EVENTS))]
        price = float(np.exp(rng.normal(9.8, 0.6)))
        cap = int(rng.integers(0, 300)) if rng.random() < 0.85 else 0
        cand_eq = [int(v) for v in rng.integers(1, 140, int(rng.integers(5, 35)))]
        booked = int(rng.integers(0, 25))
        f = featurize(event, budget, guests, cart, cand_event, price, cap, cand_eq, booked)

        s = 0.0
        s += 2.0 if f[0] else (0.6 if f[1] else 0.0)
        if f[2]:
            s += 1.5 * min(f[3], 1.0) if f[4] else -3.0
        if f[5] and f[6] > 0:
            s += 1.0 if f[7] else -2.5 * (1.0 - f[6])
        s += 0.8 * f[8] + 1.0 * f[9]
        p = 1.0 / (1.0 + math.exp(-(s - 2.0 + rng.normal(0, 0.5))))
        X.append(f)
        y.append(1 if rng.random() < p else 0)
    return np.array(X), np.array(y)


def load_real(path: str):
    X, y = [], []
    if not os.path.exists(path):
        return np.empty((0, len(FEATURES))), np.empty((0,), dtype=int)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                X.append(featurize(
                    r["event_type"], float(r["budget"] or 0), int(float(r["guests"] or 0)), [],
                    r["cand_event_type"], float(r["price"] or 0), int(float(r["chair_capacity"] or 0)),
                    [], int(float(r["times_booked"] or 0))))
                y.append(int(float(r["booked"])))
            except (KeyError, ValueError):
                continue
    return np.array(X).reshape(-1, len(FEATURES)), np.array(y, dtype=int)


def train():
    Xs, ys = synthetic_data(N_SYNTHETIC)
    Xr, yr = load_real(REAL_CSV)
    X = np.vstack([Xs, Xr]) if len(Xr) else Xs
    y = np.concatenate([ys, yr]) if len(yr) else ys
    w = np.concatenate([np.ones(len(ys)), np.full(len(yr), REAL_WEIGHT)]) if len(yr) else np.ones(len(ys))
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1)
    clf.fit(X, y, sample_weight=w)
    return clf, {"synthetic_rows": int(len(ys)), "real_rows": int(len(yr)), "features": FEATURES}


model, MODEL_INFO = train()


# ----------------------------------------------------------------------- API
class Candidate(BaseModel):
    id: int
    price: float = 0.0
    event_type: str = "Other"
    chair_capacity: int = 0
    equipment_ids: List[int] = Field(default_factory=list)
    times_booked: int = 0


class RecommendRequest(BaseModel):
    event_type: str = "Other"
    budget: float = 0.0
    guest_count: int = 0
    cart_equipment_ids: List[int] = Field(default_factory=list)
    candidates: List[Candidate] = Field(default_factory=list)


@app.get("/")
@app.get("/health")
def home():
    return {"status": "online", "service": "e-Rent ML API", "version": "2.1", "model": MODEL_INFO,
            "endpoints": ["/recommend", "/forecast"]}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.candidates:
        return {"recommendations": [], "note": "no candidates sent"}
    rows = [featurize(req.event_type, req.budget, req.guest_count, req.cart_equipment_ids,
                      c.event_type, c.price, c.chair_capacity, c.equipment_ids, c.times_booked)
            for c in req.candidates]
    pos = list(model.classes_).index(1) if 1 in model.classes_ else None
    probs = model.predict_proba(np.array(rows))[:, pos] if pos is not None else np.zeros(len(rows))
    order = np.argsort(-probs)
    recs = [{"package_id": req.candidates[i].id, "score": round(float(probs[i]), 3)} for i in order]
    return {"recommendations": recs[:3]}


# ------------------------------------------------------- admin analytics: /forecast
import re

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
MAX_SERIES = 400
MAX_MONTHS = 120


class SeriesIn(BaseModel):
    key: str                      # hal. "bookings", "revenue", "eq:12"
    months: List[str]             # "YYYY-MM", magkakasunod, paakyat
    values: List[float]
    integer: bool = False         # True para sa bilang (bookings, units)


class ForecastRequest(BaseModel):
    horizon: int = 3
    series: List[SeriesIn] = Field(default_factory=list)


@app.post("/forecast")
def forecast(req: ForecastRequest):
    """Para sa bawat series: pumipili ng pinakamahusay na model sa backtest at nagbabalik ng
    forecast, low/high range, confidence at backtest error."""
    if not 1 <= req.horizon <= 12:
        raise HTTPException(status_code=422, detail="horizon must be 1-12")
    if len(req.series) > MAX_SERIES:
        raise HTTPException(status_code=422, detail=f"too many series (max {MAX_SERIES})")
    results = {}
    for s in req.series:
        if len(s.months) != len(s.values):
            raise HTTPException(status_code=422, detail=f"{s.key}: months and values differ in length")
        if len(s.months) > MAX_MONTHS or not all(MONTH_RE.match(m) for m in s.months):
            raise HTTPException(status_code=422, detail=f"{s.key}: invalid months")
        vals = [max(0.0, float(v)) for v in s.values]
        results[s.key] = forecast_series(s.months, vals, req.horizon, integer=s.integer)
    return {"engine": "ml_api", "version": "2.1", "horizon": req.horizon, "results": results}
