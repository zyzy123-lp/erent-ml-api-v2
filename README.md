# e-Rent ML API v2 (candidate ranking)

## Ano ang binago
- Hindi na kinakabisa ng model ang package IDs. Ang PHP ang nagpapadala ng listahan ng
  `candidates` (id, presyo, event_type, kapasidad ng upuan, equipment_ids, times_booked)
  at ang model ang nagbibigay ng score sa bawat isa.
- Gamit na ang `guest_count`, ang budget fit, ang capacity ng upuan, ang laman ng cart, at ang popularity.
- Backward compatible: kapag walang `candidates`, `{"recommendations": []}` ang sagot, kaya
  nagfa-fallback lang ang PHP (hindi nag-e-error). Puwede mong i-deploy muna ang API bago ang PHP patch.

## Deploy sa Render
1. Siguraduhing `app.py` at `requirements.txt` ang pangalan (hindi `app.py.py`).
2. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Gamitin ang `https://erent-ml-api.onrender.com/health` para sa keep-alive ping (UptimeRobot o cron-job.org, kada 10 minuto).

## Training data
- **Synthetic seed** (4000 pairs), awtomatiko tuwing mag-start. Sabihin ito sa defense.
- **Totoong data:** i-export ang query sa ibaba sa CSV, i-save bilang `data/real_training.csv`,
  at i-redeploy. Nagre-retrain ang API sa startup. Bawat totoong row ay may timbang na 10x.

```sql
SELECT pr.event_type, COALESCE(pr.guests,0) AS guests, COALESCE(pr.budget,0) AS budget,
       p.event_type AS cand_event_type, p.package_price AS price,
       COALESCE((SELECT SUM(pi.quantity) FROM package_items pi
                 JOIN equipment e ON e.equipment_id = pi.equipment_id
                 WHERE pi.package_id = p.package_id AND e.name LIKE '%chair%'), 0) AS chair_capacity,
       (SELECT COUNT(*) FROM reservation_packages rp WHERE rp.package_id = p.package_id) AS times_booked,
       (pr.reservation_id IS NOT NULL) AS booked
FROM package_recommendations pr
JOIN packages p ON p.package_id = pr.package_id;
```
Tingnan ang `data/real_training.sample.csv` para sa format. Hindi ito ni-loload (iba ang pangalan).

## Test
`GET /health` ay nagpapakita ng `synthetic_rows` at `real_rows`, kaya makikita mo kung gumagana ang real data.
---

# v2.1: Admin Analytics forecasting (`POST /forecast`)

Ginagamit ng **Predictive Analytics** ng admin (booking forecast, revenue forecast, equipment demand).

**Request** (isang tawag para sa lahat ng series):
```json
{"horizon": 3,
 "series": [{"key": "bookings", "months": ["2026-09","2026-10","2026-11"], "values": [6,9,8], "integer": true}]}
```
**Response:** `results[key] = {model, method, confidence, history_months, backtest_mae, backtest_mape_pct, points:[{month,value,low,high}]}`

## Paano gumagana (para sa defense)
Bawat series ay sinusubukan sa 4 na model: `mean3` (baseline), `wma_trend`, `ridge_ar` (Ridge regression sa lag1-3 + trend, kailangan ng 8+ buwan), at `gbr_season` (Gradient Boosting + buwan ng taon, kailangan ng 24+ buwan). Pinipili ang pinakamahusay sa **rolling-origin backtest** (hinuhulaan ang bawat isa sa huling ≤8 buwan gamit lang ang mga naunang buwan). Kapag halos magkapantay, ang mas simple ang panalo. Ang `low`/`high` ay galing sa error ng backtest, at ang `confidence` ay nakabatay sa dami ng history at sa laki ng error. Kapag kulang ang data, baseline lang ang lalabas at mababa ang confidence. Hindi nito pinipilit ang ML kung hindi ito mas mahusay.

Ang `/recommend` (package ranking) ay hindi nagbago.
