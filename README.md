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
