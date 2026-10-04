# LCA Multi-Target ML Service

A FastAPI microservice that predicts Life Cycle Assessment metrics per kg of material from
`material`, `country` and `supply_chain_step`:

| Target               | Unit    |
| -------------------- | ------- |
| `co2_emission`       | kg CO2e |
| `energy_consumption` | MJ      |
| `water_usage`        | litres  |

It **learns continuously**: new verified records arrive over REST or from an upstream API, a
background scheduler retrains when enough have accumulated, and the improved model is
hot-reloaded in memory without downtime. Candidates that fail validation never go live.

## Quick start

```bash
pip install -r requirements.txt
uvicorn src.main:app --port 8000
```

On first start the service creates the DB (`data/lca_store.db`), seeds it from
`data/life_cycle.tsv` (compound rows pivoted into the three target columns, gaps filled with
group medians) and trains model `v1`. Docker: `docker compose up --build` (mounts `./data` and
`./artifacts`).

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"material":"Cotton","country":"India","supply_chain_step":"Dyeing","amount_kg":2.5}'
```

Interactive docs: <http://localhost:8000/docs>.

## Configuration

Environment variables (or a `.env` file):

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./data/lca_store.db` | Any SQLAlchemy URL, e.g. `postgresql+psycopg://user:pw@host/db` (install the driver yourself) |
| `EXTERNAL_DATA_API_URL` | _unset_ | Upstream endpoint polled for verified records |
| `EXTERNAL_DATA_API_KEY` | _unset_ | Sent as `Authorization: Bearer <key>` |
| `RETRAIN_THRESHOLD_NEW_RECORDS` | `50` | New records (since the last training run) that trigger retraining |
| `RETRAIN_CRON_SCHEDULE` | `0 2 * * *` | Cron for the scheduled retrain (empty disables) |
| `SYNC_POLL_INTERVAL_MINUTES` | `15` | How often to poll upstream and check the threshold (`0` disables) |
| `ARTIFACTS_DIR` | `artifacts` | Model files, `current_model.joblib`, `metadata.json` |
| `MIN_TRAIN_SAMPLES` / `MIN_AVG_R2` / `MAX_R2_REGRESSION` | `50` / `0.0` / `0.05` | Training guards (below) |
| `ADMIN_API_KEY` | _unset_ | If set, `POST /records`, `/retrain`, `/sync/external`, `/model/rollback` require an `X-API-Key` header |
| `CORS_ALLOW_ORIGINS` | _empty_ | Comma-separated browser origins allowed to call the API (only needed when a web frontend calls it directly), e.g. `http://localhost:8080` |
| `ENABLE_SCHEDULER` | `true` | Disable background jobs (e.g. in tests) |

## Pointing at an external database / API

Set `EXTERNAL_DATA_API_URL` (and `EXTERNAL_DATA_API_KEY`). The service issues
`GET <url>` with a bearer token and accepts either a JSON list or
`{"records": [...], "next": "<next page url>"}`:

```json
[
  {"id": 8841, "material": "Hemp", "country": "China", "supply_chain_step": "Spinning",
   "co2_emission": 1.9, "energy_consumption": 14.2, "water_usage": 35.0, "verified": true}
]
```

* `id` (optional) is the upstream identifier used for de-duplication; without it, records
  identical to an existing row are skipped.
* Records with `"verified": false` are ignored; invalid records (missing/negative/non-numeric
  values) are counted and skipped, not inserted.
* To bridge an upstream **database**, expose a small read-only view/endpoint with this shape (or
  point `DATABASE_URL` at the shared database and write rows into `lca_records` directly).

Trigger a sync immediately with `POST /sync/external`; otherwise it runs every
`SYNC_POLL_INTERVAL_MINUTES`.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | `material`, `country`, `supply_chain_step`, `amount_kg` (default 1) → predictions scaled by `amount_kg`, clipped at 0. Unknown categories never raise: they are listed in `unseen_categories` and the model uses the remaining features. Matching ignores case/whitespace. |
| `POST /records` | One record, a list, or `{"records": [...]}` of verified ground truth. Queues a background retrain once the new-record threshold is reached. |
| `POST /sync/external` | Poll the upstream API now (background task). |
| `POST /retrain` | Force a retrain and hot-reload. `422` if the candidate is rejected (the old model stays), `409` if one is already running. |
| `POST /model/rollback` | Re-activate the previous stable version. |
| `GET /model/status` | Active version, test metrics, training record count, pending new records. |
| `GET /health` | Service and model availability. |

## How continuous learning works

* **Triggers** – (1) `POST /records` and the periodic poll retrain once
  `RETRAIN_THRESHOLD_NEW_RECORDS` new records exist since the last run; (2) the cron job retrains
  on any new data (and trains if no model exists).
* **Training** – all clean records → mean per material/country/step (no train/test leakage) →
  80/20 split → `OneHotEncoder(handle_unknown="ignore")` + `MultiOutputRegressor(LGBMRegressor)`;
  R², MAE and RMSE per target and averaged.
* **Guards** – at least `MIN_TRAIN_SAMPLES` records; average test R² ≥ `MIN_AVG_R2`; and the
  candidate may not trail the currently active model (scored on the same held-out split) by more
  than `MAX_R2_REGRESSION`. Failing candidates are discarded; nothing changes.
* **Versioning** – `artifacts/models/model_v<N>.joblib`, run history in `model_runs`,
  `artifacts/current_model.joblib` (atomically replaced symlink) and `artifacts/metadata.json`
  (active version, history, vocabulary).
* **Hot reload** – the new model is loaded and smoke-tested off to the side, then swapped in with
  a single lock-protected reference assignment; in-flight predictions are never interrupted.
  `ModelManager.rollback()` / `POST /model/rollback` restores the previous version.

## Verify continuous retraining

```bash
curl -s localhost:8000/model/status            # active_version: v1

# add 50 verified records (the default threshold) -> background retrain
python - <<'PY' | curl -s localhost:8000/records -H 'content-type: application/json' -d @-
import json
print(json.dumps([{"material": "Hemp", "country": "China", "supply_chain_step": "Spinning",
  "co2_emission": 1.0 + i / 100, "energy_consumption": 10.0, "water_usage": 30.0}
  for i in range(50)]))
PY

curl -s localhost:8000/model/status            # active_version: v2, new_records_since_last_train: 0
curl -s -X POST localhost:8000/retrain         # or force one manually
```

(Add `-H "X-API-Key: ..."` to the mutating calls if `ADMIN_API_KEY` is set.)

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

Layout: `src/` (config, database, preprocess, data_sync, train, model_manager, scheduler, predict,
main), `tests/`, `data/life_cycle.tsv` (baseline), `artifacts/` (runtime models, git-ignored).
