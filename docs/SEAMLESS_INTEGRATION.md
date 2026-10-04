# Integration brief: seamless-saas (PLM + PIM) ⇄ LCA service

**Audience:** an engineer / coding agent working in the `seamless-saas` repo on the
`adding-lca-model` branch. **Goal:** make the existing LCA feature in **PLM** and **PIM** use
the LCA prediction service (this repo, `lca_model`) for inputs → outputs, and feed verified LCA
data from the seamless database back so the model retrains continuously.

You do **not** have access to the `lca_model` repo from that session, so everything you need
about the service contract is below. Do not modify the service; if you believe it needs a change,
write it down in the PR description instead.

---

## 0. How to work (read first)

1. **Discover before coding.** On `adding-lca-model`, find: the backend language/framework, DB +
   migration tool, how PLM and PIM are structured (separate services or modules?), where the
   current LCA feature lives (models, API routes, UI), how config/secrets and background jobs are
   done, and the test setup. Follow existing conventions for all of it. Summarise what you found at
   the top of your PR.
2. Put the LCA service behind **one internal client module** (`LcaClient`) shared by PLM and PIM
   (or one per service if they cannot share code). Nothing else talks to the service.
3. **Call the service from the backend only.** Never from browser code, and never expose
   `LCA_SERVICE_API_KEY` to a client.
4. Small commits, tests with each, and finish with the acceptance checklist in §9.

---

## 1. The service at a glance

| | |
| --- | --- |
| Dev base URL | `http://localhost:8000` (env `LCA_SERVICE_URL`) |
| Auth | `/predict`, `/health`, `/model/status`: none. `/records`, `/retrain`, `/sync/external`, `/model/rollback`: header `X-API-Key: <LCA_SERVICE_API_KEY>` (only if the service has `ADMIN_API_KEY` set; assume yes) |
| Content type | `application/json` |
| Docs | `GET /docs` (OpenAPI) |

What it predicts, **per 1 kg** of material, for one *material / country / supply-chain step*:

| Output field | Unit |
| --- | --- |
| `co2_emission` | kg CO2e |
| `energy_consumption` | MJ |
| `water_usage` | litres |

Only these three categorical inputs exist. The model has no knowledge of weight, product type,
process details, etc. — those are handled by multiplying by `amount_kg` and by summing steps.

### Known vocabulary (exact spellings the model was trained on)

Matching is case- and whitespace-insensitive, but anything else (synonyms, plurals, typos) is
treated as **unseen** and the prediction degrades (see §2).

- **Materials (13):** Abaca, Cashmere, Cotton, Hemp, Linen, Nylon, Polyamide, Polyester, Rayon,
  Silk, Spandex, Viscose, Wool
- **Supply chain steps (7):** Raw Material, Spinning, Weaving, Dyeing, Manufacturing, Sampling,
  Trims
- **Countries (53):** Argentina, Australia, Austria, Bangladesh, Belgium, Brazil, Bulgaria,
  Cambodia, Canada, China, Colombia, Czech Republic, Denmark, Egypt, El Salvador, Ethiopia, Europe,
  France, Germany, Greece, Guatemala, Honduras, Hungary, India, Indonesia, Italy, Japan, Kenya,
  Lithuania, Mexico, Morocco, Myanmar, Nepal, Netherlands, New Zealand, Pakistan, Peru,
  Philippines, Poland, Portugal, Romania, South Africa, South Korea, Spain, Sri Lanka, Sweden,
  Taiwan, Thailand, Tunisia, Turkey, UK, United States, Vietnam

The vocabulary **grows** when new verified records with new values are ingested, so do not
hard-code a closed list as validation. Build a **mapping layer** (§4) from seamless's own
material/country/step values to these names, and treat "no mapping" as a first-class case.

---

## 2. API contract — inference (inputs → outputs)

### `POST /predict`

Request:

```json
{
  "material": "Cotton",
  "country": "India",
  "supply_chain_step": "Dyeing",
  "amount_kg": 2.5
}
```

- `material`, `country`, `supply_chain_step`: non-empty strings, ≤100 chars.
- `amount_kg`: number ≥ 0, optional, default `1.0`. Output scales linearly with it.

Response `200`:

```json
{
  "model_version": "v2",
  "amount_kg": 2.5,
  "input": {"material": "Cotton", "country": "India", "supply_chain_step": "Dyeing"},
  "unseen_categories": [],
  "predictions": {"co2_emission": 7.74, "energy_consumption": 39.89, "water_usage": 0.0},
  "predictions_per_kg": {"co2_emission": 3.10, "energy_consumption": 15.96, "water_usage": 0.0}
}
```

- `input` is the canonicalised input actually used (e.g. `" cotton "` → `"Cotton"`).
- `predictions` = per-kg value × `amount_kg`, clipped at 0.
- `unseen_categories` lists which of `material | country | supply_chain_step` were **not in the
  training vocabulary**. The call still succeeds, but that value contributed nothing, so treat the
  result as **low confidence**: store the flag and surface it in the UI.
- `model_version` must be persisted next to every stored result (traceability, re-computation
  after retraining).

Errors: `422` invalid body (field-level detail) · `503` no model loaded (retry later / degrade) ·
`401` never for this endpoint.

### Other read endpoints

- `GET /health` → `{"status":"ok|degraded","model_available":true,"model_version":"v2"}`.
  Use for readiness checks and an admin status badge.
- `GET /model/status` → active `active_version`, `metrics` (`avg_r2`, per-target `r2/mae/rmse`),
  `training_record_count`, `total_records`, `new_records_since_last_train`, `retrain_threshold`,
  `retraining`. Show in an admin/ops view, not on product pages.

### Accuracy caveats the UI must respect

- Results are **estimates**, not measured LCA. Label them as such, with unit and model version.
- Quality differs per target. Energy and water fit well overall; CO2 is the weakest (test R² ≈
  0.37 on the current model). Avoid false precision: show 2–3 significant figures.
- A `0.0` prediction means the model's raw estimate was ≤ 0 and was clipped. Show it as
  "≈ 0 / low confidence", not as a measured zero. (A planned service change should reduce this.)

---

## 3. Computing a product-level LCA in PLM / PIM

The service gives *per material, per step, per kg*. A product result is composed in seamless:

```
for each BOM / composition line (material m, quantity q_kg):
    for each supply-chain step s the product goes through (with country c_s for that step):
        r = predict(material=m, country=c_s, step=s, amount_kg=q_kg)
product.total[target]     = Σ r.predictions[target]
product.by_step[s][target]  = Σ over materials
product.by_material[m][target] = Σ over steps
```

Rules to implement:

1. **Quantity** must be in **kg** per unit of product. Convert grams/percent-of-weight before
   calling. A line with unknown/zero quantity is skipped and reported, not guessed.
2. **Step country**: use the country recorded for that step (supplier/factory country). If a step
   has no country, do not invent one — mark the line "incomplete" and either skip it or use a
   clearly flagged default configured by the tenant.
3. **Batching**: a product with 5 materials × 5 steps = 25 calls. Run them concurrently with a
   small pool (≤ 8) and a per-request timeout (5 s). The service has no batch endpoint.
4. **Result record** (persist, don't recompute on every page view): product id, product version
   (PLM revision / PIM SKU version), inputs snapshot (lines used), per-line outputs, totals,
   `model_version`, `computed_at`, `has_unseen_categories`, `incomplete_lines[]`.
5. **When to recompute**: BOM/material/country/step changes (debounced, async job), and after a
   model upgrade — add an admin action "recompute with latest model" that finds results whose
   `model_version` ≠ current `/health.model_version`.
6. **Never block saves** of PLM/PIM data on the LCA service. Calculation runs as a background
   job; on failure the result is `pending/failed` with a retry. Use retries with backoff (3 tries,
   only on 5xx / timeouts / `503`), and a circuit breaker so an outage doesn't stall workers.
7. **Idempotent & cacheable**: key by `(material, country, step)` per-kg values and cache them
   (e.g. 24 h, or until `model_version` changes — include it in the cache key). Then scale by
   quantity locally. This turns 25 calls into a handful.

PLM vs PIM: PLM owns the product structure (materials, processes, countries per step) and is the
natural place for the calculation; PIM exposes the stored result (and flags) on product data.
If the two are separate services, compute in one and share the stored result via your existing
integration mechanism rather than calling the LCA service twice.

---

## 4. Mapping layer (important for correctness)

Create a small, explicit, tenant-aware mapping table or config:

- `seamless material → model material` (e.g. "Organic Cotton", "Cotton 100%" → `Cotton`;
  "Recycled Polyester" → `Polyester`). Do the mapping in code/config, not by free-text matching at
  call time.
- `seamless process/stage → model supply_chain_step` (e.g. "Garment assembly" → `Manufacturing`,
  "Fabric dyeing" → `Dyeing`).
- `seamless country (ISO code/name) → model country` using the names listed in §1 (e.g. `GB` →
  `UK`, `US` → `United States`, `CZ` → `Czech Republic`).

Unmapped values → keep the line, call with the raw cleaned value only if the business wants a
degraded estimate; otherwise skip it. Either way, record it and show "not covered by the model".
Add a periodic report/endpoint listing the most frequent unmapped values so the mapping (and
later the training data) can be extended.

---

## 5. Retraining from the seamless database

The service learns from **verified ground truth only**. Never send its own predictions back as
training data (that is a feedback loop that makes the model confirm its own errors).

### 5.1 What counts as a training record

One row = measured/verified **per-kg** values for one material/country/step:

`material`, `country`, `supply_chain_step`, `co2_emission` (kg CO2e/kg), `energy_consumption`
(MJ/kg), `water_usage` (L/kg) — all three present, finite and ≥ 0. Use the same mapped names as §4.
Examples of valid sources: supplier-declared / third-party-verified LCA (EPD), lab or audited
factory data, an LCA specialist's approved values. Record who verified it and when.

### 5.2 Database changes in seamless

Add (using the repo's migration tool) a table, e.g. `lca_verified_records`:

| column | notes |
| --- | --- |
| `id` | stable, unique, never reused — becomes the service's `external_id` for de-duplication |
| `tenant_id` | if multi-tenant (see §8) |
| `material`, `country`, `supply_chain_step` | already mapped to model vocabulary |
| `co2_emission`, `energy_consumption`, `water_usage` | numeric, per kg, NOT NULL, CHECK ≥ 0 |
| `verified` | boolean, default false |
| `verified_by`, `verified_at`, `source_ref` | audit trail |
| `created_at`, `updated_at` | |

Plus a UI/API path for authorised users to create/import and verify records (CSV import is a
good first version). **Records are immutable once verified:** a correction creates a new row (new
`id`) and marks the old one `superseded`/excluded — the service de-duplicates by id and will not
update an already-ingested row.

### 5.3 The endpoint the service polls

Implement in seamless (internal network only):

`GET /internal/lca/verified-records?page_token=…` with header
`Authorization: Bearer <LCA_SYNC_TOKEN>` (a dedicated random secret, rotated independently).

Response — either a bare JSON array, or (preferred, paginated):

```json
{
  "records": [
    {
      "id": "8841",
      "material": "Hemp",
      "country": "China",
      "supply_chain_step": "Spinning",
      "co2_emission": 1.9,
      "energy_consumption": 14.2,
      "water_usage": 35.0,
      "verified": true
    }
  ],
  "next": "https://seamless.internal/internal/lca/verified-records?page_token=abc"
}
```

Rules:

- Return only `verified = true`, non-superseded rows. (The service also drops `verified: false`.)
- `id` must be stable. `next` must be an **absolute URL** (omit/`null` on the last page). Use
  keyset (cursor) pagination ordered by `id`, page size 500–1000.
- The service follows at most **100 pages per poll** and polls every 15 minutes (configurable),
  re-reading the whole list each time — it has no "since" parameter and de-duplicates itself. So
  keep the endpoint cheap (indexed, no joins on hot paths) and keep total verified rows within
  ~100 × page size, or tell us so the service can add a `since` filter.
- Invalid rows are skipped and counted by the service, not fatal, but fix them at the source.
- Respond `401` on a bad token, `5xx` on error (the service logs a warning and retries next poll).

### 5.4 Service configuration (done by whoever runs the service)

```
EXTERNAL_DATA_API_URL=https://<seamless-internal-host>/internal/lca/verified-records
EXTERNAL_DATA_API_KEY=<LCA_SYNC_TOKEN>      # sent as Authorization: Bearer
RETRAIN_THRESHOLD_NEW_RECORDS=50            # retrain once ≥50 new rows since last training
RETRAIN_CRON_SCHEDULE=0 2 * * *             # nightly retrain if any new data
SYNC_POLL_INTERVAL_MINUTES=15
ADMIN_API_KEY=<LCA_SERVICE_API_KEY>
```

Docker note: if the service runs in a container and seamless on the host, use
`http://host.docker.internal:<port>/...` (Linux: add `extra_hosts: ["host.docker.internal:host-gateway"]`).

### 5.5 Choose ONE ingestion path

Use the **pull/poll** path above as the system of record (idempotent). Do **not** also push the
same rows to `POST /records`: that endpoint does not de-duplicate, so double-ingestion would
duplicate training rows. Only if seamless cannot expose an endpoint, use push instead
(`POST /records` with `X-API-Key`, body = one record, a list, or `{"records":[…]}`, fields as in
§5.1 plus optional `external_id`) and make sure each record is sent exactly once (outbox table).

### 5.6 Operating the loop from seamless

- Admin action "Sync now" → `POST /sync/external` (202, background). Admin action "Retrain now" →
  `POST /retrain` (200 with new `version_id` and metrics; `422` = candidate rejected by the
  quality gates and the old model stays active; `409` = already running). `POST /model/rollback`
  restores the previous version.
- After a successful retrain, results computed with an older `model_version` become stale (§3.5).
- The service never deploys a model that fails its gates (average test R² ≥ 0, and not clearly
  worse than the active model on the same held-out split), so a bad batch of data cannot silently
  degrade production. Surface `422` messages to admins.

---

## 6. Configuration in seamless

```
LCA_SERVICE_URL=http://localhost:8000
LCA_SERVICE_API_KEY=...          # service ADMIN_API_KEY; backend only
LCA_SYNC_TOKEN=...               # token the service uses to read seamless
LCA_REQUEST_TIMEOUT_MS=5000
LCA_CACHE_TTL_SECONDS=86400
LCA_ENABLED=true                 # kill switch; false → hide LCA numbers, skip calls
```

Validate on startup (warn, don't crash, if the service is unreachable) and expose
`LCA service: up/degraded/down` + `model_version` in the admin/health view.

---

## 7. Suggested internal API / UI in seamless

- `GET /products/{id}/lca` → stored result (totals, by-step, by-material, model_version,
  computed_at, flags, incomplete lines). `POST /products/{id}/lca/recompute` → enqueue.
- PLM UI: show LCA per BOM line and per step, totals, a "low confidence" indicator for
  `unseen_categories`/incomplete lines, a "pending/failed" state, and the model version.
- PIM UI: read-only summary on the product (CO2e / energy / water per product unit, with units
  and "estimate").
- Admin: verified-records import/verify screen, unmapped-values report, service/model status,
  Sync now / Retrain now / Rollback.

---

## 8. Multi-tenancy, privacy, security

- **One model is shared by everyone** who feeds the service. If tenants must not influence each
  other's predictions, either run one service instance per tenant (separate DB + artifacts) or
  only contribute data you are entitled to share. Decide this explicitly and record it.
- The training data contains no personal data; keep it that way (no supplier names in records).
- Secrets via your secret manager. The sync endpoint is internal-only, token-authenticated, and
  rate-limited. The service itself should not be exposed publicly without an auth proxy.

---

## 9. Tests and acceptance checklist

Contract tests with a mocked service (record/replay of the JSON above), including: `200`,
`422`, `503`, timeout, `unseen_categories` non-empty, retries/circuit-breaker, cache hit, and
result scaling by quantity. Unit tests for the mapping layer and for the composition math (totals
equal the sum of parts; by-step and by-material sum to the same totals). Endpoint tests for
`/internal/lca/verified-records`: auth, only verified rows, stable ids, cursor pagination, last
page has no `next`.

Done when all are true:

- [ ] With the service running, creating/editing a product's materials/steps/countries in PLM
      produces a stored LCA result (CO2e, energy, water; per line, per step, total) with
      `model_version`, visible in PLM and summarised in PIM.
- [ ] Unmapped/unseen/incomplete inputs are shown as low-confidence or "not covered", never as
      silent zeros or exceptions. A service outage never blocks saving PLM/PIM data.
- [ ] Verified records created in seamless appear at the verified-records endpoint, are pulled by
      the service (`POST /sync/external` or the poll), and `GET /model/status` shows
      `total_records` increasing.
- [ ] After ≥ threshold new records (or "Retrain now"), `model_version` increments, new
      predictions use it, and stale results can be recomputed from the admin screen.
- [ ] No API key reaches client code; config documented; kill switch works.
- [ ] A short README section in seamless documents the flow, env vars and how to run the service
      locally (`pip install -r requirements.txt && uvicorn src.main:app --port 8000` in
      `lca_model`, with a `.env` containing `ADMIN_API_KEY` and, for browser-direct calls only,
      `CORS_ALLOW_ORIGINS`).

---

## 10. Quick manual test script

```bash
curl -s localhost:8000/health
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"material":"Cotton","country":"India","supply_chain_step":"Dyeing","amount_kg":2.5}'
curl -s -X POST localhost:8000/sync/external -H "X-API-Key: $LCA_SERVICE_API_KEY"
curl -s localhost:8000/model/status
curl -s -X POST localhost:8000/retrain -H "X-API-Key: $LCA_SERVICE_API_KEY"
```
