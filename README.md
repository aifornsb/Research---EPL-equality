# TxDOT Toll & Traffic Camera Analytics

Automated collection and analysis of two TxDOT Dallas ITS traffic cameras,
producing continuously updated CSV datasets of toll rates and observed
traffic/vehicles using Claude vision.

## 1. Project purpose

This project periodically visits the public TxDOT Dallas camera page
(<https://its.txdot.gov/its/District/DAL/cameras>), captures a snapshot
image from two specific cameras, analyzes each image with Claude's vision
capabilities, and appends the structured results to two CSV datasets:

- **Toll rates** displayed on the electronic toll sign at IH30 @ Loop 12.
- **Vehicles observed** at IH30 @ Carrier Pkwy, including lane type
  (Express vs. General Purpose), an estimated make/model/year where
  visually supportable, and a rough price-range estimate.

It's designed to run either as a scheduled GitHub Action (one snapshot
cycle every 5 minutes) or as a long-running local/server process that
collects data continuously for 7 days.

## 2. Cameras

| Role    | Exact display name on TxDOT page             | Slug used in file paths        |
|---------|------------------------------------------------|---------------------------------|
| Toll    | `IH30 @ Loop 12 WB TRDMS Sta 1251-37`          | `ih30_loop12_wb_trdms_sta_1251_37` |
| Traffic | `IH30 @ Carrier Pkwy`                          | `ih30_carrier_pkwy`             |

Camera names/slugs are configured in `config/cameras.yaml`, not hardcoded
in the Python source, so they can be adjusted without code changes.

## 3. Output CSV schemas

### `data/toll_rates.csv`

One row per toll-camera snapshot.

| Column                 | Description                                                       |
|-------------------------|---------------------------------------------------------------------|
| `snapshot_date`         | Local date (`America/Chicago`), `YYYY-MM-DD`                       |
| `snapshot_time`         | Local time, `HH:MM:SS`                                              |
| `toll_rate_1/2/3`       | Extracted toll rate text, or `UNKNOWN` if not legible               |
| `image_path`            | Path to the saved snapshot image                                    |
| `extraction_confidence` | Claude's confidence (0.0–1.0) in the extracted values               |
| `raw_extracted_text`    | Literal text Claude read from the sign                              |

### `data/traffic_observations.csv`

One row **per vehicle** detected in a traffic-camera snapshot (zero or
more rows per snapshot).

| Column                  | Description                                                         |
|---------------------------|-----------------------------------------------------------------------|
| `snapshot_date` / `snapshot_time` | Local date/time of the snapshot                                |
| `vehicle_id`             | Unique ID: `<timestamp>_<camera_slug>_<sequence_number>`             |
| `direction_facing`       | e.g. `toward camera`, `away from camera`, `left`, `right`, `unknown` |
| `lane_type`              | `Express`, `General Purpose`, or `Unknown` (see lane rule below)     |
| `lane_description`       | Free-text lane detail from Claude                                    |
| `vehicle_make/model/year_estimate` | Only populated when visually supportable; else `UNKNOWN`   |
| `vehicle_body_type`      | e.g. `sedan`, `pickup truck`, `SUV`, or `UNKNOWN`                     |
| `vehicle_color`          | Dominant visible color, or `UNKNOWN`                                  |
| `price_range_low/high`   | Estimated USD price range, or `UNKNOWN`                               |
| `price_range_currency`   | `USD`                                                                 |
| `price_source`           | `manual_lookup` or `UNKNOWN` (see pricing note below)                 |
| `vehicle_confidence`     | Claude's confidence in the vehicle identification                    |
| `price_confidence`       | Confidence in the price estimate                                     |
| `image_path`             | Path to the saved snapshot image                                     |

**Lane rule:** the two middle lanes separated by concrete barriers are
`Express` lanes; every other visible lane is `General Purpose`.

**Pricing note:** Kelley Blue Book's terms of use prohibit automated
scraping, and there is no publicly accessible KBB/Edmunds API suitable
for this use case. To stay compliant, `src/vehicle_pricing.py` uses a
small, manually maintained lookup table of approximate US market price
ranges (`price_source = "manual_lookup"`) instead of scraping any live
pricing site. If a vehicle isn't in the table, or make/model/year is
`UNKNOWN`, all price fields are written as `UNKNOWN` — prices are never
fabricated. You can extend `_PRICE_TABLE` in that file, or swap in a
licensed pricing API later.

## 4. Local setup

Requires Python 3.11+.

```bash
git clone <your-fork-url> txdot-camera-collector
cd txdot-camera-collector

python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m playwright install --with-deps chromium

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## 5. GitHub setup

1. Fork or push this repository to your own GitHub account.
2. In **Settings → Secrets and variables → Actions**, add a repository
   secret named `ANTHROPIC_API_KEY` with your Claude API key.
3. Make sure Actions are enabled for the repository.
4. (Optional) Adjust the cron schedule or switch to artifact uploads
   instead of committing images — see `.github/workflows/collect.yml`.

## 6. Required secrets / environment variables

| Variable            | Required | Description                                  |
|----------------------|----------|-----------------------------------------------|
| `ANTHROPIC_API_KEY`  | Yes*     | Claude API key used for vision analysis       |
| `ANTHROPIC_MODEL`    | No       | Claude model ID to use (default: `claude-sonnet-4-6`, set in `config/cameras.yaml`) |
| `TXDOT_SITE_URL`     | No       | Overrides `site_url` from `config/cameras.yaml` |
| `TIMEZONE`           | No       | Overrides `timezone` (default `America/Chicago`) |
| `INTERVAL_MINUTES`   | No       | Overrides `interval_minutes` (default `5`)    |
| `DURATION_DAYS`      | No       | Overrides `duration_days` (default `7`)       |

\* Not required when running with `--skip-analysis` (capture-only mode) — see
below. It is only enforced when the pipeline actually calls Claude.

Locally, these go in `.env` (loaded via `python-dotenv`). In GitHub
Actions, only `ANTHROPIC_API_KEY` needs to be a secret; the workflow
passes it in as an environment variable. The model name lives in
configuration (`config/cameras.yaml` / `ANTHROPIC_MODEL`), never hardcoded
in `src/toll_extraction.py` or `src/traffic_extraction.py`, so you can
switch models without touching source code.

## 7. Run once (single-run mode)

```bash
python -m src.main --mode single-run
```

This captures both cameras once, analyzes both images with Claude,
appends rows to `data/toll_rates.csv` and `data/traffic_observations.csv`,
writes images under `data/images/<camera_slug>/<date>/`, logs everything
to `logs/collector.log`, and exits. This is the mode used by the GitHub
Actions workflow.

### Capture-only mode (no API calls, no CSV writes)

```bash
python -m src.main --mode single-run --skip-analysis
```

Use this to verify that the browser automation correctly captures both
cameras — real snapshot images, correctly named and placed under
`data/images/<camera_slug>/<date>/` — **without calling Claude or spending
any API tokens**. It does not require `ANTHROPIC_API_KEY` to be set. Check
`logs/collector.log` for `[capture-only]` lines confirming success/failure
per camera, then open the saved `.jpg` files to confirm each one really
shows the correct camera's live feed (not a blank placeholder or a
screenshot of just the camera's name label). `--skip-analysis` also works
with `--mode continuous` if you want to soak-test capture reliability over
a longer period before spending API tokens.

## 8. Run continuously for 7 days

```bash
python -m src.main --mode continuous --duration-days 7 --interval-minutes 5
```

Runs locally (or on any server/self-hosted runner you control),
capturing both cameras every 5 minutes, and stops automatically after 7
days. A failure in any single cycle is logged and does **not** stop the
run — the loop continues to the next scheduled cycle.

Run it in the background with `nohup`, `tmux`, `screen`, or as a
systemd/launchd service if you need it to survive a terminal closing.

## 9. Using GitHub Actions

`.github/workflows/collect.yml` runs on a `*/5 * * * *` cron schedule
(every 5 minutes) and can also be triggered manually from the Actions
tab (`workflow_dispatch`). Each run:

1. Checks out the repo.
2. Sets up Python 3.11 and installs `requirements.txt`.
3. Installs Playwright's Chromium browser and OS dependencies.
4. Pulls/rebases the latest remote state (in case another run pushed
   in-between the checkout and this run).
5. Runs `python -m src.main --mode single-run`.
6. Commits the updated CSVs, new images, and log file, then pushes with
   an automatic pull-rebase-and-retry loop (up to 5 attempts) if another
   concurrent run already pushed in the meantime.

The job timeout is set to 15 minutes (`timeout-minutes: 15`) to comfortably
cover Playwright browser installation + page load + Claude analysis, well
short of GitHub's hard per-job limits.

**Why not one 7-day job?** GitHub-hosted runners have hard job time
limits (far short of 7 days), so this project intentionally uses many
short scheduled `single-run` jobs instead of one long `continuous` job on
GitHub-hosted infrastructure. If you want true continuous mode, run it on
a self-hosted runner or your own always-on server instead.

If committing raw images back to the repo grows the repository too large
over 7 days, switch the "Commit" step in the workflow for the commented-out
`actions/upload-artifact` step, and remove `data/images` tracking from git.

## 10. Known limitations

- **Camera downtime / site changes:** the TxDOT camera page's structure
  or availability can change; the scraper uses a resilient text-based
  camera lookup, but a site redesign may require updating
  `src/camera_capture.py`.
- **Image quality:** poor lighting, weather, glare, or low resolution can
  make toll rates or vehicles unreadable — these are recorded as
  `UNKNOWN` rather than guessed.
- **Claude misclassification:** vision analysis is not perfect and may
  occasionally misread text or misidentify a vehicle; `extraction_confidence`
  / `vehicle_confidence` fields flag uncertainty, and invalid responses are
  retried once, then logged and skipped rather than fabricated.
- **Vehicle identification uncertainty:** make/model/year are only
  reported when visually supportable; many vehicles will legitimately be
  `UNKNOWN`.
- **Pricing uncertainty:** price ranges come from a small, manually
  maintained lookup table (not a live valuation service), so they are
  coarse estimates, not appraisals — treat `price_source = manual_lookup`
  accordingly, and expect frequent `UNKNOWN` values for less common vehicles.
- **GitHub Actions timing:** scheduled cron jobs on GitHub Actions are
  "best effort" and can be delayed several minutes during high load, so
  the actual 5-minute cadence may drift on the hosted-runner setup (this
  does not affect true `continuous` mode run elsewhere).
- **Terms of service:** this project deliberately avoids scraping
  KBB/Edmunds or any site whose terms prohibit automated access; only the
  public TxDOT camera page is accessed, and pricing uses a local lookup
  table instead.

## Project structure

```text
txdot-camera-collector/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── cameras.yaml          # camera names/slugs, timing, site URL
├── data/
│   ├── toll_rates.csv
│   ├── traffic_observations.csv
│   └── images/                # <camera_slug>/<date>/<timestamp>.jpg
├── logs/
│   └── collector.log
├── src/
│   ├── main.py                # CLI entry point: single-run/continuous, --skip-analysis
│   ├── camera_capture.py      # Playwright browser automation; verified real-media screenshots
│   ├── toll_extraction.py     # Claude vision prompt/parsing for toll rates (model from config)
│   ├── traffic_extraction.py  # Claude vision prompt/parsing for vehicles (model from config)
│   ├── vehicle_pricing.py     # Local price-range lookup table
│   ├── csv_writer.py          # Safe, deduped, append-only CSV writing
│   ├── config.py              # YAML + env var configuration loading, incl. ANTHROPIC_MODEL
│   └── utils.py                # logging, timestamps, slugs, JSON parsing
├── tests/
│   ├── test_csv_writer.py
│   ├── test_toll_extraction.py
│   ├── test_traffic_schema.py
│   └── test_utils.py
└── .github/
    └── workflows/
        └── collect.yml         # scheduled single-run workflow
```

## Test sequence: from a fresh clone to a verified working setup

Run these in order — each step builds confidence before you spend API
tokens or commit to a 7-day run.

**1. Install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install --with-deps chromium
```

**2. Run the automated test suite** (no network, no API key needed)

```bash
pytest tests/ -v
```

All tests should pass. These cover CSV append/dedup, timestamped image
path generation, toll/traffic JSON parsing (valid and invalid/retry
cases), and price lookup — with the Claude API and browser mocked/stubbed
out entirely.

**3. Run capture-only mode** (real browser, no API key, no tokens spent)

```bash
cp .env.example .env   # ANTHROPIC_API_KEY can stay as the placeholder for this step
python -m src.main --mode single-run --skip-analysis
```

Watch `logs/collector.log` for two `[capture-only] OK` lines, one per
camera. If either camera logs `[capture-only] FAILED`, the error message
will say whether the camera name couldn't be found on the page or a
valid media element couldn't be captured — check the TxDOT site is up
and the names in `config/cameras.yaml` still match exactly.

**4. Manually inspect the saved images**

```bash
find data/images -name "*.jpg" -newer .env
```

Open the newest file under `data/images/ih30_loop12_wb_trdms_sta_1251_37/`
and `data/images/ih30_carrier_pkwy/` and visually confirm each one really
shows that camera's live road view — not a blank/broken-image icon and
not just a cropped text label. Repeat step 3 a couple of times a few
minutes apart to confirm timestamps in the filenames advance correctly
and match `logs/collector.log`.

**5. Run one full Claude analysis cycle**

```bash
# edit .env and set a real ANTHROPIC_API_KEY first
python -m src.main --mode single-run
```

Confirm:
- `data/toll_rates.csv` gained exactly one new row, and its
  `snapshot_date`/`snapshot_time` match the timestamp in the toll image's
  filename.
- `data/traffic_observations.csv` gained zero or more new rows (one per
  detected vehicle), each with a matching `snapshot_date`/`snapshot_time`
  and a `vehicle_id` containing the traffic camera's slug.
- `logs/collector.log` shows the raw Claude responses and no unexpected
  errors.
- Run the same command again — re-running immediately should not
  duplicate rows for the same snapshot if you re-process the same image
  path (dedup is keyed on date+time+image_path, or vehicle_id+image_path).

**6. Run continuous mode for 7 days** (only after steps 1–5 pass)

```bash
nohup python -m src.main --mode continuous --duration-days 7 --interval-minutes 5 > continuous.out 2>&1 &
```

Check in periodically with `tail -f logs/collector.log` or
`tail -f continuous.out`. The process exits automatically after 7 days;
a single failed cycle is logged and does not stop the run.

**7. Run GitHub Actions manually**

1. Push the repository to GitHub and add the `ANTHROPIC_API_KEY` secret
   (see section 5/6 above).
2. Go to the **Actions** tab → **TxDOT Camera Collection (single-run)** →
   **Run workflow** → select your branch → **Run workflow**.
3. Watch the run: it should install dependencies, install Playwright,
   run one `single-run` cycle, and commit the new CSV rows/images back to
   the branch. Confirm the commit appears in the repo's history and that
   `data/toll_rates.csv` / `data/traffic_observations.csv` show the new
   row(s).
4. Once confirmed, the `*/5 * * * *` cron schedule will pick up
   automatically.

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
