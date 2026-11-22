# Project Conventions

## Pull Requests
- Keep every PR description within 256 words. Lead with the user-facing
  impact, then a short bullet list of changes. Trim background/history —
  the diff and commit messages carry the detail.

## Commit Messages
- Keep commit messages concise for the same reason: PR bodies are often
  prefilled from them.

---

## Engineering Guardrails (RCA Rules — Read Before Editing)
Full post-mortem table and code-review grep checklist live in `docs/ENGINEERING_GUARDRAILS.md`.

### 1. Trading Dates & Staleness (`bar_date` vs Wall Clock)
- Never use `datetime.now()` or snapshot filenames as the market session date. Always use `bar_date` / NSE `DATE1`.
- Check `data/nse/holidays.csv` and verify current closes differ from the prior session (`price_panel`, `nse_prices`, `engine.py`, `book_logic._closes`) so holiday repeats or partial fetches never write duplicate rows.

### 2. Corporate Actions & Splits
- Keep one factor convention everywhere: `factor = new_price / old_price` (`< 1.0` for splits/bonuses). Pre-ex-date prices multiply by `factor`; shares/volumes divide by `factor`.
- Demergers and rights adjust **price only**, never volume.
- Record `price_basis_date` on broker syncs so `utils/positions.py` adjusts only splits *after* the sync date (no double split adjustment). Keep all adjustment/repair passes idempotent.

### 3. Symbol Canonicalization & Matching
- Normalize broker/external tickers with `utils/portfolio_import.py:clean_symbol()` at ingestion before keying or merging.
- Never compare bare symbols against `.NS`-suffixed keys (`f"{base}.NS" in eq_symbols()`).
- Never fuzzy-match a symbol that already exists in `utils/extended.py:eq_symbols()` (including `EQ`, `BE`, and `BZ` series).
- Guard `symbol_changes` traversal against cycles/reused tickers, and clamp Dhan `correlationId` (`<= 25` alphanumeric chars).

### 4. Real-Money Orders, Alerts & Error Handling
- Persist the `SENDING` order lock synchronously (`background=False`) **before** any broker API call (`utils/model_book.py`); raise/abort if `repo_store` fails to commit. Never auto-resend on timeout (`UncertainOrder`).
- Sequence `SELL` orders before `BUY` and enforce the SEBI 80% same-day CNC peak-margin check (`utils/model_orders_view.py`).
- Block live order placement during NSE holidays and the `09:00–09:15 IST` pre-open window (`utils/dhan_orders.py`), and convert `requests.RequestException` into `BrokerError`.
- Mark alerts/notifications `"sent"` only **after** Telegram/Email delivery succeeds (`utils/alert_store.py`).
- Never swallow pipeline failures with silent `except Exception: pass` or `continue-on-error` on state commits (`scripts/commit_and_push.sh`, `scripts/health_check.py`).

### 5. Time-Aligned Financial Math & Bounded Thresholds
- Never index quarterly/annual series positionally (`iloc[-4]`, `iloc[-5]`) without verifying calendar spans (`<= 395d` for 4Q TTM, `±45d` for YoY lookup, `>= 220d` between FY rows).
- Compute `N`-session returns with `iloc[-(N + 1)]`.
- Bound classification rules on **both** ends (`_PULLBACK_MIN <= dist_ma50 <= _PULLBACK_MAX`; avoid `abs()` shortcuts that misclassify deep drawdowns).
- Score missing sub-metrics explicitly as neutral (`0`) and scale weighted sums across actually scored items (e.g., US stocks lacking promoter/FII rows in `utils/valuation.py`).

### 6. Secrets, Auth & Path Safety
- Never dump `st.secrets` or environment credentials into `config.json` or repo files (`utils/telegram_notifier.py` reads local keys only and keeps `config.json` `0600`).
- Scrub bot tokens/passwords via `_redact_secrets()` before logging or surfacing errors in `st.toast`.
- Validate all MCP/user ticker and path inputs using allowlist regex + `os.path.realpath` containment (`mcp_server/data_access.py`).
- Require TLS 1.2+ on external order proxies (`deploy/order-proxy/proxy.py`) and preserve the `utils/auth_gate.py` gate in `main.py`.

### 7. Incremental Stores & Schema/Format Changes
- Never overwrite multi-month history from a partial scrape or pruned snapshot window: preserve cached historical rows (`utils/signal_scorecard.py` keeps `scorecard_derived_history.parquet`, `utils/book_logic.py` maintains `close_panel.parquet`, `fundamentals_annual.py` retains unparsed tickers' prior rows).
- Refuse remote commits/saves if a GitHub API read truncated (`> 1 MB`).
- Resolve fundamentals/price files through `utils/fund_paths.py`. When renaming or changing file formats, grep the entire repo including `os.path.getmtime()`, tests, and GitHub workflows.

### 8. UI Architecture & Repository Hygiene
- Dashboard pages are **read-only** on pipeline CSVs/parquets (only `AlertStore` / `alert-state` branch stores are mutated by UI actions).
- Keep `main.py` as a thin shell; place page modules in `views/` with lazy imports and module-level `@st.cache_data` helpers. Never introduce local `st` or `df` assignments inside page functions that shadow outer names before assignment.
- Never duplicate scoring/RS formula weights across UI files — import from the core engine.
- Do not commit throwaway research scripts, multi-MB HTML/CSV dumps, or one-off workflows to `main`; record findings in `analysis/*.md`.

### 9. CI, Python 3.10 & Workflow Concurrency
- Add `from __future__ import annotations` to Python modules (GitHub Actions CI and EOD run **Python 3.10**, while local machines may run newer versions).
- Keep `requirements.txt` in sync with all imports used by `main.py`, `views/`, and `test_*.py`.
- Isolate `GITHUB_ACTIONS` environment checks in unit tests (`monkeypatch.delenv`/`setenv`).
- Remember `concurrency: group: repo-data-write, cancel-in-progress: false` retains at most 1 running + 1 pending workflow; queuing a 3rd cancels the pending run.

### 10. Pre-PR Verification Checklist (Required Before Merging)
1. Run `pytest -q --ignore=test_fpdf.py --ignore=test_volume_alert.py` (must be `0` failures).
2. Run `python mcp_server/server.py --selftest` and `python -m py_compile` on changed `.py` files.
3. Add offline regression tests (using `tmp_path` + `monkeypatch`, zero live network) for every bug fix.
4. Ensure GitHub Actions `CI Tests` finishes green on the PR before merging.
