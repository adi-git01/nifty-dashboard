# Industry "turn" alerts -- tested before building (13 Oct 2026)

Question: does an industry-level turn (breadth and relative strength recovering) predict the industry
beating the market? `industry_turn_study.py`, monthly data Sep 2021 - Sep 2026, 48 sub-industries
(>= 5 members), equal-weight basket minus equal-weight universe. Current universe only (survivorship).

| Signal | n | 6m excess mean / median (pp) | 6m hit |
|---|---|---|---|
| Enters TURNING UP (>= 60% above 50d, RS > 0, < 50% above 200d) | 71 | -2.3 / -3.3 | 31% |
| Still in DOWNTREND (bottom-fishing) | 348 | -1.5 / -3.0 | 39% |
| A. >= 50% back above 200d after a < 30% low | 151 | +0.1 / -1.0 | 44% |
| B. RS turns positive while below 200d | 138 | -0.1 / -3.5 | 38% |
| C. basket index crosses its 10-month average | 202 | -0.5 / -1.2 | 44% |
| TURNING UP followed by UPTREND (not pre-registered) | 35 | +2.6 / +1.3 | 57% |
| All industry-months | 2827 | +0.4 / -1.4 | 45% |

Pre-registered criteria (excess > 0, 6m hit >= 55%, beats bottom-fishing): all failed. An early turn
was the worst time to buy an industry; no turn definition tested had an edge. Not built as a buy alert.

Hotels basket (15 stocks) on 12 Oct 2026: 13% above 50d, 27% above 200d, median CompRS -0.6
(from -3.9 in Aug) -- still DOWNTREND; only relative strength is improving.

## Built anyway as a watch tool (utils/industry_alerts.py, industry_alert_daemon.py)
Status changes and a user-chosen trend-score level (0-100 = 40% share above 50d + 40% above 200d +
20% median RS), Telegram from the EOD workflow. Replayed over 162 daily snapshots (Feb-Oct 2026) with
6 alerts: raw daily readings fired 161 messages; 5-day smoothing + 3-day confirmation 90; adding a
5-point hysteresis on the status lines (RS 1 pp) and a +/-2 buffer on the score level 57 (~1.3 per
alert per month). Hotels replay: DOWNTREND -> BASING -> TURNING UP (Apr), back to DOWNTREND (Jun),
TURNING UP (Jun), BASING (Jul), DOWNTREND (Aug) -- a real round trip, not noise.
