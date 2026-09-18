# Filings Alpha — Archive Spec and Pre-Registration

Written 2026-09-13, before any filings data has been stored, so the hypotheses
below are fixed in advance of seeing a single forward return.

## Why this source is different

Every factor tested in this repo so far -- CompRS, IAS, earnings buckets,
extension deciles -- is the same price/volume panel re-sliced. They test null
against each other partly because they are correlated views of one dataset.
Weekly exchange filings are the first genuinely orthogonal input: order values,
promoter transactions, commissioning dates and deal counterparties are not
recoverable from a price panel.

That is a structural reason to expect something here. It is not evidence of
anything. The point of this document is to make sure that in twelve months we
can tell the difference.

## The binding constraint: there is no history

The weekly review exists only as a rendered snapshot. Nothing is stored. No
filings signal can be backtested until an archive exists, and every unstored
week is permanently untestable. Volume is roughly:

    ~120 business-change events/week   -> ~1,500 by Jan 2027, ~6,000 by Sep 2027
    ~20 block/bulk net-buy rows/week
    ~1  materially-sized promoter buy/week (see H2 -- the raw count of 6 is not
        the real yield once welfare trusts and token sizes are removed)

At ~1,500 events a first 21-day read is possible around January. The promoter
hypothesis needs about two years.

## Archive schema

Append-only, one row per event, written the same day the review is built. Never
revise a past row -- if the scoring changes, add a column, because the whole
point is knowing what was visible at the time.

`data/filings/events_YYYY.csv`

    week_start, week_end, built_at        window and build timestamp
    ticker, company, mcap_cr             mcap AS AT BUILD, not today
    event_date                           the FILING date, not the review date
    category, tags                       raw NSE category + derived tags
    value_cr                             rupee value in the filing, NaN if absent
    value_pct_mcap                       value_cr / mcap_cr
    headline, filing_url
    impact_score                         as scored that week

`data/filings/insider_YYYY.csv`

    week_start, ticker, mcap_cr
    person, person_type                  promoter / promoter group / director /
                                         designated person / TRUST  <- keep distinct
    bought_cr, sold_cr, net_cr, pct_mcap
    is_open_market                       False if dealt >15% from market price
    promoter_hold_pct, promoter_delta_4q

`data/filings/deals_YYYY.csv`

    week_start, ticker, mcap_cr
    gross_bought_cr, gross_sold_cr, net_cr
    net_to_gross                         net_cr / gross_bought_cr  <- the key ratio
    n_buyers, buyer_names
    n_real_institutions                  excludes the prop-desk list below

Store mcap as at build. A ratio recomputed later against today's mcap is
contaminated by the very return being measured.

### Prop / market-making desks to exclude from institutional counts

These appear on both sides of the tape continuously; their presence is
liquidity provision, not demand:

    MICROCURVES TRADING, JUNOMONETA FINSOL, QE SECURITIES, HRTI,
    ALPHAGREP SECURITIES, JUMP TRADING FINANCIAL INDIA, NEO APEX SHARE BROKING,
    PROGNOSIS SECURITIES, SOCIETE GENERALE ODI

Maintain as a list, review quarterly, and store the version used each week.

## Pre-registered hypotheses

Horizons fixed now: 21 and 63 trading days from the FILING date (not the review
date), excess over Nifty 500. Minimum 30 events per bucket before reading any
result -- the winner-EPS test died at n=2 for exactly this reason.

**H1 -- Material order wins drift.** Events tagged order win / commissioning /
capacity with `value_pct_mcap >= 2%` show positive 21d and 63d excess return.
Control: same category with `value_pct_mcap < 0.5%` should show nothing. If both
drift, it is the category not the magnitude, and H1 fails.
*Prior: moderate. Closest analogue to documented post-announcement drift, with a
cleanly quantified surprise. Most weekly order wins fail the 2% floor -- that is
intended, it makes the sample small and clean.*

**H2 -- Meaningful promoter buys drift.** `person_type` in (promoter, promoter
group, director), `is_open_market`, `pct_mcap >= 0.05%`, welfare/ESOP trusts
excluded. Positive 63d excess.
*Prior: strongest on the page -- insider open-market buying is among the more
robust global anomalies, concentrated in small caps. But the real yield is ~1
name/week, so this needs ~2 years.*

**H3 -- Net accumulation, not gross turnover.** Among block/bulk rows,
`net_to_gross >= 50%` with `n_real_institutions >= 1` outperforms rows with
`net_to_gross < 5%`. The current screen ranks by gross and inverts this: in the
06-13 Sep week, the only two genuinely one-sided buys (GMMPFAUDLR, Persistence
Capital 100% net; HEG, HDFC MF 100% net) ranked BELOW two names that were
99%+ market-maker round-trips (RAYMOND 0.8% net, AWFIS 3.5%).
*Prior: low for the gross version (it is measuring HFT), plausible for net.*

**H4 -- Dilution events drift NEGATIVE.** QIP / preferential / warrant issues
underperform over 63d. Worth testing because it is the only short/avoid signal
here, and because such names currently rank HIGH on the momentum screen --
RAYMOND was #1 on OptComp with a preferential warrant issue as its catalyst. If
H4 holds, that is an anti-signal wearing a momentum costume.
*Prior: moderate and negative.*

**H5 -- Confluence.** Buying AND a business change in the same week beats either
leg alone. The conjunction of two orthogonal sources is the most interesting
idea in the stack, but at ~2/week it yields ~100/year. Do not read before 2027.

## What NOT to do

**Do not gate catalysts through OptComp or IAS.** analysis/FACTOR_VERDICT.md and
analysis/H2_VERDICT.md establish CompRS IC ~ 0 over ten years and IAS tiers as
regime-inherited with no predictive power. The OptComp gate cuts 120 catalyst
names to 52; on our own evidence which 52 survive is close to a coin flip, and
it discards more than half of an already small event sample. Keep both as
display context. Record `optcomp_pass` and `ias_score` as COLUMNS so they can be
tested as interactions later -- but never as filters that decide what is stored.

**Do not read a result before its minimum n.** Every null in this repo so far
arrived first as an encouraging-looking small sample.

## Scoring fixes worth making before the next weekly run

1. Rank block/bulk by `net_cr`, not gross; surface `net_to_gross`; exclude prop
   desks from the institutional count.
2. Make `value_pct_mcap` a hard gate on business-change ranking rather than a
   minor term in a composite. Currently intra-group subsidiary mergers with no
   economic content (INDGN merging its own German subsidiaries, PHOENIXLTD
   merging subsidiaries into each other) score 73, while a 3.9%-of-mcap order
   win scores 48.
3. Separate welfare/ESOP trusts from promoters and directors in the insider
   table, and apply a `pct_mcap` floor. The week of 06-13 Sep reported 6 insider
   buys; one (HATSUN) survives a materiality filter.
