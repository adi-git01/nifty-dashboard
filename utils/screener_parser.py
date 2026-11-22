"""
Shared HTML table parser for Screener.in annual, quarterly, and shareholding tables.
Extracted from winners_setup_scan.py so fundamentals_annual.py has zero coupling to research scripts.
"""
from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import re

import numpy as np
import pandas as pd

QROWS = {"sales": "sales", "revenue": "sales", "net profit": "np"}
SROWS = {
    "promoters": "promoters",
    "fiis": "fii",
    "diis": "dii",
    "public": "public",
    "no of shareholders": "holders",
}


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def _table(fragment):
    m = re.search(r"<table.*?</table>", fragment or "", re.S)
    if not m:
        return []
    p = _Rows()
    p.feed(m.group(0))
    return p.rows


def _num(s):
    s = (s or "").replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _records(rows, names):
    """rows -> {(field, quarter_end): value} for the row labels in `names`."""
    if len(rows) < 2:
        return {}
    qs = []
    for h in rows[0][1:]:
        try:
            qs.append(pd.Timestamp(datetime.strptime(h.strip(), "%b %Y")) + pd.offsets.MonthEnd(0))
        except ValueError:
            qs.append(None)
    out = {}
    for r in rows[1:]:
        label = re.sub(r"[^a-z% ]", "", r[0].lower()).strip()
        field = names.get(label)
        if not field:
            continue
        for q, v in zip(qs, r[1:]):
            if q is not None and not np.isnan(_num(v)):
                out[(field, q)] = _num(v)
    return out
