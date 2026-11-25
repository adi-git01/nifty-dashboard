#!/usr/bin/env python3
"""CLI helper for the Antigravity `expert-council` skill.

Builds the strictly grounded live telemetry dossier for any NSE stock from a
Nifty Dashboard repo, and optionally runs the fast deterministic Quant Council
to produce a Markdown brief.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _find_repo(explicit: str | None) -> str:
    candidates = [explicit] if explicit else []
    candidates += [
        os.getcwd(),
        os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "scratch", "nifty-dashboard-py"),
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "utils", "expert_council.py")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not locate a Nifty Dashboard repo containing utils/expert_council.py. "
        "Pass --repo <path_to_nifty_dashboard>."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build grounded Expert Council dossier for an NSE stock.")
    ap.add_argument("query", help="Company name fragment or NSE ticker (e.g. 'Manorama Industries' or 'MANORAMA')")
    ap.add_argument("--repo", default=None, help="Path to Nifty Dashboard repository")
    ap.add_argument("--out", default=None, help="Optional path to save dossier JSON")
    ap.add_argument("--quant", action="store_true", help="Also run deterministic 2-round Quant Council")
    ap.add_argument("--md-out", default=None, help="Optional path to save Markdown brief (implies --quant)")
    args = ap.parse_args()

    repo = _find_repo(args.repo)
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from utils import expert_council as ec

    dossier = ec.build_dossier(args.query, repo=repo)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2, default=str)

    if args.quant or args.md_out:
        res = ec.run_quant_council(dossier)
        md = ec.to_markdown(res)
        if args.md_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
            with open(args.md_out, "w", encoding="utf-8") as f:
                f.write(md)
        print(md)
    else:
        print(json.dumps(dossier, indent=2, default=str))


if __name__ == "__main__":
    main()
