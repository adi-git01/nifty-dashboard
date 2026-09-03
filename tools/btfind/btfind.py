#!/usr/bin/env python3
"""btfind - hunt for your own lost Bluetooth earbuds by signal strength.

Five modes:

  scan     list everything advertising nearby, ranked by how much it looks
           like a pair of earbuds
  learn    identify the lost bud's Fast Pair model ID using the sibling bud
           or the case, by diffing before/after you put it in pairing mode
  hunt     live hot/cold meter locked onto one target - walk the flat
  survey   stand in each room for a fixed dwell, then rank the rooms
  classic  page a known paired address over classic Bluetooth, for buds that
           are alive but not advertising at all

This finds a device you own. It matches a model and an address you supply; it
does not identify or follow people.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btfind_core import (  # noqa: E402
    DEFAULT_PATH_LOSS,
    DEFAULT_TX_POWER_1M,
    RssiTracker,
    SurveySession,
    estimate_distance_m,
    parse_fast_pair,
    parse_hcitool_rssi,
    parse_l2ping_output,
    proximity_band,
    render_meter,
    score_candidate,
)

BLEAK_HINT = (
    "btfind needs the 'bleak' package for BLE modes:\n"
    "    pip install -r tools/btfind/requirements.txt\n"
    "(the 'classic' mode needs no Python packages, only BlueZ on Linux)"
)


def _import_bleak():
    try:
        from bleak import BleakScanner
    except ImportError:
        sys.exit(BLEAK_HINT)
    return BleakScanner


class Target:
    """What we are hunting for.

    Address matching is exact but fragile: Fast Pair devices can advertise a
    rotating resolvable private address, and macOS hides the MAC behind a
    per-host UUID. Model ID matching survives both, which is why 'learn' exists.
    """

    def __init__(self, address=None, model_id=None, name=None):
        self.address = address.lower() if address else None
        self.model_id = model_id.lower().replace("0x", "") if model_id else None
        self.name = name.lower() if name else None

    def matches(self, device, adv):
        if self.address and str(device.address).lower() == self.address:
            return True
        if self.model_id:
            fast_pair = parse_fast_pair(adv.service_data)
            if fast_pair and fast_pair.model_id == self.model_id:
                return True
        if self.name:
            local = (adv.local_name or getattr(device, "name", "") or "").lower()
            if self.name in local:
                return True
        return False

    def describe(self):
        parts = []
        if self.address:
            parts.append(f"address {self.address}")
        if self.model_id:
            parts.append(f"Fast Pair model {self.model_id}")
        if self.name:
            parts.append(f"name containing {self.name!r}")
        return " or ".join(parts) if parts else "anything"

    @property
    def is_empty(self):
        return not (self.address or self.model_id or self.name)


def _target_from_args(args):
    target = Target(
        address=getattr(args, "address", None),
        model_id=getattr(args, "model_id", None),
        name=getattr(args, "name", None),
    )
    if target.is_empty:
        sys.exit(
            "Give me something to hunt: --address, --model-id or --name.\n"
            "No idea which? Run `btfind.py scan` first, or `btfind.py learn` "
            "with the bud you still have."
        )
    return target


def _adv_label(device, adv):
    name = adv.local_name or getattr(device, "name", None) or "(no name)"
    return f"{device.address}  {name}"


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

async def cmd_scan(args):
    BleakScanner = _import_bleak()
    seen = {}

    def callback(device, adv):
        appearance = getattr(adv, "appearance", None)
        score, reasons = score_candidate(
            name=adv.local_name or getattr(device, "name", None),
            service_data=adv.service_data,
            manufacturer_data=adv.manufacturer_data,
            appearance=appearance,
        )
        entry = seen.setdefault(
            device.address,
            {"label": _adv_label(device, adv), "score": score, "reasons": reasons,
             "tracker": RssiTracker()},
        )
        entry["tracker"].add(adv.rssi)
        if score > entry["score"]:
            entry["score"], entry["reasons"] = score, reasons
        if adv.local_name:
            entry["label"] = _adv_label(device, adv)

    print(f"Scanning for {args.seconds}s ... (Ctrl-C to stop early)")
    scanner = BleakScanner(detection_callback=callback, scanning_mode="active")
    async with scanner:
        try:
            await asyncio.sleep(args.seconds)
        except asyncio.CancelledError:
            pass

    if not seen:
        print("\nNothing advertising at all. Either the adapter is not scanning "
              "(check permissions) or the room is genuinely quiet.")
        return

    rows = sorted(
        seen.values(),
        key=lambda e: (e["score"], e["tracker"].smoothed or -999),
        reverse=True,
    )
    print(f"\n{len(rows)} devices seen. Best earbud candidates first:\n")
    for entry in rows:
        if args.only_candidates and entry["score"] < 3:
            continue
        tracker = entry["tracker"]
        rssi = tracker.smoothed
        band, _ = proximity_band(rssi)
        print(f"  score {entry['score']:>2}  {rssi:6.1f} dBm  {band:<12} "
              f"{entry['label']}  ({tracker.count} ads)")
        for reason in entry["reasons"]:
            print(f"            - {reason}")
    print("\nA bud in its case with the lid shut advertises nothing. Open the "
          "lid if you can reach it, or use `classic` mode with its MAC.")


# --------------------------------------------------------------------------
# learn
# --------------------------------------------------------------------------

async def _collect(BleakScanner, seconds, sink):
    scanner = BleakScanner(detection_callback=sink, scanning_mode="active")
    async with scanner:
        await asyncio.sleep(seconds)


async def cmd_learn(args):
    """Diff the airwaves before and after you put a known bud in pairing mode.

    Whatever is new in the second pass is your device. The Fast Pair model ID it
    broadcasts is identical on the twin you lost, so this is how you get a
    stable identifier for a bud you cannot touch.
    """
    BleakScanner = _import_bleak()

    baseline = {}
    print(f"Pass 1/2: baseline. Keep the bud OFF or in its shut case. {args.seconds}s ...")
    await _collect(BleakScanner, args.seconds,
                   lambda d, a: baseline.setdefault(d.address, True))
    print(f"  {len(baseline)} devices in the background.\n")

    input("Now put the bud you DO have into pairing mode (open the case lid, or "
          "hold the case button until it flashes), then press Enter: ")

    after = {}

    def callback(device, adv):
        entry = after.setdefault(
            device.address,
            {"label": _adv_label(device, adv), "fast_pair": None, "tracker": RssiTracker()},
        )
        entry["tracker"].add(adv.rssi)
        fast_pair = parse_fast_pair(adv.service_data)
        if fast_pair:
            entry["fast_pair"] = fast_pair
        if adv.local_name:
            entry["label"] = _adv_label(device, adv)

    print(f"Pass 2/2: listening for {args.seconds}s ...")
    await _collect(BleakScanner, args.seconds, callback)

    fresh = {addr: e for addr, e in after.items() if addr not in baseline}
    if not fresh:
        print("\nNothing new appeared. The bud may not have entered pairing mode - "
              "try holding the case button longer, and keep it right next to this "
              "machine.")
        return

    rows = sorted(fresh.values(), key=lambda e: e["tracker"].smoothed or -999, reverse=True)
    print(f"\n{len(rows)} new advertiser(s), strongest first:\n")
    for entry in rows:
        rssi = entry["tracker"].smoothed
        print(f"  {rssi:6.1f} dBm  {entry['label']}")
        if entry["fast_pair"]:
            print(f"            {entry['fast_pair'].describe()}")
            if entry["fast_pair"].model_id:
                print(f"            -> hunt the twin with: "
                      f"--model-id {entry['fast_pair'].model_id}")


# --------------------------------------------------------------------------
# hunt
# --------------------------------------------------------------------------

def _beep_interval(rssi):
    """Geiger-counter pacing: closer means faster clicks."""
    lo, hi = -95.0, -40.0
    fraction = max(0.0, min(1.0, (rssi - lo) / (hi - lo)))
    return 2.0 - 1.85 * fraction


async def cmd_hunt(args):
    BleakScanner = _import_bleak()
    target = _target_from_args(args)
    tracker = RssiTracker(alpha=args.alpha)
    state = {"last_beep": 0.0, "best_announced": None, "writer": None, "fh": None}

    if args.csv:
        state["fh"] = open(args.csv, "w", newline="")
        state["writer"] = csv.writer(state["fh"])
        state["writer"].writerow(["unix_time", "address", "rssi", "smoothed", "est_metres"])

    def callback(device, adv):
        if not target.matches(device, adv):
            return
        now = time.monotonic()
        smoothed = tracker.add(adv.rssi, now)
        distance = estimate_distance_m(smoothed, args.tx_power, args.path_loss)
        band, advice = proximity_band(smoothed)
        trend, delta = tracker.trend()

        if state["writer"]:
            state["writer"].writerow([
                f"{time.time():.3f}", device.address, adv.rssi,
                f"{smoothed:.1f}", f"{distance:.2f}",
            ])

        arrow = {"WARMER": "^^", "COLDER": "vv", "STEADY": "=="}[trend]
        line = (f"\r{render_meter(smoothed)} {smoothed:6.1f} dBm  ~{distance:4.1f} m  "
                f"{arrow} {trend:<7} {band:<12}")
        sys.stdout.write(line.ljust(shutil.get_terminal_size().columns - 1))
        sys.stdout.flush()

        if tracker.best is not None and (
            state["best_announced"] is None or tracker.best > state["best_announced"] + 3.0
        ):
            state["best_announced"] = tracker.best
            print(f"\n  new peak {tracker.best:6.1f} dBm - {advice}")

        if args.beep and (now - state["last_beep"]) > _beep_interval(smoothed):
            state["last_beep"] = now
            sys.stdout.write("\a")
            sys.stdout.flush()

    print(f"Hunting: {target.describe()}")
    print("Walk slowly. Hold the laptop still for 2-3 s per spot - your own body "
          "blocks 2.4 GHz, so turn around rather than trusting one reading.\n")

    scanner = BleakScanner(detection_callback=callback, scanning_mode="active")
    try:
        async with scanner:
            while True:
                await asyncio.sleep(1.0)
                if tracker.count == 0:
                    sys.stdout.write("\rno advertisements yet ...".ljust(60))
                    sys.stdout.flush()
                elif tracker.is_stale(timeout=args.stale):
                    sys.stdout.write(
                        f"\rlost it - last seen {time.monotonic() - tracker.last_seen:.0f}s "
                        f"ago at {tracker.smoothed:.1f} dBm".ljust(70))
                    sys.stdout.flush()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if state["fh"]:
            state["fh"].close()
        print()
        if tracker.count:
            print(f"\n{tracker.count} advertisements, best {tracker.best:.1f} dBm.")
            if args.csv:
                print(f"Log written to {args.csv}")
        else:
            print("\nNever heard from it. See the 'If nothing shows up' section "
                  "of tools/btfind/README.md.")


# --------------------------------------------------------------------------
# survey
# --------------------------------------------------------------------------

async def cmd_survey(args):
    BleakScanner = _import_bleak()
    target = _target_from_args(args)
    session = SurveySession()

    print(f"Room survey: {target.describe()}")
    print(f"Stand still in a spot, name it, and hold position for {args.dwell}s.")
    print("Empty name or 'done' finishes.\n")

    current = {"spot": None}
    scanner = BleakScanner(
        detection_callback=lambda d, a: (
            session.record(current["spot"], a.rssi)
            if current["spot"] and target.matches(d, a) else None
        ),
        scanning_mode="active",
    )

    async with scanner:
        while True:
            spot = (await asyncio.get_running_loop().run_in_executor(
                None, input, "Spot name (or 'done'): ")).strip()
            if not spot or spot.lower() == "done":
                break
            current["spot"] = spot
            before = len(session.spots.get(spot, []))
            for remaining in range(args.dwell, 0, -1):
                sys.stdout.write(f"\r  holding at {spot!r} ... {remaining}s ")
                sys.stdout.flush()
                await asyncio.sleep(1.0)
            current["spot"] = None
            got = len(session.spots.get(spot, [])) - before
            print(f"\r  {spot!r}: {got} samples" + " " * 20)
            if got == 0:
                print("    (nothing heard here - that is a real result, note it)")

    if not session.spots:
        print("\nNo spots recorded.")
        return

    print("\nStrongest spots first:\n")
    print(f"  {'spot':<24} {'n':>4} {'median':>8} {'p90':>8} {'max':>8}")
    for row in session.summary():
        print(f"  {row['spot']:<24} {row['samples']:>4} "
              f"{str(row['median']):>8} {str(row['p90']):>8} {str(row['max']):>8}")

    if args.csv:
        with open(args.csv, "w") as handle:
            handle.write(session.to_csv() + "\n")
        print(f"\nSaved to {args.csv}")
    print("\nTake the top spot and re-run `hunt` there for the final metre.")


# --------------------------------------------------------------------------
# classic
# --------------------------------------------------------------------------

def cmd_classic(args):
    """Page a paired classic-Bluetooth device by address.

    Earbuds spend most of their life connectable but not discoverable, so they
    never show up in a scan - yet they still answer a page from someone who
    knows their address. That makes this the mode that works when `scan` is
    silent. Linux/BlueZ only, and it needs root.
    """
    if not shutil.which("l2ping"):
        sys.exit("l2ping not found. Install BlueZ (apt install bluez) - "
                 "this mode is Linux only.")
    if os.geteuid() != 0:
        print("warning: l2ping normally needs root; re-run with sudo if this fails.\n")

    print(f"Paging {args.address} every {args.interval}s. Ctrl-C to stop.")
    print("Walk around: a rising reply rate and falling round-trip time both "
          "mean you are getting closer.\n")

    tracker = RssiTracker(alpha=0.3)
    try:
        while True:
            proc = subprocess.run(
                ["l2ping", "-c", str(args.count), "-t", "2", args.address],
                capture_output=True, text=True,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            sent, received, times = parse_l2ping_output(output)

            rssi = None
            if shutil.which("hcitool"):
                rssi_proc = subprocess.run(
                    ["hcitool", "rssi", args.address], capture_output=True, text=True
                )
                rssi = parse_hcitool_rssi((rssi_proc.stdout or "") + (rssi_proc.stderr or ""))

            if received == 0:
                print(f"  {time.strftime('%H:%M:%S')}  no reply "
                      f"({sent} paged) - out of range, off, or flat")
            else:
                mean_ms = sum(times) / len(times) if times else float("nan")
                bits = [f"{received}/{sent} replies", f"rtt {mean_ms:6.1f} ms"]
                if rssi is not None:
                    tracker.add(rssi)
                    trend, _ = tracker.trend(threshold=1.0)
                    bits.append(f"rssi {rssi:+d} (golden-range delta, {trend})")
                print(f"  {time.strftime('%H:%M:%S')}  " + "  ".join(bits))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="btfind",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="list nearby advertisers, earbuds first")
    scan.add_argument("--seconds", type=float, default=20.0)
    scan.add_argument("--only-candidates", action="store_true",
                      help="hide everything that scores below 3")
    scan.set_defaults(func=cmd_scan, is_async=True)

    learn = sub.add_parser("learn", help="find your model ID using the bud you still have")
    learn.add_argument("--seconds", type=float, default=15.0)
    learn.set_defaults(func=cmd_learn, is_async=True)

    def add_target(sp):
        sp.add_argument("--address", help="BLE address (MAC, or per-host UUID on macOS)")
        sp.add_argument("--model-id", help="Fast Pair model ID hex, from `learn`")
        sp.add_argument("--name", help="substring of the advertised name")

    hunt = sub.add_parser("hunt", help="live hot/cold meter for one target")
    add_target(hunt)
    hunt.add_argument("--beep", action="store_true", help="click faster as you close in")
    hunt.add_argument("--alpha", type=float, default=0.25,
                      help="EWMA weight; lower is steadier, higher is twitchier")
    hunt.add_argument("--tx-power", type=float, default=DEFAULT_TX_POWER_1M)
    hunt.add_argument("--path-loss", type=float, default=DEFAULT_PATH_LOSS)
    hunt.add_argument("--stale", type=float, default=10.0)
    hunt.add_argument("--csv", help="log every sample here")
    hunt.set_defaults(func=cmd_hunt, is_async=True)

    survey = sub.add_parser("survey", help="dwell in each room, then rank rooms")
    add_target(survey)
    survey.add_argument("--dwell", type=int, default=20, help="seconds per spot")
    survey.add_argument("--csv", help="write the ranking here")
    survey.set_defaults(func=cmd_survey, is_async=True)

    classic = sub.add_parser(
        "classic", help="page a paired classic-BT address (Linux, needs root)")
    classic.add_argument("--address", required=True, help="the buds' Bluetooth MAC")
    classic.add_argument("--count", type=int, default=3, help="pings per round")
    classic.add_argument("--interval", type=float, default=2.0, help="seconds between rounds")
    classic.set_defaults(func=cmd_classic, is_async=False)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.is_async:
        try:
            asyncio.run(args.func(args))
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
