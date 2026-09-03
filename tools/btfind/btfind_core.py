"""Pure signal/parsing logic for btfind.

Deliberately free of any Bluetooth imports so the maths can be unit tested on a
machine with no radio. All hardware I/O lives in btfind.py.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field

# Google Fast Pair service UUID. OnePlus/OPPO earbuds (Nord Buds series) use it,
# which is what makes them findable at all when they are not connected.
FAST_PAIR_UUID_SHORT = "fe2c"

# RSSI at 1 m for a small BLE earbud. Measured values for cheap TWS buds sit
# between -55 and -65; -59 is the usual starting point.
DEFAULT_TX_POWER_1M = -59.0

# Indoor path loss exponent. Free space is 2.0; furniture, walls and a human
# body in the way push a flat towards 2.5-3.5.
DEFAULT_PATH_LOSS = 2.5


def median(values):
    """Median of a sequence, without pulling in numpy."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median() of empty sequence")
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values, pct):
    """Linear-interpolated percentile, pct in 0..100."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("percentile() of empty sequence")
    if n == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (n - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def estimate_distance_m(rssi, tx_power_1m=DEFAULT_TX_POWER_1M, path_loss=DEFAULT_PATH_LOSS):
    """Log-distance path loss estimate, in metres.

    Indoors this is worth roughly a factor of two or three either way: multipath
    off walls and a body between you and the bud both swamp the model. Use it to
    tell "same table" from "other end of the flat", never as a ruler.
    """
    exponent = (tx_power_1m - rssi) / (10.0 * path_loss)
    return max(0.05, min(100.0, math.pow(10.0, exponent)))


def proximity_band(rssi):
    """Coarse human-readable range bucket for a smoothed RSSI."""
    if rssi >= -45:
        return "ARM'S REACH", "within about a metre - start lifting cushions"
    if rssi >= -58:
        return "VERY CLOSE", "1-3 m: this sofa, this drawer, this shelf"
    if rssi >= -70:
        return "SAME ROOM", "3-8 m: right room, wrong corner"
    if rssi >= -82:
        return "NEARBY", "8-15 m, or a wall in between"
    return "FAINT", "edge of range - keep walking, watch for it to climb"


def render_meter(rssi, width=40, lo=-100.0, hi=-35.0):
    """Terminal bar for an RSSI value, clamped to the lo..hi window."""
    span = hi - lo
    fraction = (rssi - lo) / span
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass
class FastPairAd:
    """Decoded Fast Pair service data."""

    kind: str  # "model_id" | "account_key_filter" | "unknown"
    model_id: str | None
    raw_hex: str

    def describe(self):
        if self.kind == "model_id":
            return f"Fast Pair pairing mode, model ID {self.model_id}"
        if self.kind == "account_key_filter":
            return "Fast Pair account key filter (paired, lid shut or idle)"
        return f"Fast Pair data {self.raw_hex}"


def _short_uuid(uuid):
    """Reduce a 128-bit Bluetooth base UUID to its 16-bit short form."""
    text = str(uuid).lower()
    if len(text) == 36 and text.endswith("-0000-1000-8000-00805f9b34fb"):
        return text[4:8]
    return text


def parse_fast_pair(service_data):
    """Pull the Fast Pair payload out of a {uuid: bytes} advertisement map.

    Three bytes means the device is in pairing mode and is shouting its model
    ID - the same for every unit of that model, so a sibling bud can teach you
    the ID of the lost one. Anything longer is the account key filter, which a
    paired device emits while idle: less identifying, but proof it is alive.
    """
    for uuid, payload in (service_data or {}).items():
        if _short_uuid(uuid) != FAST_PAIR_UUID_SHORT:
            continue
        data = bytes(payload)
        raw_hex = data.hex()
        if len(data) == 3:
            return FastPairAd("model_id", raw_hex, raw_hex)
        if len(data) > 3:
            return FastPairAd("account_key_filter", None, raw_hex)
        return FastPairAd("unknown", None, raw_hex)
    return None


# Names TWS buds tend to advertise. Matched case-insensitively as substrings.
_EARBUD_NAME_HINTS = (
    "buds",
    "earbud",
    "nord",
    "oneplus",
    "oppo",
    "tws",
    "headset",
    "airpods",
)

# GAP appearance is a 10-bit category plus a 6-bit subcategory. Matching on the
# category catches every earbud/headset/speaker variant without hardcoding a
# list of subcategory values that vendors pick inconsistently anyway.
_AUDIO_APPEARANCE_CATEGORIES = {
    0x21: "wearable audio device",
    0x25: "audio sink",
}


def audio_appearance(appearance):
    """Category name if this GAP appearance is an audio device, else None."""
    if appearance is None:
        return None
    return _AUDIO_APPEARANCE_CATEGORIES.get(appearance >> 6)


def score_candidate(name=None, service_data=None, manufacturer_data=None, appearance=None):
    """Rank how much an advertiser looks like a lost pair of earbuds.

    Returns (score, reasons). Higher is more interesting. This only ever ranks
    what is already broadcasting in the clear; it identifies a device model, not
    a person.
    """
    score = 0
    reasons = []

    fast_pair = parse_fast_pair(service_data)
    if fast_pair is not None:
        if fast_pair.kind == "model_id":
            score += 5
            reasons.append(f"Fast Pair pairing mode (model ID {fast_pair.model_id})")
        elif fast_pair.kind == "account_key_filter":
            score += 4
            reasons.append("Fast Pair account key filter - paired device, alive")
        else:
            score += 2
            reasons.append("Fast Pair service data present")

    if name:
        lowered = name.lower()
        for hint in _EARBUD_NAME_HINTS:
            if hint in lowered:
                score += 3
                reasons.append(f"name contains {hint!r}")
                break

    category = audio_appearance(appearance)
    if category:
        score += 2
        reasons.append(f"GAP appearance 0x{appearance:04x} is a {category}")

    if manufacturer_data:
        ids = ", ".join(f"0x{cid:04x}" for cid in sorted(manufacturer_data))
        reasons.append(f"manufacturer data from {ids}")

    return score, reasons


@dataclass
class RssiTracker:
    """Smooths a stream of RSSI samples into something you can walk towards.

    Raw BLE RSSI swings 10-15 dB between consecutive advertisements even when
    nothing moves. A median filter kills the spikes, then an EWMA gives a needle
    that settles fast enough to be useful while you walk.
    """

    alpha: float = 0.25
    median_window: int = 5
    trend_window: int = 8
    _raw: deque = field(default_factory=lambda: deque(maxlen=5))
    _smoothed: float | None = None
    _history: deque = field(default_factory=lambda: deque(maxlen=64))
    _timestamps: deque = field(default_factory=lambda: deque(maxlen=64))
    best: float | None = None
    count: int = 0
    last_seen: float | None = None

    def __post_init__(self):
        self._raw = deque(maxlen=self.median_window)

    def add(self, rssi, now=None):
        """Feed one advertisement's RSSI. Returns the new smoothed value."""
        now = time.monotonic() if now is None else now
        self.count += 1
        self.last_seen = now
        self._raw.append(float(rssi))
        filtered = median(self._raw)
        if self._smoothed is None:
            self._smoothed = filtered
        else:
            self._smoothed = self.alpha * filtered + (1 - self.alpha) * self._smoothed
        self._history.append(self._smoothed)
        self._timestamps.append(now)
        if self.best is None or self._smoothed > self.best:
            self.best = self._smoothed
        return self._smoothed

    @property
    def smoothed(self):
        return self._smoothed

    def trend(self, threshold=1.5):
        """WARMER / COLDER / STEADY by comparing the last two half-windows."""
        history = list(self._history)[-self.trend_window:]
        if len(history) < 4:
            return "STEADY", 0.0
        half = len(history) // 2
        delta = (sum(history[half:]) / len(history[half:])) - (sum(history[:half]) / half)
        if delta >= threshold:
            return "WARMER", delta
        if delta <= -threshold:
            return "COLDER", delta
        return "STEADY", delta

    def rate_hz(self):
        """Advertisements per second over the retained history."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def is_stale(self, now=None, timeout=10.0):
        now = time.monotonic() if now is None else now
        if self.last_seen is None:
            return True
        return (now - self.last_seen) > timeout


@dataclass
class SurveySession:
    """Room-by-room signal survey.

    Walking a smooth meter around a whole flat is slow and easy to fool. Standing
    still in each room for a fixed dwell and comparing the distributions is both
    faster and far more robust to multipath.
    """

    spots: dict = field(default_factory=dict)

    def record(self, spot, rssi):
        self.spots.setdefault(spot, []).append(float(rssi))

    def summary(self):
        """Per-spot stats, strongest first. Ranked on p90 then median."""
        rows = []
        for spot, samples in self.spots.items():
            if not samples:
                rows.append({
                    "spot": spot, "samples": 0, "median": None,
                    "p90": None, "max": None,
                })
                continue
            rows.append({
                "spot": spot,
                "samples": len(samples),
                "median": round(median(samples), 1),
                "p90": round(percentile(samples, 90), 1),
                "max": round(max(samples), 1),
            })
        rows.sort(
            key=lambda r: (r["p90"] is not None, r["p90"] or -999, r["median"] or -999),
            reverse=True,
        )
        return rows

    def to_csv(self):
        lines = ["spot,samples,median_rssi,p90_rssi,max_rssi"]
        for row in self.summary():
            lines.append(
                f"{row['spot']},{row['samples']},{row['median']},{row['p90']},{row['max']}"
            )
        return "\n".join(lines)


_L2PING_STAT = re.compile(
    r"(\d+)\s+sent,\s*(\d+)\s+received", re.IGNORECASE
)
_L2PING_TIME = re.compile(r"time\s+([\d.]+)ms", re.IGNORECASE)


def parse_l2ping_output(text):
    """Read `l2ping` output into (sent, received, times_ms).

    l2ping pages a classic-Bluetooth device by address. It answers even when the
    device is non-discoverable, which is exactly the state a paired-but-idle
    earbud sits in - so this reaches buds that no amount of scanning will show.
    """
    sent = received = 0
    match = _L2PING_STAT.search(text or "")
    if match:
        sent, received = int(match.group(1)), int(match.group(2))
    times = [float(t) for t in _L2PING_TIME.findall(text or "")]
    return sent, received, times


_HCITOOL_RSSI = re.compile(r"RSSI return value:\s*(-?\d+)", re.IGNORECASE)


def parse_hcitool_rssi(text):
    """Read `hcitool rssi` output.

    Classic Bluetooth reports RSSI as a delta from the controller's "golden
    receive power range", not an absolute dBm: 0 means "comfortably in range",
    negative means below it. Useful as hot/cold, meaningless as a distance.
    """
    match = _HCITOOL_RSSI.search(text or "")
    return int(match.group(1)) if match else None
