"""Tests for btfind's pure logic. No Bluetooth adapter required.

Run:  python3 tools/btfind/test_btfind_core.py
  or: pytest tools/btfind/test_btfind_core.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btfind_core import (
    RssiTracker,
    SurveySession,
    audio_appearance,
    estimate_distance_m,
    median,
    parse_fast_pair,
    parse_hcitool_rssi,
    parse_l2ping_output,
    percentile,
    proximity_band,
    render_meter,
    score_candidate,
)

FAST_PAIR_UUID = "0000fe2c-0000-1000-8000-00805f9b34fb"


def test_median_and_percentile():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5
    assert percentile([10, 20, 30], 100) == 30
    assert percentile([10, 20, 30], 0) == 10
    assert percentile([10, 20, 30], 50) == 20
    assert percentile([5], 90) == 5


def test_distance_is_monotonic_and_anchored():
    # By definition the model puts the reference tx power at exactly 1 m.
    assert abs(estimate_distance_m(-59.0) - 1.0) < 0.01
    # Weaker signal must always mean further away.
    distances = [estimate_distance_m(r) for r in (-40, -55, -70, -85)]
    assert distances == sorted(distances)
    # And it stays inside the clamp at absurd inputs.
    assert 0.05 <= estimate_distance_m(10) <= 100.0
    assert 0.05 <= estimate_distance_m(-200) <= 100.0


def test_proximity_bands_are_ordered():
    assert proximity_band(-30)[0] == "ARM'S REACH"
    assert proximity_band(-50)[0] == "VERY CLOSE"
    assert proximity_band(-65)[0] == "SAME ROOM"
    assert proximity_band(-75)[0] == "NEARBY"
    assert proximity_band(-95)[0] == "FAINT"


def test_render_meter_clamps():
    assert render_meter(-100, width=10) == "[" + "-" * 10 + "]"
    assert render_meter(-35, width=10) == "[" + "#" * 10 + "]"
    assert render_meter(-500, width=10).count("#") == 0
    assert render_meter(0, width=10).count("#") == 10


def test_fast_pair_model_id():
    ad = parse_fast_pair({FAST_PAIR_UUID: bytes.fromhex("aabbcc")})
    assert ad.kind == "model_id"
    assert ad.model_id == "aabbcc"


def test_fast_pair_account_key_filter():
    ad = parse_fast_pair({FAST_PAIR_UUID: bytes.fromhex("00112233445566")})
    assert ad.kind == "account_key_filter"
    assert ad.model_id is None


def test_fast_pair_ignores_other_services():
    assert parse_fast_pair({"0000180f-0000-1000-8000-00805f9b34fb": b"\x64"}) is None
    assert parse_fast_pair({}) is None
    assert parse_fast_pair(None) is None


def test_audio_appearance_matches_by_category():
    assert audio_appearance(0x0841) == "wearable audio device"   # earbud
    assert audio_appearance(0x0843) == "wearable audio device"   # headphones
    assert audio_appearance(0x0941) == "audio sink"
    assert audio_appearance(0x00C1) is None                      # keyboard
    assert audio_appearance(None) is None


def test_score_prefers_pairing_mode_earbuds():
    pairing, _ = score_candidate(
        name="OnePlus Nord Buds 2r",
        service_data={FAST_PAIR_UUID: bytes.fromhex("aabbcc")},
        appearance=0x0841,
    )
    idle, _ = score_candidate(
        name="OnePlus Nord Buds 2r",
        service_data={FAST_PAIR_UUID: bytes.fromhex("00112233")},
    )
    unrelated, _ = score_candidate(name="Living Room TV")
    assert pairing > idle > unrelated
    assert unrelated == 0


def test_tracker_smooths_towards_the_signal():
    tracker = RssiTracker(alpha=0.5)
    for _ in range(20):
        tracker.add(-60)
    assert abs(tracker.smoothed - (-60)) < 0.5
    assert tracker.count == 20


def test_tracker_median_filter_rejects_a_lone_spike():
    steady = RssiTracker(alpha=0.5)
    spiky = RssiTracker(alpha=0.5)
    for i in range(12):
        steady.add(-70)
        spiky.add(-20 if i == 6 else -70)   # one absurd outlier
    # The median window should absorb the spike almost entirely.
    assert abs(spiky.smoothed - steady.smoothed) < 1.0


def test_tracker_trend_detects_approach_and_retreat():
    warming = RssiTracker(alpha=0.6)
    for rssi in range(-90, -50, 4):
        warming.add(rssi)
    assert warming.trend()[0] == "WARMER"

    cooling = RssiTracker(alpha=0.6)
    for rssi in range(-50, -90, -4):
        cooling.add(rssi)
    assert cooling.trend()[0] == "COLDER"

    flat = RssiTracker(alpha=0.6)
    for _ in range(12):
        flat.add(-65)
    assert flat.trend()[0] == "STEADY"


def test_tracker_tracks_best_and_staleness():
    tracker = RssiTracker()
    tracker.add(-80, now=100.0)
    tracker.add(-50, now=101.0)
    tracker.add(-85, now=102.0)
    assert tracker.best > -80
    assert not tracker.is_stale(now=105.0, timeout=10.0)
    assert tracker.is_stale(now=120.0, timeout=10.0)
    assert RssiTracker().is_stale(now=1.0)


def test_tracker_rate():
    tracker = RssiTracker()
    for i in range(11):
        tracker.add(-60, now=float(i))
    assert abs(tracker.rate_hz() - 1.0) < 0.01
    assert RssiTracker().rate_hz() == 0.0


def test_survey_ranks_the_strongest_room_first():
    session = SurveySession()
    for rssi in (-88, -90, -86):
        session.record("kitchen", rssi)
    for rssi in (-62, -58, -65):
        session.record("bedroom", rssi)
    for rssi in (-75, -78, -74):
        session.record("hallway", rssi)

    rows = session.summary()
    assert [r["spot"] for r in rows] == ["bedroom", "hallway", "kitchen"]
    assert rows[0]["samples"] == 3
    assert rows[0]["max"] == -58

    csv_text = session.to_csv()
    assert csv_text.splitlines()[0] == "spot,samples,median_rssi,p90_rssi,max_rssi"
    assert "bedroom" in csv_text.splitlines()[1]


def test_survey_handles_a_silent_spot():
    session = SurveySession()
    session.record("bedroom", -60)
    session.spots["attic"] = []
    rows = session.summary()
    assert rows[0]["spot"] == "bedroom"
    assert rows[-1]["spot"] == "attic"
    assert rows[-1]["median"] is None


def test_parse_l2ping_output():
    text = (
        "Ping: AA:BB:CC:DD:EE:FF from 00:11:22:33:44:55 (data size 44) ...\n"
        "44 bytes from AA:BB:CC:DD:EE:FF id 0 time 31.12ms\n"
        "44 bytes from AA:BB:CC:DD:EE:FF id 1 time 27.05ms\n"
        "2 sent, 2 received, 0% loss\n"
    )
    sent, received, times = parse_l2ping_output(text)
    assert (sent, received) == (2, 2)
    assert times == [31.12, 27.05]


def test_parse_l2ping_output_when_host_is_down():
    sent, received, times = parse_l2ping_output(
        "Can't connect: Host is down\n3 sent, 0 received, 100% loss\n")
    assert (sent, received, times) == (3, 0, [])
    assert parse_l2ping_output("") == (0, 0, [])
    assert parse_l2ping_output(None) == (0, 0, [])


def test_parse_hcitool_rssi():
    assert parse_hcitool_rssi("RSSI return value: -7") == -7
    assert parse_hcitool_rssi("RSSI return value: 0") == 0
    assert parse_hcitool_rssi("Not connected.") is None
    assert parse_hcitool_rssi("") is None


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc or 'assertion failed'}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run())
