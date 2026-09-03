# btfind — finding lost earbuds by Bluetooth signal strength

Built for a pair of OnePlus Nord Buds 2r lost inside a house. Works for any
Bluetooth earbuds, headphones or speaker **that you own**.

---

## Read this first: will this work at all?

Bluetooth hunting can only find a radio that is still transmitting. Whether
yours is comes down to battery, and a week is a long time:

| Where the bud has been sitting | Battery after ~1 week | Findable by radio? |
| --- | --- | --- |
| Loose on the floor / under a cushion | Flat. Standby drains a bud in hours, not days | **No.** It transmits nothing |
| In the case, case had charge | Likely still alive — the case tops it up and self-discharges slowly | **Yes** |
| In the case, case was already low | Probably flat | Unlikely |

If you lost **one loose bud** seven days ago, it is almost certainly dead, and
no amount of analysis will change that — there is no signal to receive. Skip to
[If nothing shows up](#if-nothing-shows-up). Passive tags like an AirTag are
findable when dead only because they are built for it; earbuds are not.

## Do these three things before running any code

They beat signal hunting outright, and take two minutes.

1. **Google Find My Device** — Nord Buds 2r support Google Fast Pair, so they
   are registered as a device on your Google account with the location where
   your phone last held a connection. That tells you which *building* or which
   end of the house, even if the buds are now flat.
2. **HeyMelody / OnePlus Buds app → Find My Earbuds** — makes a live bud shriek.
   If they have any charge, this finds them in seconds and everything below is
   unnecessary. Try each bud separately; one may be alive and the other flat.
3. **Your phone's Bluetooth screen** — if the buds still show as *Connected*,
   they are alive and close. That alone narrows it to one room.

Signal hunting is the fallback for when they will not ring: a bud that is alive
but whose speaker you cannot hear (buried in a sofa, inside a bag, in a closed
drawer, one bud silent while the other rings).

## Install

```bash
pip install -r tools/btfind/requirements.txt      # just 'bleak'
python3 tools/btfind/btfind.py --help
```

Use a **laptop**, not the phone that owns the buds — a phone that auto-connects
to them stops advertising them to your scanner. Linux, macOS and Windows all
work for the BLE modes; `classic` mode is Linux-only.

On Linux you may need `sudo setcap 'cap_net_raw,cap_net_admin+eip' $(which python3)`
or simply run under `sudo`.

## Step 1 — get something to hunt for

You need an identifier. Two ways, best first:

**A. The Fast Pair model ID, from the bud you still have.** Model IDs are
per-model, not per-unit — the twin you lost broadcasts the exact same one. This
is the robust option, because BLE addresses can rotate but the model ID cannot.

```bash
python3 tools/btfind/btfind.py learn
```

It records the background radio, asks you to put the bud you *do* have into
pairing mode, then shows what is new. Note the model ID it prints.

**B. The Bluetooth MAC, from your phone.** Android: Settings → Connected devices
→ gear icon next to the buds; if it is not shown there, `adb shell dumpsys
bluetooth_manager | grep -i -A3 buds` will list paired addresses. You need this
for `classic` mode regardless.

Nothing to hand at all? `scan` ranks everything nearby by how much it looks like
earbuds — Fast Pair data, name, and GAP appearance category:

```bash
python3 tools/btfind/btfind.py scan --seconds 30 --only-candidates
```

## Step 2 — narrow to a room

Do **not** wander around watching a live meter first. Standing still in each
room for a fixed dwell and comparing distributions is faster and much harder to
fool, because it averages out the multipath that makes a single reading lie.

```bash
python3 tools/btfind/btfind.py survey --model-id aabbcc --dwell 20 --csv survey.csv
```

Name each spot, hold still for the dwell, repeat. It ranks rooms on the 90th
percentile of signal. A room where it hears *nothing* is a real result — write
it off and move on.

## Step 3 — close the last few metres

```bash
python3 tools/btfind/btfind.py hunt --model-id aabbcc --beep
```

Live meter, smoothed signal, rough distance, a WARMER/COLDER arrow, and beeps
that quicken as you close in. It logs every peak so you can see where it was
strongest.

**Technique matters more than the tool:**

- Move in **2–3 m steps and pause 3 seconds**. The smoother needs samples; a
  reading taken mid-stride is noise.
- **Your own body blocks 2.4 GHz.** A 10 dB drop can just mean you turned round.
  At each spot, rotate slowly and take the *best* reading, not the average.
- **Trust rising, distrust falling.** Signal peaks are informative; nulls are
  usually reflections. Chase the peaks.
- Sweep **low and high separately** — under a sofa reads very differently from
  on top of it.
- Metal and water kill the signal: inside a washing machine, under a mattress,
  in a coat pocket in a wardrobe, or behind a fridge all read far weaker than
  the true distance.
- The distance figure is a **log-distance path loss estimate, indoors** — treat
  it as accurate to a factor of two or three. It separates "this sofa" from
  "other end of the flat"; it is not a tape measure.

## Step 4 — when scanning finds nothing but you think they are alive

Earbuds spend most of their life *connectable but not discoverable*: they answer
if you know their address, but they do not shout. Scanners never see them. You
can still page them directly:

```bash
sudo python3 tools/btfind/btfind.py classic --address AA:BB:CC:DD:EE:FF
```

A reply proves the buds are powered and within range of your adapter. Rising
reply rate and falling round-trip time as you walk both mean you are closing in.

Note that classic Bluetooth reports RSSI as a delta from the controller's
"golden receive power range", not an absolute dBm — 0 means comfortably in
range, negative means below it. Read it as hot/cold only. Linux/BlueZ, root.

## If nothing shows up

Silence across `scan`, `survey` and `classic` means the buds are flat, out of
range, or both. At that point the radio has nothing more to give, and physical
search is what is left. Where they actually turn up, roughly in order:

- Inside sofa and armchair crevices, and *underneath* the cushions, not between
- Bedding — dragged into a duvet fold, or fallen behind the headboard
- In the laundry: pockets of worn clothes, the wash basket, the machine drum
- Coat and bag pockets, and the bag's inner lining through a torn seam
- Under the bed, sofa or wardrobe against the skirting, where a light shows them
- Car seat rails and door pockets, if you were out that day
- Wherever you last charged your phone — buds get put down beside chargers

A torch held at floor level, sweeping sideways, catches the gloss of a bud
casing far better than overhead light.

Then: charge the case for an hour and try steps 1–3 again. A bud that was too
flat to advertise on Monday can come back if it is sitting in a case that still
has charge — and the ringing feature in the OnePlus app becomes an option again.

## Phone alternative

If you would rather not use a laptop, the free **nRF Connect** app (Nordic
Semiconductor, Android and iOS) does the scan-and-RSSI part well, with a live
graph per device. It has no room-survey ranking or classic-BT paging, but for
step 3 alone it is a perfectly good substitute — use whichever is to hand.

## What is in here

| File | Purpose |
| --- | --- |
| `btfind.py` | CLI: `scan`, `learn`, `hunt`, `survey`, `classic` |
| `btfind_core.py` | Signal maths and parsing — no Bluetooth imports |
| `test_btfind_core.py` | Unit tests, no adapter needed |
| `requirements.txt` | `bleak` |

```bash
python3 tools/btfind/test_btfind_core.py     # 19 tests, no hardware
```

This is standalone; the Streamlit dashboard does not import it and its
dependencies stay out of the top-level `requirements.txt`.

## Scope

This tracks a device you own, matched against an address or model ID you supply
yourself. It reads advertisements that Bluetooth devices already broadcast in
the clear, and it is not built for and should not be used for following anyone.
