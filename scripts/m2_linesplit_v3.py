"""mlkit_linesplit_v3 — STRIKE-AWARE SAFETY ROUTER (no strike removal).

Reference-free geometric detector over the persisted BASELINE stroke
files (rtl_a1 — unbanded: v1 banding chops long strikes at band
boundaries, which defeated detection in the first structural check)
(image/stroke structure only; no reference text, no CER, no semantics):

A stroke is a LONG CROSSING STROKE if
    x_extent >= 0.25 * crop_width  AND  straightness >= 0.70
where straightness = dist(first_pt, last_pt) / path_length.
An item is STRIKE/SCRIBBLE-SUSPICIOUS if
    (#long-crossing strokes >= 2)  OR
    (#long-crossing strokes >= 1 with x_extent >= 0.45 * crop_width).
Rationale: strike-throughs and border curves are long, near-straight,
horizontally-crossing paths; ordinary Hebrew writing (including connected
thick-pen words) is short-extent or strongly curved. Band-crossing tests
were deliberately NOT used: genuinely touching multi-line writing shares
components across bands and would false-flag.

Routing (v2 rules preserved verbatim for non-suspicious items):
  clean single band            -> v2 passthrough  (= frozen baseline output)
  genuine multi-line           -> v2 split path   (= frozen v1 output)
  strike/scribble-suspicious   -> policy A: fall back to frozen BASELINE
                                  policy B: REVIEW (unsafe for auto)

Because every route maps to an already-persisted frozen recognition output
(identical stroke input -> identical output; determinism established by
the v2 passthrough reproducing baseline exactly), v3 is evaluated OFFLINE
by construction — no new recognition runs.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
V1S = BENCH / "ink_strokes" / "linesplit_v1"
BASES = BENCH / "ink_strokes" / "rtl_a1"

spec = importlib.util.spec_from_file_location("mls", REPO / "scripts" / "m2_linesplit.py")
mls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mls)
GATE_20 = mls.GATE_20


def long_crossing_strokes(rec: dict) -> list[dict]:
    W = rec["width"]
    found = []
    for s in rec["strokes"]:
        xs = [p[0] for p in s]
        extent = (max(xs) - min(xs)) / max(W, 1)
        if extent < 0.25:
            continue
        plen = sum(math.dist(a, b) for a, b in zip(s, s[1:]))
        straight = math.dist(s[0], s[-1]) / max(plen, 1e-9)
        if straight >= 0.70:
            found.append({"x_extent": round(extent, 3),
                          "straightness": round(straight, 3)})
    return found


def classify(item: str) -> dict:
    rec = json.loads((BASES / f"{item}.json").read_text(encoding="utf-8"))
    lcs = long_crossing_strokes(rec)
    rec_v1 = json.loads((V1S / f"{item}.json").read_text(encoding="utf-8"))
    suspicious = len(lcs) >= 2 or any(c["x_extent"] >= 0.45 for c in lcs)
    passthrough = rec_v1["n_bands"] == 1 and not rec_v1["trimmed_bands"]
    if suspicious:
        route = "SUSPICIOUS"
    elif passthrough:
        route = "PASSTHROUGH"
    else:
        route = "SPLIT"
    return {"item": item, "route": route, "long_crossing": lcs,
            "n_bands": rec_v1["n_bands"]}


def main() -> int:
    routing = [classify(it) for it in GATE_20]
    for r in routing:
        note = f" long-crossing={r['long_crossing']}" if r["long_crossing"] else ""
        print(f"{r['item']:<20} {r['route']:<12} bands={r['n_bands']}{note}")
    out = BENCH / "ink_strokes" / "linesplit_v3_routing.json"
    out.write_text(json.dumps(routing, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
