"""Phase 2A: deterministic image -> synthetic-stroke conversion (rtl heuristic).

Converts frozen hebrew_bench_v2 handwriting crops into ordered stroke paths
for a later ML Kit Digital Ink experiment. The student's true pen order is
UNKNOWN — reconstructed strokes are an approximation and every output says
so. No randomness anywhere; references.json is never read.

Pipeline per crop:
  grayscale -> Otsu threshold -> drop specks (<6 px) -> skeletonize ->
  skeleton graph (endpoints / junctions / edge paths) -> stroke assembly
  (rtl heuristic: components right-to-left; start at the endpoint nearest
  the top-right; greedy direction-preserving continuation at junctions) ->
  resampled ordered (x, y) points.

Outputs (per item):
  evaluation/hebrew_bench_v2/ink_strokes/rtl/<item>.json   stroke data+stats
  evaluation/hebrew_bench_v2/ink_strokes/diag/<item>.png   4-panel diagnostic
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import skeletonize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
OUT = BENCH / "ink_strokes"

GATE_ITEMS = [
    "hc_e002_q1_r1", "hc_e002_q1_r2", "hc_e002_q1_r3", "hc_e002_q1_r4",
    "hl_e003_q1_r1__l1", "hl_e003_q1_r2__l2", "hl_e003_q1_r3__l1",
    "hl_e003_q1_r4__l1", "hl_e004_q1_r1__l1", "hl_e004_q1_r2__l1",
    "hl_e004_q1_r3__l1", "hl_e004_q1_r3__l2", "hl_e005_q1_r1__l1",
    "hl_e005_q1_r1__l2", "hl_e005_q1_r2__l1", "hl_e005_q1_r2__l2",
    "hl_e006_q1_r1__l1", "hl_e006_q1_r2__l1", "hl_e006_q1_r3__l1",
    "hl_e007_q1_r1__l1",
]

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unreadable image {path}")
    return img


def binarize(gray: np.ndarray) -> np.ndarray:
    thr = threshold_otsu(gray)
    ink = gray < thr
    lab = label(ink, connectivity=2)
    keep = np.zeros_like(ink)
    for i in range(1, lab.max() + 1):
        m = lab == i
        if m.sum() >= 6:
            keep |= m
    return keep


def remove_structural_lines(ink: np.ndarray) -> np.ndarray:
    """Phase 2A.1: remove printed cell/table borders — long straight runs
    only (>=60% of width horizontally, >=70% of height vertically), so short
    handwritten strokes and mid-length strike-outs survive."""
    H, W = ink.shape
    u8 = ink.astype(np.uint8)
    hk = max(20, int(W * 0.60))
    vk = max(20, int(H * 0.70))
    horiz = cv2.morphologyEx(u8, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(u8, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    lines = (horiz | vert).astype(bool)
    if not lines.any():
        return ink
    lines_d = cv2.dilate(lines.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return ink & ~lines_d.astype(bool)


def solidify(ink: np.ndarray) -> np.ndarray:
    """Phase 2A.1: thick-ink fix. Diagnosed cause of the doubled 'contour'
    skeletons: Otsu keeps only the dark RIMS of ballpoint strokes (hollow
    centers -> the skeleton traces both contour walls). Close 1-2 px gaps and
    fill small interior holes so skeletonization yields a true centerline.
    Kernel/hole sizes are fixed a priori and small enough not to merge
    separate Hebrew letters (typical inter-letter gaps are far larger)."""
    from skimage.morphology import remove_small_holes

    u8 = ink.astype(np.uint8)
    closed = cv2.morphologyEx(
        u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    return remove_small_holes(closed, area_threshold=40)


def skeleton_graph(skel: np.ndarray):
    """Return (degree image, edge paths). Edges connect nodes (deg!=2 pixels)
    or close cycles; each edge is an ordered pixel path [(y,x), ...]."""
    ys, xs = np.nonzero(skel)
    pix = set(zip(ys.tolist(), xs.tolist()))
    deg = {}
    for p in pix:
        deg[p] = sum(((p[0] + dy, p[1] + dx) in pix) for dy, dx in NB8)
    nodes = {p for p, d in deg.items() if d != 2}
    visited_dir = set()
    edges = []

    def walk(start, first):
        path = [start, first]
        prev, cur = start, first
        while cur not in nodes and cur != start:
            nxts = [
                (cur[0] + dy, cur[1] + dx) for dy, dx in NB8
                if (cur[0] + dy, cur[1] + dx) in pix
                and (cur[0] + dy, cur[1] + dx) != prev
            ]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            path.append(cur)
        return path

    for n in sorted(nodes):
        for dy, dx in NB8:
            first = (n[0] + dy, n[1] + dx)
            if first not in pix or (n, first) in visited_dir:
                continue
            path = walk(n, first)
            visited_dir.add((n, first))
            if len(path) > 1:
                visited_dir.add((path[-1], path[-2]))
                edges.append(path)
    # pure cycles (no nodes): trace remaining unvisited deg==2 pixels
    on_edges = {p for e in edges for p in e}
    leftover = pix - on_edges
    while leftover:
        start = min(leftover, key=lambda p: (p[1], p[0]))  # deterministic
        path = [start]
        prev, cur = None, start
        while True:
            nxts = [
                (cur[0] + dy, cur[1] + dx) for dy, dx in NB8
                if (cur[0] + dy, cur[1] + dx) in pix
                and (cur[0] + dy, cur[1] + dx) != prev
                and ((cur[0] + dy, cur[1] + dx) in leftover
                     or (cur[0] + dy, cur[1] + dx) == start)
            ]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            if cur == start:
                break
            path.append(cur)
        leftover -= set(path)
        if len(path) > 2:
            edges.append(path + [start])
    return deg, nodes, edges


def assemble_strokes_rtl(nodes, edges, shape):
    """rtl heuristic: group edges into components; order components by
    rightmost ink x (descending); within a component start at the endpoint
    nearest the top-right corner and continue greedily along the least
    direction change; unreachable edges start new strokes (rightmost first)."""
    if not edges:
        return []
    # component grouping over shared endpoints
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, e in enumerate(edges):
        parent.setdefault(i, i)
        for j in range(i):
            if {edges[j][0], edges[j][-1]} & {e[0], e[-1]}:
                union(i, j)
    comps = defaultdict(list)
    for i in range(len(edges)):
        comps[find(i)].append(i)

    def comp_right(ids):
        return max(p[1] for i in ids for p in edges[i])

    strokes = []
    H, W = shape
    for ids in sorted(comps.values(), key=lambda ids: -comp_right(ids)):
        unused = set(ids)
        endpoint_edges = [
            i for i in ids
            if edges[i][0] in nodes or edges[i][-1] in nodes
        ]
        # candidate starts: edge tips that are endpoints (degree-1) preferred
        def start_key(i):
            tip = edges[i][0]
            return (tip[0] + (W - tip[1]))  # distance-ish to top-right

        while unused:
            best = None
            for i in sorted(unused):
                for rev in (False, True):
                    path = edges[i][::-1] if rev else edges[i]
                    k = path[0][0] + (W - path[0][1])
                    if best is None or k < best[0]:
                        best = (k, i, rev)
            _, i, rev = best
            path = list(edges[i][::-1] if rev else edges[i])
            unused.discard(i)
            extended = True
            while extended:
                extended = False
                tail = path[-1]
                head_dir = np.array(path[-1]) - np.array(path[max(-4, -len(path))])
                cands = []
                for j in sorted(unused):
                    for rev2 in (False, True):
                        p2 = edges[j][::-1] if rev2 else edges[j]
                        if p2[0] == tail:
                            d2 = np.array(p2[min(3, len(p2) - 1)]) - np.array(p2[0])
                            ang = float(np.dot(head_dir, d2))
                            cands.append((ang, j, rev2))
                if cands:
                    cands.sort(key=lambda c: -c[0])
                    _, j, rev2 = cands[0]
                    p2 = list(edges[j][::-1] if rev2 else edges[j])
                    path.extend(p2[1:])
                    unused.discard(j)
                    extended = True
            strokes.append(path)
    return strokes


def resample(path, step=2.0):
    pts = [(int(x), int(y)) for y, x in path]
    out = [pts[0]]
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        acc += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        if acc >= step:
            out.append(b)
            acc = 0.0
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


PALETTE = [(46, 134, 222), (231, 76, 60), (39, 174, 96), (155, 89, 182),
           (241, 196, 15), (26, 188, 156), (230, 126, 34), (52, 73, 94)]


def diagnostics(gray, skel, strokes, out_path: Path):
    H, W = gray.shape
    orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    skimg = np.zeros((H, W, 3), np.uint8)
    skimg[skel] = (255, 255, 255)
    paths = np.full((H, W, 3), 255, np.uint8)
    marks = np.full((H, W, 3), 255, np.uint8)
    for k, s in enumerate(strokes):
        color = PALETTE[k % len(PALETTE)]
        pts = np.array([(x, y) for x, y in s], np.int32)
        cv2.polylines(paths, [pts], False, color, 1)
        cv2.polylines(marks, [pts], False, (210, 210, 210), 1)
        cv2.circle(marks, tuple(pts[0]), 3, (0, 180, 0), -1)
        cv2.circle(marks, tuple(pts[-1]), 3, (0, 0, 220), -1)
        cv2.putText(marks, str(k + 1), tuple(pts[0] + np.array([2, -3])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    sep = np.full((4, W, 3), 128, np.uint8)
    panel = np.vstack([orig, sep, skimg, sep, paths, sep, marks])
    cv2.imwrite(str(out_path), panel)


VERSION = "rtl"  # set to "rtl_a1" by --version a1


def convert(item_id: str) -> dict:
    t0 = time.monotonic()
    gray = load_gray(BENCH / "crops" / f"{item_id}.png")
    ink = binarize(gray)
    if VERSION == "rtl_a1":
        ink = remove_structural_lines(ink)
        ink = solidify(ink)
        # re-drop specks the line removal may have orphaned
        lab2 = label(ink, connectivity=2)
        keep = np.zeros_like(ink)
        for i in range(1, lab2.max() + 1):
            m = lab2 == i
            if m.sum() >= 6:
                keep |= m
        ink = keep
    n_components = int(label(ink, connectivity=2).max())
    skel = skeletonize(ink)
    deg, nodes, edges = skeleton_graph(skel)
    n_skel = max(len(deg), 1)
    junctions = [p for p, d in deg.items() if d >= 3]
    endpoints = [p for p, d in deg.items() if d == 1]
    ambiguous = set()
    for p in junctions:
        ambiguous.add(p)
        for dy, dx in NB8:
            q = (p[0] + dy, p[1] + dx)
            if q in deg:
                ambiguous.add(q)
    strokes_px = assemble_strokes_rtl(nodes, edges, gray.shape)
    strokes = [resample(s) for s in strokes_px if len(s) >= 3]
    dt = time.monotonic() - t0
    diagnostics(gray, skel, strokes, OUT / ("diag" if VERSION == "rtl" else "diag_a1") / f"{item_id}.png")
    rec = {
        "item": item_id,
        "heuristic": VERSION,
        "stroke_order_note": (
            "SYNTHETIC approximation — the student's true pen order is unknown"
        ),
        "width": int(gray.shape[1]), "height": int(gray.shape[0]),
        "n_ink_components": n_components,
        "n_strokes": len(strokes),
        "n_junction_pixels": len(junctions),
        "junction_density": round(len(junctions) / n_skel, 4),
        "n_endpoints": len(endpoints),
        "ambiguous_skeleton_fraction": round(len(ambiguous) / n_skel, 4),
        "conversion_latency_s": round(dt, 2),
        "strokes": [[[int(x), int(y)] for x, y in s] for s in strokes],
    }
    (OUT / VERSION / f"{item_id}.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8"
    )
    return rec


def main() -> int:
    global VERSION
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "a1"], default="v1")
    args = ap.parse_args()
    VERSION = "rtl" if args.version == "v1" else "rtl_a1"
    (OUT / VERSION).mkdir(parents=True, exist_ok=True)
    (OUT / ("diag" if VERSION == "rtl" else "diag_a1")).mkdir(parents=True, exist_ok=True)
    rows = []
    for item in GATE_ITEMS:
        try:
            r = convert(item)
            rows.append(r)
            print(f"{item}: comps={r['n_ink_components']} strokes={r['n_strokes']} "
                  f"junc_density={r['junction_density']} "
                  f"ambig={r['ambiguous_skeleton_fraction']} "
                  f"endpoints={r['n_endpoints']} t={r['conversion_latency_s']}s")
        except Exception as e:  # noqa: BLE001
            rows.append({"item": item, "error": f"{type(e).__name__}: {e}"})
            print(f"{item}: CONVERSION FAILED {type(e).__name__}: {e}")
    ok = [r for r in rows if "error" not in r]
    if ok:
        med = lambda k: sorted(r[k] for r in ok)[len(ok) // 2]  # noqa: E731
        print(f"\nconverted {len(ok)}/{len(rows)}")
        print(f"median components={med('n_ink_components')} strokes={med('n_strokes')} "
              f"junction_density={med('junction_density')} "
              f"ambiguous_fraction={med('ambiguous_skeleton_fraction')}")
    (OUT / f"summary_{VERSION}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1, default=str)[:400000],
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
