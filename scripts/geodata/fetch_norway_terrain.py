#!/usr/bin/env python3
"""
fetch_norway_terrain.py - build a threepp "region pack" from live Norwegian
open geodata: national DTM elevation (Kartverket) + road network (NVDB).

Output (frozen C++ contract) written to <out>/<name>/:
    region.json     metadata + coordinate mapping
    heights.f32     dim*dim little-endian float32, row-major (iz*dim + ix)
    roads.json      polylines in local world coords
    preview.(png|tif)   optional hillshade + roads overlay (--preview)

Data sources (verified live 2026-07-15):
  Elevation: Kartverket national DTM via WCS 1.0.0 GetCoverage.
             CRS EPSG:25833 (UTM33). License CC BY 4.0, (c) Kartverket.
  Roads:     NVDB API Les v4 vegnett/segmentert. License NLOD,
             (c) Statens vegvesen.

Usage:
    python fetch_norway_terrain.py --preset trollstigen
    python fetch_norway_terrain.py --center 62.4482,7.6714 --size 8000 --res 2 --name myregion
    python fetch_norway_terrain.py --preset alesund --preview --include-paths
"""
import argparse
import io
import json
import math
import os
import re
import sys
import time

import numpy as np
import requests
import tifffile
from pyproj import Transformer

# --------------------------------------------------------------------------
# Presets  (lat, lon in WGS84).
#   trollstigen: centered on the Fv63 switchback wall (the famous 11 hairpins).
#     The wall climbs ~62.449N..62.456N; this center places it in the middle
#     of the region.  Peaks Bispen/Kongen (>1400 m) fall within +-4 km.
#   aalesund: coastal town center (exercises the sea / nodata=0 path).
#     (Spelled "aalesund" to match the pack dir; "alesund" stays accepted as
#     an alias so older invocations keep working.)
# --------------------------------------------------------------------------
PRESETS = {
    "trollstigen": (62.4525, 7.6675),
    "aalesund": (62.4722, 6.1495),
    "alesund": (62.4722, 6.1495),  # alias for aalesund (deprecated spelling)
}

# WCS elevation endpoint (ArcGIS - only the WCS 1.0.0 bbox form works here;
# the OGC 2.0.1 subset= form 400s on this server).
WCS_URL = "https://wms.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
WCS_COVERAGE = "nhm_dtm_topo_25833"

# NVDB road network endpoint.
NVDB_URL = "https://nvdbapiles.atlas.vegvesen.no/vegnett/api/v4/veglenkesekvenser/segmentert"
NVDB_HEADERS = {"X-Client": "threepp-geodata", "Accept": "application/json"}

# Default carriageway widths (m) by vegkategori.
CATEGORY_WIDTH = {"E": 9.0, "R": 9.0, "F": 7.0, "K": 5.5, "P": 4.0, "S": 3.0}
PATH_WIDTH = 3.0

# NVDB detail levels: keep the single-geometry-per-road levels.
#   'Vegtrase og kjorebane' -> simple roads (one combined geometry)
#   'Vegtrase'              -> centerline of divided roads
# Drop 'Kjorebane' / 'Kjorefelt' which are the parallel-carriageway / lane
# duplicates of the same physical road (would double-draw divided highways).
DETALJ_KEEP = {"Vegtrase og kjørebane", "Vegtrase"}

# typeVeg classification.
VEHICLE_TYPEVEG = {"Enkel bilveg", "Kanalisert veg", "Rampe", "Rundkjøring"}
PATH_TYPEVEG = {"Gangveg", "Gang- og sykkelveg", "Gågate"}
# (Fortau/Trapp/Gangfelt = sidewalks/stairs/crossings: always skipped as noise.)

CHUNK = 2000  # max px per WCS request dimension (server limit ~2048)


# ==========================================================================
# Elevation
# ==========================================================================
def _wcs_tile(minE, minN, maxE, maxN, w, h, session, retries=4):
    """Fetch one node-aligned DTM tile as a float32 (h, w) array."""
    params = {
        "service": "WCS", "version": "1.0.0", "request": "GetCoverage",
        "coverage": WCS_COVERAGE, "crs": "EPSG:25833",
        "bbox": f"{minE},{minN},{maxE},{maxN}",
        "width": int(w), "height": int(h), "format": "GeoTIFF",
    }
    last = None
    for attempt in range(retries):
        try:
            r = session.get(WCS_URL, params=params, timeout=120)
            if r.status_code == 200 and r.content[:4] in (b"II*\x00", b"MM\x00*"):
                arr = tifffile.imread(io.BytesIO(r.content))
                if arr.shape == (h, w):
                    return np.asarray(arr, dtype=np.float32)
                raise ValueError(f"tile shape {arr.shape} != {(h, w)}")
            # HTML error page or bad status
            snippet = r.content[:80]
            last = f"HTTP {r.status_code} {snippet!r}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"WCS tile failed after {retries} tries: {last}")


def fetch_heights(originE, originN, size, res, dim):
    """
    Fetch a dim x dim node-aligned height grid.

    Grid nodes sit exactly at:
        E_ix = originE - size/2 + ix*res,   ix = 0..dim-1  (west->east)
        N_iz = originN + size/2 - iz*res,   iz = 0..dim-1  (north->south)
    WCS uses a pixel-IS-AREA convention (scale = bbox_span / width), so to
    land pixel *centers* on the nodes we request a bbox expanded by res/2 on
    every side and width/height = node count.  Tiles are cut on node index
    boundaries so they stitch with no overlap and no resampling.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "threepp-geodata/1.0"})
    grid = np.zeros((dim, dim), dtype=np.float32)

    west = originE - size / 2.0
    north = originN + size / 2.0

    col_bounds = list(range(0, dim, CHUNK)) + [dim]
    row_bounds = list(range(0, dim, CHUNK)) + [dim]
    ntiles = (len(col_bounds) - 1) * (len(row_bounds) - 1)
    t = 0
    for ri in range(len(row_bounds) - 1):
        r0, r1 = row_bounds[ri], row_bounds[ri + 1]
        rh = r1 - r0
        # row r0 node northing = north - r0*res  (top of this tile band)
        n_top = north - r0 * res
        maxN = n_top + res / 2.0
        minN = n_top - (rh - 1) * res - res / 2.0
        for ci in range(len(col_bounds) - 1):
            c0, c1 = col_bounds[ci], col_bounds[ci + 1]
            cw = c1 - c0
            e_left = west + c0 * res
            minE = e_left - res / 2.0
            maxE = e_left + (cw - 1) * res + res / 2.0
            t += 1
            print(f"  DTM tile {t}/{ntiles}  rows[{r0}:{r1}] cols[{c0}:{c1}] "
                  f"({cw}x{rh})", flush=True)
            tile = _wcs_tile(minE, minN, maxE, maxN, cw, rh, session)
            grid[r0:r1, c0:c1] = tile
            time.sleep(0.15)  # be polite

    # Nodata handling: the DTM already encodes sea as 0.0 (verified: no
    # negative sentinel, min ~ -0.003 float noise).  Guard against NaN and
    # any large-negative sentinel, then clamp the floor to sea level (0).
    grid[~np.isfinite(grid)] = 0.0
    grid[grid < -1e4] = 0.0
    np.clip(grid, 0.0, None, out=grid)
    return grid


# ==========================================================================
# Roads
# ==========================================================================
_WKT_RE = re.compile(r"LINESTRING\s+Z?\s*\(([^)]*)\)", re.IGNORECASE)


def parse_wkt_linestring(wkt):
    """'LINESTRING Z (E N h, ...)' -> list[(E, N, h)]."""
    m = _WKT_RE.search(wkt or "")
    if not m:
        return []
    pts = []
    for chunk in m.group(1).split(","):
        parts = chunk.split()
        if len(parts) >= 3:
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
        elif len(parts) == 2:
            pts.append((float(parts[0]), float(parts[1]), 0.0))
    return pts


def fetch_roads(originE, originN, size, include_paths, verbose=True):
    """
    Fetch, filter and merge the NVDB road network intersecting the region.

    Returns list of dicts: {seqid, category, typeVeg, width, pts=[(E,N,h)...]}
    (still in UTM33 world coords; local mapping happens later).
    """
    session = requests.Session()
    session.headers.update(NVDB_HEADERS)
    half = size / 2.0
    minE, maxE = originE - half, originE + half
    minN, maxN = originN - half, originN + half
    bbox = f"{minE},{minN},{maxE},{maxN}"

    # Collect segments grouped by veglenkesekvensid so we can merge runs.
    segs = {}          # seqid -> list of {start,stop,pts,category,typeVeg}
    start = None
    pages = 0
    kept = dropped = 0
    while True:
        params = {"kartutsnitt": bbox, "antall": 1000}
        if start:
            params["start"] = start
        r = session.get(NVDB_URL, params=params, timeout=120)
        r.raise_for_status()
        d = r.json()
        objs = d.get("objekter", [])
        for o in objs:
            detalj = o.get("detaljnivå")
            typeveg = o.get("typeVeg")
            if detalj not in DETALJ_KEEP:
                dropped += 1
                continue
            is_vehicle = typeveg in VEHICLE_TYPEVEG
            is_path = typeveg in PATH_TYPEVEG
            if not (is_vehicle or (include_paths and is_path)):
                dropped += 1
                continue
            pts = parse_wkt_linestring((o.get("geometri") or {}).get("wkt"))
            if len(pts) < 2:
                dropped += 1
                continue
            vsr = o.get("vegsystemreferanse") or {}
            cat = (vsr.get("vegsystem") or {}).get("vegkategori")
            if cat not in CATEGORY_WIDTH:
                cat = "S" if is_path else "K"  # fallbacks
            width = PATH_WIDTH if is_path else CATEGORY_WIDTH[cat]
            seqid = o["veglenkesekvensid"]
            segs.setdefault(seqid, []).append({
                "start": o.get("startposisjon", 0.0),
                "pts": pts, "category": cat, "typeVeg": typeveg,
                "width": width,
            })
            kept += 1
        pages += 1
        nxt = d.get("metadata", {}).get("neste")
        if not nxt or not objs:
            break
        start = nxt.get("start")
        if not start:
            break
        time.sleep(0.1)
    if verbose:
        print(f"  NVDB: {pages} pages, {kept} segments kept, {dropped} dropped, "
              f"{len(segs)} sequences", flush=True)

    # Merge consecutive segments of the same sequence into longer polylines.
    roads = []
    for seqid, seglist in segs.items():
        seglist.sort(key=lambda s: s["start"])
        # Chain by coincident endpoints (within 0.1 m); attrs from first seg.
        cur = None
        for s in seglist:
            if cur is None:
                cur = {"seqid": seqid, "category": s["category"],
                       "typeVeg": s["typeVeg"], "width": s["width"],
                       "pts": list(s["pts"])}
                continue
            a = cur["pts"][-1]
            b = s["pts"][0]
            if math.hypot(a[0] - b[0], a[1] - b[1]) <= 0.1:
                cur["pts"].extend(s["pts"][1:])
            else:
                roads.append(cur)
                cur = {"seqid": seqid, "category": s["category"],
                       "typeVeg": s["typeVeg"], "width": s["width"],
                       "pts": list(s["pts"])}
        if cur is not None:
            roads.append(cur)
    return roads


# ---- geometry helpers ----------------------------------------------------
def douglas_peucker(pts, tol):
    """Simplify a list of (E,N,h) using 2D (E,N) distance, keep h."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    P = np.asarray([(p[0], p[1]) for p in pts])
    while stack:
        i0, i1 = stack.pop()
        a, b = P[i0], P[i1]
        ab = b - a
        L = math.hypot(ab[0], ab[1])
        seg = P[i0 + 1:i1]
        if len(seg) == 0:
            continue
        if L < 1e-9:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            d = np.abs(ab[0] * (a[1] - seg[:, 1]) - (a[0] - seg[:, 0]) * ab[1]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            idx = i0 + 1 + k
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return [pts[i] for i in range(len(pts)) if keep[i]]


def polyline_length(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def clip_to_aabb(pts, minE, maxE, minN, maxN):
    """
    Split a UTM polyline into runs that lie inside the region AABB.
    Endpoints crossing the boundary are interpolated onto the edge (h too).
    Returns list of point-lists.
    """
    def inside(p):
        return minE <= p[0] <= maxE and minN <= p[1] <= maxN

    def intersect(p, q):
        # Liang-Barsky style: find param t in [0,1] where segment p->q first
        # crosses into/out of the box, clip to the boundary. Return clipped p,q.
        x0, y0 = p[0], p[1]
        x1, y1 = q[0], q[1]
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for pp, qq in ((-dx, x0 - minE), (dx, maxE - x0),
                       (-dy, y0 - minN), (dy, maxN - y0)):
            if pp == 0:
                if qq < 0:
                    return None
            else:
                t = qq / pp
                if pp < 0:
                    if t > t1:
                        return None
                    if t > t0:
                        t0 = t
                else:
                    if t < t0:
                        return None
                    if t < t1:
                        t1 = t
        if t0 > t1:
            return None

        def at(t):
            return (x0 + dx * t, y0 + dy * t, p[2] + (q[2] - p[2]) * t)
        return at(t0), at(t1)

    runs = []
    cur = []
    n = len(pts)
    for i in range(n - 1):
        p, q = pts[i], pts[i + 1]
        clip = intersect(p, q)
        if clip is None:
            if cur:
                runs.append(cur)
                cur = []
            continue
        cp, cq = clip
        if not cur:
            cur = [cp]
        else:
            # continue only if this run's tail matches cp (fully-inside chain)
            a = cur[-1]
            if math.hypot(a[0] - cp[0], a[1] - cp[1]) > 1e-6:
                runs.append(cur)
                cur = [cp]
        cur.append(cq)
        # if q was clipped (exits box), close the run
        if math.hypot(cq[0] - q[0], cq[1] - q[1]) > 1e-6:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def roads_to_local(roads, originE, originN, size, dp_tol=0.5, min_len=30.0):
    """
    UTM polylines -> local-world road entries per the frozen contract:
      world x = easting - originEasting          (east = +x)
      world z = -(northing - originNorthing)     (north = -z)
      world y = NVDB linestring Z height
    Clip to region AABB, drop < min_len, DP-simplify, round to 2 decimals.
    """
    half = size / 2.0
    minE, maxE = originE - half, originE + half
    minN, maxN = originN - half, originN + half
    out = []
    for rd in roads:
        for run in clip_to_aabb(rd["pts"], minE, maxE, minN, maxN):
            if len(run) < 2 or polyline_length(run) < min_len:
                continue
            run = douglas_peucker(run, dp_tol)
            if len(run) < 2 or polyline_length(run) < min_len:
                continue
            local = [[round(E - originE, 2),
                      round(h, 2),
                      round(-(N - originN), 2)] for (E, N, h) in run]
            out.append({
                "id": rd["seqid"],
                "category": rd["category"],
                "typeVeg": rd["typeVeg"],
                "width": rd["width"],
                "points": local,
            })
    return out


# ==========================================================================
# Preview
# ==========================================================================
def hillshade(z, res, azimuth=315.0, altitude=45.0):
    gy, gx = np.gradient(z.astype(np.float32), res)
    az = math.radians(360.0 - azimuth + 90.0)
    alt = math.radians(altitude)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    hs = (np.sin(alt) * np.sin(slope) +
          np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(hs, 0.0, 1.0)


def write_preview(pack_dir, heights, roads, size, res, dim):
    """Hillshade + road overlay. PNG via matplotlib if available, else TIF."""
    hs = hillshade(heights, res)
    # blend hillshade with a light elevation tint
    hmin, hmax = float(heights.min()), float(heights.max())
    tint = (heights - hmin) / max(1e-6, (hmax - hmin))
    img = (0.65 * hs + 0.35 * (0.3 + 0.7 * tint))
    img = np.clip(img, 0, 1)
    rgb = np.stack([img, img, img], axis=-1)

    # rasterize roads (local x,z -> pixel).  ix = (x + size/2)/res ; same as
    # column; iz = (z + size/2)/res = row.  x = E-originE, z = -(N-originN).
    def to_px(x, z):
        ix = (x + size / 2.0) / res
        iz = (z + size / 2.0) / res
        return ix, iz

    for rd in roads:
        col = {"E": (1, 0, 0), "R": (1, 0.4, 0), "F": (1, 0.85, 0),
               "K": (0.1, 0.6, 1), "P": (0.6, 0.6, 0.6),
               "S": (0.4, 0.4, 0.4)}.get(rd["category"], (0, 1, 0))
        pts = rd["points"]
        for i in range(len(pts) - 1):
            x0, _, z0 = pts[i]
            x1, _, z1 = pts[i + 1]
            ix0, iz0 = to_px(x0, z0)
            ix1, iz1 = to_px(x1, z1)
            steps = int(max(abs(ix1 - ix0), abs(iz1 - iz0))) + 1
            for s in range(steps + 1):
                t = s / steps
                px = int(round(ix0 + (ix1 - ix0) * t))
                py = int(round(iz0 + (iz1 - iz0) * t))
                if 0 <= px < dim and 0 <= py < dim:
                    rgb[py, px] = col

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(pack_dir, "preview.png")
        plt.imsave(out, rgb)
        return out
    except Exception:
        out = os.path.join(pack_dir, "preview.tif")
        tifffile.imwrite(out, (rgb * 255).astype(np.uint8))
        return out


# ==========================================================================
# Driver
# ==========================================================================
def build_region(name, lat, lon, size, res, out_dir, include_paths,
                 do_roads=True, do_preview=False):
    dim = int(round(size / res)) + 1
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    originE, originN = transformer.transform(lon, lat)
    print(f"Region '{name}': center ({lat},{lon}) -> UTM33 "
          f"E={originE:.1f} N={originN:.1f}, size={size} m, res={res} m, dim={dim}",
          flush=True)

    pack_dir = os.path.join(out_dir, name)
    os.makedirs(pack_dir, exist_ok=True)

    print("Fetching elevation (Kartverket DTM)...", flush=True)
    heights = fetch_heights(originE, originN, size, res, dim)
    hmin, hmax = float(heights.min()), float(heights.max())
    print(f"  heights: min={hmin:.2f} max={hmax:.2f} m", flush=True)

    roads_local = []
    if do_roads:
        print("Fetching roads (NVDB)...", flush=True)
        raw = fetch_roads(originE, originN, size, include_paths)
        roads_local = roads_to_local(raw, originE, originN, size)
        npts = sum(len(r["points"]) for r in roads_local)
        print(f"  roads: {len(roads_local)} polylines, {npts} points", flush=True)

    # --- write heights.f32 (row-major, iz*dim+ix; row 0 = north, no flip) ---
    heights.astype("<f4").tofile(os.path.join(pack_dir, "heights.f32"))

    # --- write roads.json ---
    with open(os.path.join(pack_dir, "roads.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 1, "roads": roads_local}, f,
                  ensure_ascii=False, separators=(",", ":"))

    # --- write region.json ---
    region = {
        "version": 1,
        "name": name,
        "crs": "EPSG:25833",
        "originEasting": round(originE, 4),
        "originNorthing": round(originN, 4),
        "worldSize": float(size),
        "dim": dim,
        "heightMin": round(hmin, 3),
        "heightMax": round(hmax, 3),
        "seaLevel": 0.0,
        "heights": "heights.f32",
        "roads": "roads.json",
        "attribution": ("Terrain: © Kartverket, hoydedata.no (CC BY 4.0). "
                        "Roads: © Statens vegvesen, NVDB (NLOD)."),
    }
    with open(os.path.join(pack_dir, "region.json"), "w", encoding="utf-8") as f:
        json.dump(region, f, ensure_ascii=False, indent=2)

    preview_path = None
    if do_preview:
        print("Rendering preview...", flush=True)
        preview_path = write_preview(pack_dir, heights, roads_local, size, res, dim)
        print(f"  preview: {preview_path}", flush=True)

    return pack_dir, region, heights, roads_local, preview_path


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Build a threepp region pack from Norwegian open geodata.")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()))
    ap.add_argument("--center", help="lat,lon in WGS84 (e.g. 62.4482,7.6714)")
    ap.add_argument("--size", type=float, default=8000.0, help="region size in meters (default 8000)")
    ap.add_argument("--res", type=float, default=2.0, help="grid resolution m/cell (default 2)")
    ap.add_argument("--name", help="region name (defaults to preset name)")
    ap.add_argument("--out", default=r"C:\dev\threepp\geodata", help="output root dir")
    ap.add_argument("--include-paths", action="store_true", help="include foot/bike paths")
    ap.add_argument("--no-roads", action="store_true", help="skip road fetch")
    ap.add_argument("--preview", action="store_true", help="render hillshade+roads preview")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.preset:
        lat, lon = PRESETS[args.preset]
        name = args.name or args.preset
    elif args.center:
        lat_s, lon_s = args.center.split(",")
        lat, lon = float(lat_s), float(lon_s)
        name = args.name or "region"
    else:
        print("error: supply --preset or --center", file=sys.stderr)
        return 2

    build_region(name, lat, lon, args.size, args.res, args.out,
                 include_paths=args.include_paths,
                 do_roads=not args.no_roads,
                 do_preview=args.preview)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
