#!/usr/bin/env python3
"""
fetch_norway_terrain.py - build a threepp "region pack" from live Norwegian
open geodata: national DTM elevation (Kartverket) + road network (NVDB).

Output (frozen C++ contract) written to <out>/<name>/:
    region.json     metadata + coordinate mapping
    heights.f32     dim*dim little-endian float32, row-major (iz*dim + ix)
    roads.json      polylines in local world coords
    buildings.json  extruded-footprint buildings (--buildings)
    texture.png     square RGB basemap drape, row 0 = north (--texture)
    preview.(png|tif)   optional hillshade + roads/buildings overlay (--preview)

Data sources (verified live 2026-07-15, buildings 2026-07-17):
  Elevation: Kartverket national DTM via WCS 1.0.0 GetCoverage.
             CRS EPSG:25833 (UTM33). License CC BY 4.0, (c) Kartverket.
  Roads:     NVDB API Les v4 vegnett/segmentert. License NLOD,
             (c) Statens vegvesen.
  Buildings: OSM footprints via Overpass (ODbL, (c) OpenStreetMap
             contributors); heights measured from the Kartverket national
             DOM (surface model) minus the DTM (nDSM) where OSM lacks tags.
  Texture:   Kartverket topo raster via WMS GetMap (topo / topograatone).
             License CC BY 4.0, (c) Kartverket. Verified live 2026-08-25.

Usage:
    python fetch_norway_terrain.py --preset trollstigen
    python fetch_norway_terrain.py --center 62.4482,7.6714 --size 8000 --res 2 --name myregion
    python fetch_norway_terrain.py --preset alesund --preview --include-paths --buildings
    python fetch_norway_terrain.py --preset trollstigen --texture topo
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
# Packs are written to the repo's (gitignored) geodata/ root by default -- the
# same place examples/extras/terrain/norway_terrain.cpp reads them from as
# PROJECT_FOLDER/geodata/<name>. --out overrides it.
# --------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO_ROOT, "geodata")

# --------------------------------------------------------------------------
# Presets  (lat, lon in WGS84).
#   trollstigen: centered on the Fv63 switchback wall (the famous 11 hairpins).
#     The wall climbs ~62.449N..62.456N; this center places it in the middle
#     of the region.  Peaks Bispen/Kongen (>1400 m) fall within +-4 km.
#   aalesund: coastal town center (exercises the sea / nodata=0 path).
#     (Spelled "aalesund" to match the pack dir; "alesund" stays accepted as
#     an alias so older invocations keep working.)
# --------------------------------------------------------------------------
#   geiranger: the Seven Sisters (De syv søstre) wall on the north side of
#     Geirangerfjorden.  The falls plunge ~250 m off the shelf at Knivsflå
#     into the fjord; the center below sits ON the water just south of the
#     falls so a 4 km square holds the whole wall (0 -> ~1400 m), the fjord
#     channel and the opposite (Skageflå) side.  Verify with --preview.
PRESETS = {
    "trollstigen": (62.4525, 7.6675),
    "aalesund": (62.4722, 6.1495),
    "alesund": (62.4722, 6.1495),  # alias for aalesund (deprecated spelling)
    "geiranger": (62.1090, 7.1122),
}

# WCS elevation endpoint (ArcGIS - only the WCS 1.0.0 bbox form works here;
# the OGC 2.0.1 subset= form 400s on this server).
WCS_URL = "https://wms.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
WCS_COVERAGE = "nhm_dtm_topo_25833"

# WCS surface-model endpoint (DOM = terrain + buildings/vegetation). Same
# server family / request form as the DTM; used to MEASURE building heights
# (nDSM = DOM - DTM) where OSM has no height tag. Verified live 2026-07-17.
WCS_DOM_URL = "https://wms.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25833"
WCS_DOM_COVERAGE = "nhm_dom_topo_25833"

# Open Kartverket topo raster WMS (basemap texture, --texture). Same geonorge
# server family as the WCS. WMS GetMap with bbox in EASTING,NORTHING order
# (verified live 2026-08-25: the E,N form returns map content on both 1.3.0
# and 1.1.1; a swapped N,E bbox returns a blank tile). Tiles up to 4096 px
# per dimension verified. License CC BY 4.0, (c) Kartverket.
WMS_TEXTURE = {
    "topo": ("https://wms.geonorge.no/skwms1/wms.topo", "topo"),
    "topograatone": ("https://wms.geonorge.no/skwms1/wms.topograatone",
                     "topograatone"),
}
TEX_CHUNK = 4096  # max px per WMS GetMap dimension (verified)

# NVDB road network endpoint.
NVDB_URL = "https://nvdbapiles.atlas.vegvesen.no/vegnett/api/v4/veglenkesekvenser/segmentert"
NVDB_HEADERS = {"X-Client": "threepp-geodata", "Accept": "application/json"}

# Overpass endpoint for OSM building footprints. Norwegian OSM buildings are
# largely a cadastre (Matrikkelen) import, so footprint coverage is good; the
# `height` tag is rare (measured 5/254 in central Aalesund) and
# `building:levels` present on ~half, hence the nDSM fallback above.
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Building height fallbacks (m). Priority: OSM height tag > nDSM (measured)
# > building:levels * LEVEL_HEIGHT > per-type default.
LEVEL_HEIGHT = 3.0
TYPE_HEIGHT_DEFAULT = {
    "garage": 3.0, "garages": 3.0, "shed": 3.0, "hut": 3.5, "cabin": 4.5,
    "roof": 3.5, "carport": 3.0, "service": 3.5, "farm_auxiliary": 5.0,
    "house": 7.0, "detached": 7.0, "semidetached_house": 7.0, "farm": 7.0,
    "terrace": 7.0, "bungalow": 4.0, "residential": 8.0, "apartments": 12.0,
    "industrial": 8.0, "warehouse": 9.0, "retail": 6.0, "commercial": 10.0,
    "office": 12.0, "school": 8.0, "civic": 8.0, "hospital": 12.0,
    "church": 14.0, "cathedral": 25.0, "hotel": 15.0,
}
TYPE_HEIGHT_FALLBACK = 6.0
BUILDING_MIN_AREA = 10.0   # m^2 - drop micro-footprints below this
BUILDING_MIN_HEIGHT = 2.5  # m - clamp floor
BUILDING_MAX_HEIGHT = 150.0# m - clamp ceiling (sanity)
NDSM_MIN_VALID = 2.0       # m - below this the lidar predates the building
                           # (or the footprint is mislocated): treat as invalid

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
def _wcs_tile(minE, minN, maxE, maxN, w, h, session, retries=4,
              url=WCS_URL, coverage=WCS_COVERAGE):
    """Fetch one node-aligned elevation tile as a float32 (h, w) array."""
    params = {
        "service": "WCS", "version": "1.0.0", "request": "GetCoverage",
        "coverage": coverage, "crs": "EPSG:25833",
        "bbox": f"{minE},{minN},{maxE},{maxN}",
        "width": int(w), "height": int(h), "format": "GeoTIFF",
    }
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=120)
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


def _fetch_grid_window(originE, originN, size, res, row0, row1, col0, col1,
                       url=WCS_URL, coverage=WCS_COVERAGE, label="DTM"):
    """
    Fetch node rows [row0,row1) x cols [col0,col1) of the region's node grid
    as a float32 (row1-row0, col1-col0) array.

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
    grid = np.zeros((row1 - row0, col1 - col0), dtype=np.float32)

    west = originE - size / 2.0
    north = originN + size / 2.0

    col_bounds = list(range(col0, col1, CHUNK)) + [col1]
    row_bounds = list(range(row0, row1, CHUNK)) + [row1]
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
            print(f"  {label} tile {t}/{ntiles}  rows[{r0}:{r1}] cols[{c0}:{c1}] "
                  f"({cw}x{rh})", flush=True)
            tile = _wcs_tile(minE, minN, maxE, maxN, cw, rh, session,
                             url=url, coverage=coverage)
            grid[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = tile
            time.sleep(0.15)  # be polite

    # Nodata handling: the DTM already encodes sea as 0.0 (verified: no
    # negative sentinel, min ~ -0.003 float noise; the DOM behaves the same).
    # Guard against NaN and any large-negative sentinel, then clamp the floor
    # to sea level (0).
    grid[~np.isfinite(grid)] = 0.0
    grid[grid < -1e4] = 0.0
    np.clip(grid, 0.0, None, out=grid)
    return grid


def fetch_heights(originE, originN, size, res, dim):
    """Fetch the full dim x dim node-aligned DTM height grid."""
    return _fetch_grid_window(originE, originN, size, res, 0, dim, 0, dim)


def fetch_canopy(heights, originE, originN, size, res, dim):
    """
    Canopy height model (CHM) = DOM - DTM over the whole region, quantised to
    unsigned quarter-metres.

    WHY quarter-metre u8: the consumer only needs "is there forest here and
    roughly how tall" -- 0.25 m steps over 0..63.75 m covers every Norwegian
    tree with room to spare, and one byte per node keeps a 4001^2 grid at
    16 MB instead of 64 MB.  Values are clamped, so buildings and power
    pylons saturate at 63.75 m rather than wrapping.

    The DOM is the same national NHM lidar campaign as the DTM and is fetched
    on the SAME node grid, so no resampling is involved and the difference is
    exact per node.  Sea and nodata are 0 in both models, so water reads 0.
    """
    dom = _fetch_grid_window(originE, originN, size, res, 0, dim, 0, dim,
                             url=WCS_DOM_URL, coverage=WCS_DOM_COVERAGE,
                             label="DOM")
    chm = dom - heights
    np.clip(chm, 0.0, 63.75, out=chm)
    return np.rint(chm * 4.0).astype(np.uint8)


# ==========================================================================
# Texture (Kartverket topo raster WMS)
# ==========================================================================
def _require_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise SystemExit("--texture requires pillow: pip install pillow")


def _wms_tile(url, layer, minE, minN, maxE, maxN, w, h, session, Image,
              retries=4):
    """Fetch one WMS GetMap tile as a uint8 (h, w, 3) RGB array."""
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap",
        "layers": layer, "styles": "", "crs": "EPSG:25833",
        "bbox": f"{minE},{minN},{maxE},{maxN}",
        "width": int(w), "height": int(h), "format": "image/png",
    }
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=180)
            if r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n":
                arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
                if arr.shape == (h, w, 3):
                    return arr
                raise ValueError(f"tile shape {arr.shape} != {(h, w, 3)}")
            last = f"HTTP {r.status_code} {r.content[:80]!r}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"WMS tile failed after {retries} tries: {last}")


def fetch_texture(originE, originN, size, layer, tex_res):
    """
    Fetch the region's basemap texture as a uint8 (px, px, 3) RGB array,
    px = round(size / tex_res).

    Unlike the node-aligned elevation grid, the texture uses plain image
    coverage: pixel (row, col) covers the tex_res x tex_res square starting
    at (west + col*tex_res, north - (row+1)*tex_res). Row 0 is the NORTH
    edge (WMS images are north-up, top-down), matching the heights
    convention, so u = (x + size/2)/size, v = (z + size/2)/size maps local
    world coords straight onto the image. Tiles are cut on pixel boundaries
    with edge-aligned bboxes, so they stitch with no overlap.
    """
    url, wms_layer = WMS_TEXTURE[layer]
    Image = _require_pil()
    px = int(round(size / tex_res))
    west = originE - size / 2.0
    north = originN + size / 2.0

    session = requests.Session()
    session.headers.update({"User-Agent": "threepp-geodata/1.0"})
    rgb = np.zeros((px, px, 3), dtype=np.uint8)

    bounds = list(range(0, px, TEX_CHUNK)) + [px]
    ntiles = (len(bounds) - 1) ** 2
    t = 0
    for ri in range(len(bounds) - 1):
        r0, r1 = bounds[ri], bounds[ri + 1]
        maxN = north - r0 * tex_res
        minN = north - r1 * tex_res
        for ci in range(len(bounds) - 1):
            c0, c1 = bounds[ci], bounds[ci + 1]
            minE = west + c0 * tex_res
            maxE = west + c1 * tex_res
            t += 1
            print(f"  {layer} tile {t}/{ntiles}  rows[{r0}:{r1}] "
                  f"cols[{c0}:{c1}] ({c1 - c0}x{r1 - r0})", flush=True)
            rgb[r0:r1, c0:c1] = _wms_tile(url, wms_layer, minE, minN, maxE,
                                          maxN, c1 - c0, r1 - r0, session,
                                          Image)
            time.sleep(0.15)  # be polite
    return rgb


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
# Buildings
# ==========================================================================
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_meters(v):
    """OSM numeric tag ('12', '12.5 m', '12,5') -> float, else None."""
    if v is None:
        return None
    m = _NUM_RE.search(str(v).replace(",", "."))
    return float(m.group(0)) if m else None


def _assemble_rings(seg_lists):
    """
    Chain relation-member way segments into closed rings by shared endpoints
    (Overpass emits exact coincident coords for shared nodes). Returns list of
    closed rings in OPEN form (first point NOT repeated); unclosable leftovers
    are dropped (counted by the caller via the length difference).
    """
    rings = []
    open_segs = []
    for s in seg_lists:
        if len(s) < 2:
            continue
        if s[0] == s[-1] and len(s) >= 4:
            rings.append(s[:-1])
        else:
            open_segs.append(list(s))
    while open_segs:
        cur = open_segs.pop()
        progress = True
        while progress and cur[0] != cur[-1]:
            progress = False
            for i, s in enumerate(open_segs):
                if s[0] == cur[-1]:
                    cur.extend(s[1:])
                elif s[-1] == cur[-1]:
                    cur.extend(reversed(s[:-1]))
                else:
                    continue
                open_segs.pop(i)
                progress = True
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur[:-1])
    return rings


def fetch_buildings_osm(originE, originN, size, verbose=True):
    """
    Fetch OSM building footprints intersecting the region via Overpass.

    Returns list of dicts {id, tags, outer, holes} with rings as lists of
    (lat, lon), rings in open form. Multipolygon relations are assembled;
    relations with several outer rings become several entries (id suffixed
    .0/.1/...), holes assigned to the outer ring that contains them.
    """
    # Region corners UTM33 -> WGS84; the lat/lon AABB over-covers the rotated
    # square slightly, extra buildings get dropped by the centroid filter.
    inv = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
    half = size / 2.0
    lons, lats = [], []
    for sE in (-1, 1):
        for sN in (-1, 1):
            lon, lat = inv.transform(originE + sE * half, originN + sN * half)
            lons.append(lon)
            lats.append(lat)
    bbox = f"{min(lats)},{min(lons)},{max(lats)},{max(lons)}"

    query = (f'[out:json][timeout:180];'
             f'(way["building"]({bbox});relation["building"]({bbox}););'
             f'out geom;')
    mirrors = [OVERPASS_URL, "https://overpass.kumi.systems/api/interpreter"]
    session = requests.Session()
    session.headers.update({"User-Agent": "threepp-geodata/1.0"})
    d = None
    last = None
    for attempt in range(4):
        url = mirrors[attempt % len(mirrors)]
        try:
            r = session.post(url, data={"data": query}, timeout=300)
            if r.status_code == 200:
                d = r.json()
                break
            last = f"HTTP {r.status_code} {r.content[:120]!r}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(5.0 * (attempt + 1))
    if d is None:
        raise RuntimeError(f"Overpass failed after retries: {last}")

    out = []
    n_ways = n_rels = n_bad_rings = 0
    for el in d.get("elements", []):
        tags = el.get("tags", {}) or {}
        if el["type"] == "way":
            geom = el.get("geometry") or []
            ring = [(g["lat"], g["lon"]) for g in geom]
            if len(ring) >= 4 and ring[0] == ring[-1]:
                out.append({"id": f'w{el["id"]}', "tags": tags,
                            "outer": ring[:-1], "holes": []})
                n_ways += 1
            else:
                n_bad_rings += 1
        elif el["type"] == "relation":
            outers_raw, inners_raw = [], []
            for m in el.get("members", []):
                geom = m.get("geometry") or []
                seg = [(g["lat"], g["lon"]) for g in geom]
                if m.get("role") == "inner":
                    inners_raw.append(seg)
                else:  # 'outer' or blank role: treat as outer
                    outers_raw.append(seg)
            outers = _assemble_rings(outers_raw)
            inners = _assemble_rings(inners_raw)
            n_bad_rings += (len(outers_raw) and not outers)
            if not outers:
                continue
            n_rels += 1
            if len(outers) == 1:
                out.append({"id": f'r{el["id"]}', "tags": tags,
                            "outer": outers[0], "holes": inners})
            else:
                # Assign each hole to the outer ring containing its 1st vertex.
                groups = [[] for _ in outers]
                for hole in inners:
                    hp = hole[0]
                    for oi, o in enumerate(outers):
                        if _point_in_ring_ll(hp, o):
                            groups[oi].append(hole)
                            break
                for oi, o in enumerate(outers):
                    out.append({"id": f'r{el["id"]}.{oi}', "tags": tags,
                                "outer": o, "holes": groups[oi]})
    if verbose:
        print(f"  Overpass: {n_ways} way footprints, {n_rels} multipolygons, "
              f"{n_bad_rings} unclosed dropped", flush=True)
    return out


def _point_in_ring_ll(p, ring):
    """Even-odd test for a single (lat, lon) point against a (lat, lon) ring."""
    lat, lon = p
    inside = False
    a = ring[-1]
    for b in ring:
        if (a[0] > lat) != (b[0] > lat):
            t = (lat - a[0]) / (b[0] - a[0])
            if lon < a[1] + t * (b[1] - a[1]):
                inside = not inside
        a = b
    return inside


def _ring_area_xz(pts):
    """Signed shoelace area of an open ring of (x, z) pairs."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, z0 = pts[i]
        x1, z1 = pts[(i + 1) % n]
        a += x0 * z1 - x1 * z0
    return 0.5 * a


def _points_in_ring(px, pz, ring):
    """Vectorized even-odd test: which points (px, pz) are inside the open
    (x, z) ring? px/pz are equal-shape numpy arrays."""
    inside = np.zeros(px.shape, dtype=bool)
    x0, z0 = ring[-1]
    for (x1, z1) in ring:
        if z0 != z1:
            cond = (z0 > pz) != (z1 > pz)
            t = (pz - z0) / (z1 - z0)
            inside ^= cond & (px < x0 + t * (x1 - x0))
        x0, z0 = x1, z1
    return inside


def buildings_to_local(raw, originE, originN, size):
    """
    (lat, lon) footprints -> local-world building entries (frozen contract):
      x = easting - originEasting, z = -(northing - originNorthing)
    Keeps buildings whose centroid lies inside the region AABB and whose
    footprint area >= BUILDING_MIN_AREA. Outer rings are normalized to
    POSITIVE shoelace area in (x, z), holes to negative. Coords rounded to
    2 decimals. Height fields are filled later by resolve_building_heights.
    """
    tf = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    half = size / 2.0

    def to_local(ring_ll):
        lats = [p[0] for p in ring_ll]
        lons = [p[1] for p in ring_ll]
        Es, Ns = tf.transform(lons, lats)
        return [(E - originE, -(N - originN)) for E, N in zip(Es, Ns)]

    out = []
    n_outside = n_tiny = 0
    for b in raw:
        outer = to_local(b["outer"])
        area = _ring_area_xz(outer)
        if abs(area) < BUILDING_MIN_AREA:
            n_tiny += 1
            continue
        cx = sum(p[0] for p in outer) / len(outer)
        cz = sum(p[1] for p in outer) / len(outer)
        if not (-half <= cx <= half and -half <= cz <= half):
            n_outside += 1
            continue
        if area < 0:
            outer.reverse()
        holes = []
        for h in b["holes"]:
            hl = to_local(h)
            if _ring_area_xz(hl) > 0:
                hl.reverse()
            holes.append(hl)
        tags = b["tags"]
        levels = _parse_meters(tags.get("building:levels"))
        if levels is not None and not (0 < levels <= 100):
            levels = None
        out.append({
            "id": b["id"],
            "type": tags.get("building", "yes"),
            "levels": levels,
            "_heightTag": _parse_meters(tags.get("height")),
            # Real appearance tags (rare — ~0.1% in Aalesund — but they win
            # over the hashed palette in the consumer when present).
            "colour": tags.get("building:colour"),
            "roofColour": tags.get("roof:colour"),
            "roofShape": tags.get("roof:shape"),
            "outer": [[round(x, 2), round(z, 2)] for x, z in outer],
            "holes": [[[round(x, 2), round(z, 2)] for x, z in h] for h in holes],
        })
    print(f"  local: {len(out)} kept, {n_outside} outside region, "
          f"{n_tiny} tiny (<{BUILDING_MIN_AREA} m^2) dropped", flush=True)
    return out


def resolve_building_heights(blds, heights, originE, originN, size, res, dim):
    """
    Fill height/groundMin/groundMax/heightSource on each building, in place.

    Ground comes from the already-fetched DTM grid. Measured height comes from
    an nDSM (DOM - DTM) fetched ONLY over the node window covering the
    footprints. Contract: roof top = groundMin + height, so tag/levels/default
    heights (which are relative to local ground) get the ground relief
    (groundMax - groundMin) folded in; the nDSM p90 already measures from
    near-groundMin on slopes.
    """
    if not blds:
        return
    half = size / 2.0

    def node_range(lo, hi):
        i0 = int(math.floor((lo + half) / res)) - 2
        i1 = int(math.ceil((hi + half) / res)) + 2
        return max(0, i0), min(dim - 1, i1)

    # Window over all footprints.
    allx = [p[0] for b in blds for p in b["outer"]]
    allz = [p[1] for b in blds for p in b["outer"]]
    wc0, wc1 = node_range(min(allx), max(allx))
    wr0, wr1 = node_range(min(allz), max(allz))
    print(f"  DOM window: rows[{wr0}:{wr1 + 1}] cols[{wc0}:{wc1 + 1}] "
          f"({(wr1 - wr0 + 1) * (wc1 - wc0 + 1) / 1e6:.1f} Mnodes)", flush=True)
    dom = _fetch_grid_window(originE, originN, size, res,
                             wr0, wr1 + 1, wc0, wc1 + 1,
                             url=WCS_DOM_URL, coverage=WCS_DOM_COVERAGE,
                             label="DOM")
    dtm_win = heights[wr0:wr1 + 1, wc0:wc1 + 1]
    ndsm = np.maximum(dom - dtm_win, 0.0)

    src_count = {}
    for b in blds:
        xs = [p[0] for p in b["outer"]]
        zs = [p[1] for p in b["outer"]]
        c0, c1 = node_range(min(xs), max(xs))
        r0, r1 = node_range(min(zs), max(zs))
        ix = np.arange(c0, c1 + 1)
        iz = np.arange(r0, r1 + 1)
        px, pz = np.meshgrid(ix * res - half, iz * res - half)
        inside = _points_in_ring(px, pz, b["outer"])
        for h in b["holes"]:
            inside &= ~_points_in_ring(px, pz, h)

        rows = np.arange(r0, r1 + 1)[:, None] - wr0
        cols = np.arange(c0, c1 + 1)[None, :] - wc0
        g = dtm_win[rows, cols][inside]
        nd = ndsm[rows, cols][inside]
        n_in = int(inside.sum())
        if n_in >= 1:
            gmin, gmax = float(g.min()), float(g.max())
        else:
            # Degenerate/tiny footprint: nearest node to the centroid.
            cx = sum(xs) / len(xs)
            cz = sum(zs) / len(zs)
            ic = min(max(int(round((cx + half) / res)), wc0), wc1) - wc0
            irr = min(max(int(round((cz + half) / res)), wr0), wr1) - wr0
            gmin = gmax = float(dtm_win[irr, ic])
        relief = gmax - gmin

        levels = b["levels"]
        htag = b.pop("_heightTag")
        ndsm_p90 = float(np.percentile(nd, 90)) if n_in >= 3 else 0.0
        if htag is not None and htag > 0:
            h, src = htag + relief, "tag"
        elif ndsm_p90 >= NDSM_MIN_VALID:
            h, src = ndsm_p90, "ndsm"
        elif levels is not None:
            h, src = levels * LEVEL_HEIGHT + relief, "levels"
        else:
            h = TYPE_HEIGHT_DEFAULT.get(b["type"], TYPE_HEIGHT_FALLBACK) + relief
            src = "default"
        h = min(max(h, BUILDING_MIN_HEIGHT), BUILDING_MAX_HEIGHT)

        b["height"] = round(h, 2)
        b["heightSource"] = src
        b["groundMin"] = round(gmin, 2)
        b["groundMax"] = round(gmax, 2)
        src_count[src] = src_count.get(src, 0) + 1
    print(f"  height sources: {src_count}", flush=True)


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


def write_preview(pack_dir, heights, roads, size, res, dim, buildings=None):
    """Hillshade + road/building overlay. PNG via matplotlib, else TIF."""
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

    # Building footprint outlines (magenta) on top of the roads.
    for b in buildings or []:
        ring = b["outer"]
        for i in range(len(ring)):
            x0, z0 = ring[i]
            x1, z1 = ring[(i + 1) % len(ring)]
            ix0, iz0 = to_px(x0, z0)
            ix1, iz1 = to_px(x1, z1)
            steps = int(max(abs(ix1 - ix0), abs(iz1 - iz0))) + 1
            for s in range(steps + 1):
                t = s / steps
                px = int(round(ix0 + (ix1 - ix0) * t))
                py = int(round(iz0 + (iz1 - iz0) * t))
                if 0 <= px < dim and 0 <= py < dim:
                    rgb[py, px] = (0.85, 0.15, 0.75)

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
                 do_roads=True, do_buildings=False, do_preview=False,
                 texture=None, texture_res=2.0, do_canopy=False):
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

    canopy = None
    if do_canopy:
        print("Fetching canopy height model (Kartverket DOM - DTM)...", flush=True)
        canopy = fetch_canopy(heights, originE, originN, size, res, dim)
        forest_frac = float((canopy > 8).mean())  # >8 quarter-m = >2 m
        print(f"  canopy: max={canopy.max() / 4.0:.2f} m, "
              f"{100.0 * forest_frac:.1f}% of cells above 2 m", flush=True)

    roads_local = []
    if do_roads:
        print("Fetching roads (NVDB)...", flush=True)
        raw = fetch_roads(originE, originN, size, include_paths)
        roads_local = roads_to_local(raw, originE, originN, size)
        npts = sum(len(r["points"]) for r in roads_local)
        print(f"  roads: {len(roads_local)} polylines, {npts} points", flush=True)

    buildings_local = []
    if do_buildings:
        print("Fetching buildings (OSM Overpass)...", flush=True)
        raw_b = fetch_buildings_osm(originE, originN, size)
        buildings_local = buildings_to_local(raw_b, originE, originN, size)
        if buildings_local:
            print("Measuring building heights (Kartverket DOM nDSM)...", flush=True)
            resolve_building_heights(buildings_local, heights,
                                     originE, originN, size, res, dim)

    if texture:
        print(f"Fetching texture (Kartverket {texture} WMS)...", flush=True)
        tex = fetch_texture(originE, originN, size, texture, texture_res)
        Image = _require_pil()
        tex_path = os.path.join(pack_dir, "texture.png")
        Image.fromarray(tex).save(tex_path)
        print(f"  texture: {tex.shape[1]}x{tex.shape[0]} px "
              f"({texture_res} m/px) -> {tex_path}", flush=True)

    # --- write heights.f32 (row-major, iz*dim+ix; row 0 = north, no flip) ---
    heights.astype("<f4").tofile(os.path.join(pack_dir, "heights.f32"))

    # --- write canopy.u8 (same row-major node layout as heights.f32) ---
    if canopy is not None:
        canopy.tofile(os.path.join(pack_dir, "canopy.u8"))

    # --- write roads.json ---
    with open(os.path.join(pack_dir, "roads.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 1, "roads": roads_local}, f,
                  ensure_ascii=False, separators=(",", ":"))

    # --- write buildings.json (only when requested) ---
    if do_buildings:
        slim = []
        for b in buildings_local:
            e = {"id": b["id"], "type": b["type"],
                 "height": b["height"], "heightSource": b["heightSource"],
                 "groundMin": b["groundMin"], "groundMax": b["groundMax"],
                 "outer": b["outer"]}
            if b["levels"] is not None:
                e["levels"] = b["levels"]
            for k in ("colour", "roofColour", "roofShape"):
                if b.get(k):
                    e[k] = b[k]
            if b["holes"]:
                e["holes"] = b["holes"]
            slim.append(e)
        with open(os.path.join(pack_dir, "buildings.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"version": 1, "buildings": slim}, f,
                      ensure_ascii=False, separators=(",", ":"))

    # --- write region.json ---
    attribution = ("Terrain: © Kartverket, hoydedata.no (CC BY 4.0). "
                   "Roads: © Statens vegvesen, NVDB (NLOD).")
    if do_buildings:
        attribution += " Buildings: © OpenStreetMap contributors (ODbL)."
    if texture:
        attribution += (f" Basemap texture: © Kartverket, {texture} WMS "
                        f"(CC BY 4.0).")
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
        "attribution": attribution,
    }
    if do_buildings:
        region["buildings"] = "buildings.json"
    if canopy is not None:
        region["canopy"] = "canopy.u8"
        region["canopyScale"] = 0.25  # u8 value * scale = canopy height in m
    if texture:
        region["texture"] = "texture.png"
        region["textureLayer"] = texture
    with open(os.path.join(pack_dir, "region.json"), "w", encoding="utf-8") as f:
        json.dump(region, f, ensure_ascii=False, indent=2)

    preview_path = None
    if do_preview:
        print("Rendering preview...", flush=True)
        preview_path = write_preview(pack_dir, heights, roads_local, size, res,
                                     dim, buildings_local)
        print(f"  preview: {preview_path}", flush=True)

    return pack_dir, region, heights, roads_local, preview_path


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Build a threepp region pack from Norwegian open geodata.")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()))
    ap.add_argument("--center", help="lat,lon in WGS84 (e.g. 62.4482,7.6714)")
    ap.add_argument("--size", type=float, default=8000.0, help="region size in meters (default 8000)")
    ap.add_argument("--res", type=float, default=2.0, help="grid resolution m/cell (default 2)")
    ap.add_argument("--name", help="region name (defaults to preset name)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output root dir (default {DEFAULT_OUT})")
    ap.add_argument("--include-paths", action="store_true", help="include foot/bike paths")
    ap.add_argument("--no-roads", action="store_true", help="skip road fetch")
    ap.add_argument("--buildings", action="store_true",
                    help="fetch OSM building footprints + DOM nDSM heights")
    ap.add_argument("--texture", choices=sorted(WMS_TEXTURE.keys()),
                    help="fetch a basemap texture from the open Kartverket "
                         "raster WMS (writes texture.png)")
    ap.add_argument("--texture-res", type=float, default=2.0,
                    help="texture resolution m/px (default 2)")
    ap.add_argument("--canopy", action="store_true",
                    help="fetch the DOM over the whole region and write a "
                         "canopy height model (canopy.u8, quarter-metres)")
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
                 do_buildings=args.buildings,
                 do_preview=args.preview,
                 texture=args.texture,
                 texture_res=args.texture_res,
                 do_canopy=args.canopy)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
