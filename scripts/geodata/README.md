# geodata — Norwegian region packs for threepp

`fetch_norway_terrain.py` builds a **region pack** from live Norwegian open
geodata: national DTM elevation (Kartverket) + the road network (NVDB). The
C++ engine consumes the pack; this tool is the preprocessor.

## Install

Active interpreter: Python 3.14. Dependencies:

```
pip install requests numpy tifffile pyproj
# optional, only for a PNG (not TIF) preview:
pip install matplotlib
```

## Usage

```
python fetch_norway_terrain.py --preset trollstigen
python fetch_norway_terrain.py --preset alesund --preview --include-paths
python fetch_norway_terrain.py --center 62.4482,7.6714 --size 8000 --res 2 --name myregion
```

| flag | default | meaning |
|------|---------|---------|
| `--preset` | — | `trollstigen` or `alesund` |
| `--center lat,lon` | — | WGS84 center (alternative to `--preset`) |
| `--size` | 8000 | region size in meters (square) |
| `--res` | 2 | grid resolution, m/cell |
| `--name` | preset name | output subdir name |
| `--out` | `C:\dev\threepp\geodata` | output root |
| `--include-paths` | off | also emit foot/bike paths (Gangveg, Gang- og sykkelveg, Gågate) |
| `--no-roads` | off | skip the NVDB fetch |
| `--preview` | off | render `preview.png` (hillshade + roads) |

Output lands in `<out>/<name>/` as `region.json`, `heights.f32`, `roads.json`
(+ `preview.png` with `--preview`). The output format is the **frozen C++
contract** — see the tool docstring / the spec; do not change it here.

`verify_pack.py <pack_dir>` cross-checks the pack (see Verification below).

## Presets

- **trollstigen** = `62.4525, 7.6675`. Nudged ~800 m north of the raw village
  coordinate so the famous **Fv63 switchback wall** (the 11 hairpins) sits in
  the middle of an 8 km region. Surrounding peaks (>1400 m, e.g. Bispen/Kongen)
  fall inside. The Trollstigen road is NVDB `vegkategori = F` (fylkesvei).
- **alesund** = `62.4722, 6.1495`. Coastal town — exercises the sea path
  (sea = 0 m). Contains the E136 European route (category `E`).

## Data sources

### Elevation — Kartverket national DTM (WCS)

`https://wms.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833`, coverage
`nhm_dtm_topo_25833`, **WCS 1.0.0 GetCoverage** with a `bbox=` request in
**EPSG:25833 (UTM33)**. Returns a float32 GeoTIFF. License CC BY 4.0,
© Kartverket (hoydedata.no).

Quirks / decisions:
- The OGC **WCS 2.0.1 `subset=` form 400s** on this ArcGIS server — only the
  1.0.0 `bbox` form works.
- **Pixel-is-area** convention: `ModelPixelScale = bbox_span / width`. To land
  pixel *centers* exactly on grid nodes we request a bbox expanded by `res/2`
  on every side with `width = height = dim`. Tiles are cut on node-index
  boundaries (≤2000 px each) so they stitch with **no overlap and no
  resampling**. `dim = round(size/res) + 1` (4001 for 8000 m @ 2 m).
- **Orientation:** the GeoTIFF `ModelTiepoint` maps raster (0,0) → (minE,
  maxN), i.e. **row 0 is the NORTH edge**. This matches the frozen mapping
  (`iz=0 ↔ north`), so rows are written with **no vertical flip**.
- **Nodata:** the DTM already encodes sea as `0.0` (verified — no negative
  sentinel; sea samples are `0.0 ± 0.003` float noise). We still guard against
  NaN / large-negative sentinels, then clamp the floor to `0.0` (sea level).

### Roads — NVDB API Les v4 (vegnett)

`https://nvdbapiles.atlas.vegvesen.no/vegnett/api/v4/veglenkesekvenser/segmentert?kartutsnitt=<minE,minN,maxE,maxN>&antall=1000`
with header `X-Client: threepp-geodata`. License NLOD, © Statens vegvesen.

Quirks / decisions:
- `srid=25833` is **rejected (400)** — omit it; the default output is already
  planar UTM33 (`geometri.wkt = LINESTRING Z (E N h, ...)`), parsed with a
  simple regex (no shapely).
- **Pagination:** follow `metadata.neste.start` until the page is empty / no
  `neste`.
- **Detail-level dedup (the important filter).** The network carries the same
  physical road at multiple `detaljnivå` levels. Empirically (Ålesund sample):
  `Vegtrase og kjørebane` 2821, `Kjørebane` 205, `Vegtrase` 102, `Kjørefelt`
  31. Simple roads appear once as *Vegtrase og kjørebane*; divided roads split
  into a *Vegtrase* centerline **plus** parallel *Kjørebane* carriageways (and
  finer *Kjørefelt* lanes). To get **each road once** we keep
  `{Vegtrase og kjørebane, Vegtrase}` and **drop** `Kjørebane` / `Kjørefelt`
  (the parallel-carriageway / lane duplicates). This also drops most detailed
  connector geometry.
- **typeVeg filter.** Vehicle roads kept by default:
  `{Enkel bilveg, Kanalisert veg, Rampe, Rundkjøring}`. Foot/bike geometry
  (`Gangveg, Gang- og sykkelveg, Gågate`) only with `--include-paths`.
  Sidewalks/stairs/crossings (`Fortau, Trapp, Gangfelt`) are always skipped as
  noise.
- **Category & width.** From `vegsystemreferanse.vegsystem.vegkategori`
  (E/R/F/K/P/S); ~20% of links lack it → fallback `K` (or `S` for paths).
  Default widths (m): E/R 9.0, F 7.0, K 5.5, P 4.0, S 3.0; paths 3.0. (NVDB
  object type 583 "Vegbredde" was not fetched — nice-to-have only.)
- **Merge & simplify.** Consecutive segments of the same `veglenkesekvensid`
  with coincident endpoints (≤0.1 m) are merged into one polyline; polylines
  are clipped to the region AABB (split where they exit/re-enter), dropped if
  <30 m, Douglas–Peucker simplified at 0.5 m, coords rounded to 2 decimals.

## Verification

`verify_pack.py` samples the DTM (bilinear) at each road vertex's local (x,z)
and compares to the road's own NVDB z-height. This is the **orientation
proof**: a wrong row order or z-sign makes the two disagree badly.

Measured (2026-07-15):

| pack | heightMin/Max (m) | roads / points | median \|Δh\| correct | median \|Δh\| z-flipped |
|------|-------------------|----------------|----------------------|-------------------------|
| trollstigen | 23.68 / 1787.96 | 10 / 614 | **0.11 m** | 436.7 m |
| alesund | 0.00 / 508.94 | 1097 / 9190 | **0.13 m** | 17.2 m |

The correct mapping matches the DTM to ~0.1 m median. Mean is higher
(Ålesund ~2.6 m) only because of **bridges and tunnels**, where the road's true
height legitimately differs from the DTM surface (e.g. an R-route bridge over a
sound where the DTM reads sea = 0). Those are correct, not orientation errors.

Pack sizes: `heights.f32` = 64,032,004 bytes (4001² × 4); `roads.json`
~16 KB (trollstigen) / ~300 KB (Ålesund); `region.json` ~0.4 KB.

## Notes for the C++ consumer

- `heights.f32` is **little-endian float32**, row-major, `index = iz*dim + ix`,
  **no flip** — `iz=0` is the north edge (max northing), `ix=0` is the west
  edge. Grid spacing is `worldSize/(dim-1)` (= `res`); nodes sit exactly on
  `x = ix*res − worldSize/2`, `z = iz*res − worldSize/2`.
- Road `points` are `[x, y, z]` in local world meters (y = terrain/road height).
  Road `y` matches the DTM to ~0.1 m except on bridges/tunnels — if you drape
  or Z-test roads against terrain, expect legitimate vertical offsets there.
- `region.json.originEasting/Northing` is the UTM33 (EPSG:25833) coordinate of
  the local origin, for georeferencing back to real-world / other layers.
