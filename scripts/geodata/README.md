# geodata — Norwegian region packs for threepp

`fetch_norway_terrain.py` builds a **region pack** from live Norwegian open
geodata: national DTM elevation (Kartverket) + the road network (NVDB) +
optionally buildings (OSM footprints, lidar-measured heights). The C++
engine consumes the pack; this tool is the preprocessor.

## Install

Active interpreter: Python 3.12 (`py -3.12`). Dependencies:

```
pip install requests numpy tifffile pyproj
# optional, only for a PNG (not TIF) preview:
pip install matplotlib
# optional, only for --texture:
pip install pillow
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
| `--out` | `<repo>/geodata` | output root |
| `--include-paths` | off | also emit foot/bike paths (Gangveg, Gang- og sykkelveg, Gågate) |
| `--no-roads` | off | skip the NVDB fetch |
| `--buildings` | off | fetch OSM building footprints + DOM nDSM heights |
| `--texture` | off | fetch a basemap texture from the open Kartverket raster WMS: `topo` (colour) or `topograatone` (greyscale) |
| `--texture-res` | 2 | texture resolution, m/px |
| `--preview` | off | render `preview.png` (hillshade + roads + building outlines) |

Output lands in `<out>/<name>/` as `region.json`, `heights.f32`, `roads.json`
(+ `buildings.json` with `--buildings`, `texture.png` with `--texture`,
`preview.png` with `--preview`). The
output format is the **frozen C++ contract** — see the tool docstring / the
spec; do not change it here.

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

### Basemap texture — Kartverket topo raster (WMS, `--texture`)

`https://wms.geonorge.no/skwms1/wms.topo` (layer `topo`, colour) /
`.../wms.topograatone` (layer `topograatone`, greyscale), **WMS 1.3.0
GetMap** in EPSG:25833, PNG. License CC BY 4.0, © Kartverket. Verified live
2026-08-25.

Quirks / decisions:
- **bbox axis order is EASTING,NORTHING** even in WMS 1.3.0 (verified: the
  E,N form returns map content on both 1.3.0 and 1.1.1; a swapped N,E bbox
  returns a blank tile).
- Written as `texture.png`, square, `round(size / texture-res)` px per side.
  **Row 0 = north edge** (WMS images are north-up, top-down), matching the
  heights convention — `u = (x + size/2)/size`, `v = (z + size/2)/size`
  drapes the image straight onto the terrain. Pixel-is-area over the exact
  region AABB (no node alignment: textures are sampled, not gridded).
- Tiled at ≤4096 px per dimension (4096 verified against the server), cut on
  pixel boundaries so tiles stitch exactly. Measured (Ålesund 2 km, 2×2
  forced tiling vs one request): base cartography is **pixel-identical**;
  only text labels (place names, depth soundings, contour labels) move —
  the WMS places labels per viewport. At the default 8000 m / 2 m/px the
  texture is 4000 px = a single request, so no seams at all; above 4096 px
  expect label clipping at internal tile boundaries.
- The C++ loader does not consume `texture.png` yet — the terrain albedo is
  the procedural Norwegian splat (GeoTerrain.hpp). The file is for external
  consumers and future drape support; `region.json` gains optional
  `texture` / `textureLayer` keys when present.

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

### Buildings — OSM footprints + Kartverket DOM heights (`--buildings`)

Footprints from **Overpass** (`way["building"]` + `relation["building"]` with
`out geom;`, WGS84 bbox over the region corners). License ODbL,
© OpenStreetMap contributors. Norwegian OSM buildings are largely a cadastre
(Matrikkelen) import, so coverage is good: central Ålesund ≈ complete.

Heights are **measured, not guessed**: the Kartverket national **DOM**
(surface model, same WCS family/request form as the DTM — endpoint
`wcs.hoyde-dom-nhm-25833`, coverage `nhm_dom_topo_25833`, verified live
2026-07-17) is fetched over the node window covering the footprints, and
`nDSM = DOM − DTM`. Per building, height = p90 of the nDSM inside the
footprint. Priority: OSM `height` tag → nDSM (≥3 nodes inside, p90 ≥ 2 m,
else the lidar predates the building) → `building:levels` × 3 m → per-type
default. Measured mix (Ålesund 8 km): **7869 ndsm / 29 tag / 4 levels /
363 default** of 8265.

Quirks / decisions:
- Multipolygon relations: member ways are chained into rings by shared
  endpoints; multiple outer rings become separate entries (`r<id>.<n>`) with
  holes assigned by point-in-ring containment.
- Kept if the footprint **centroid** is inside the region AABB and area
  ≥ 10 m²; outer rings normalized to **positive shoelace area in (x,z)**,
  holes negative; rings stored OPEN (first point not repeated).
- Contract: **roof top = groundMin + height** (`groundMin`/`groundMax` = DTM
  min/max under the footprint). For tag/levels/default sources the ground
  relief is folded into `height`; the nDSM p90 already measures from
  near-groundMin on slopes.
- Heights clamped to [2.5, 150] m.
- Real OSM appearance tags pass through when present (`colour`, `roofColour`,
  `roofShape` from `building:colour` / `roof:colour` / `roof:shape`) — rare
  (~0.1% in Ålesund) but they override the consumer's hashed palette.

## Verification

`verify_pack.py` samples the DTM (bilinear) at each road vertex's local (x,z)
and compares to the road's own NVDB z-height. This is the **orientation
proof**: a wrong row order or z-sign makes the two disagree badly. With
buildings present it repeats the proof against each building's stored
`groundMin/Max` at the footprint centroid (measured 2026-07-17: Ålesund
median |Δg| **0.12 m** as-is vs 18.05 m z-flipped; trollstigen 0.07 m vs
376.95 m) and sanity-checks bounds / height stats.

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
- `buildings.json` entries extrude straight up: walls from `groundMin`
  (sink slightly below for slope embedding) to the flat roof at
  `groundMin + height`. `GeoTerrainPack` parses the file when present;
  `terrain::buildGeoBuildingMeshes` (GeoBuildings.hpp) turns it into batched
  vertex-coloured chunk meshes with procedural facades (FacadeTexture.hpp):
  wall UVs snapped to (window-bay, floor) units so window grids always come
  out complete, glass panes glossy for env/sun reflections, utility types
  windowless. `GeoBuildingsOptions::seed` re-rolls all hashed facades
  (domain randomization); real `colour`/`roofColour` tags override the hash.
