# geodata — generated region packs (not tracked)

This folder holds **region packs**: real-world terrain + road data that the
Norway terrain/driving examples load at runtime. The packs are **not committed**
— each is ~64 MB of raw elevation data (`.gitignore` keeps only this README).
Generate them locally before running the examples.

## Generate

The preprocessor lives in [`scripts/geodata`](../scripts/geodata) and fetches
live Norwegian open data: national DTM elevation (Kartverket) + the road network
(NVDB) + optionally buildings (OSM footprints, heights measured from the
Kartverket DOM surface model). See
[`scripts/geodata/README.md`](../scripts/geodata/README.md) for full options.

```
pip install requests numpy tifffile pyproj      # matplotlib optional (PNG preview)

cd scripts/geodata
python fetch_norway_terrain.py --preset trollstigen --buildings   # the Fv63 hairpins (driving showcase)
python fetch_norway_terrain.py --preset aalesund --buildings      # coastal town (sea path, ~8k buildings)
```

Each run writes `geodata/<name>/` containing:

| file | contents |
|------|----------|
| `region.json` | metadata: origin (EPSG:25833), `worldSize`, `dim`, height range, sea level, attribution |
| `heights.f32`  | raw little-endian float32 DEM, `dim×dim`, row-major (`iz*dim+ix`) |
| `roads.json`   | road polylines in local world coords, with category + width |
| `buildings.json` | extruded-footprint buildings (only with `--buildings`) |
| `preview.png`  | hillshade + roads + building outlines (only with `--preview`) |

Coordinates are threepp-native: Y-up metres, terrain centred on the origin,
world x = east, z = −north.

## Use

The examples default to `geodata/trollstigen` (driving) / `geodata/aalesund`
(terrain viewer). Point them elsewhere with a path argument or the
`THREEPP_REGION_PACK` environment variable:

```
norway_terrain [<pack-dir>]      # viewer  (or env THREEPP_REGION_PACK)
```

## Attribution & licensing

Generated packs embed an attribution string. The source data is:

- **Elevation** — © Kartverket, [høydedata.no](https://hoydedata.no), CC BY 4.0
  (DTM for the terrain; DOM for measured building heights).
- **Roads** — © Statens vegvesen, [NVDB](https://nvdb.no), NLOD.
- **Buildings** — © [OpenStreetMap](https://www.openstreetmap.org/copyright)
  contributors, ODbL (only in packs fetched with `--buildings`).

Respect those licences when redistributing anything derived from a pack.
