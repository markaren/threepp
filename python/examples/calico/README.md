# Calico Tanks — Spot in a Gaussian-splat scan of a real place

A trail scan of Calico Tanks (Red Rock Canyon) loaded as Gaussian splats, with the
Spot policy from [`../spot/`](../spot) walking it: the splats are the picture, a baked
surface is the collider, and the depth sensor the policy reads sees that surface. Run
`fetch_calico_asset.py` first; everything else expects the scan in the asset cache.

| Script | What it is |
| --- | --- |
| `fetch_calico_asset.py` | Fetch the Calico Tanks trail scan the other scripts walk on. |
| `spot_calico.py` | The demo: Spot walks the scan, with the LOD budget, the SLAM pip and the film shot list wired together. |
| `calico_collider.py` | The waterproof collider for the scan, split off from the sensor mesh. |
| `calico_slam_pip.py` | WP3: the live SLAM surface and the depth picture-in-picture. |
| `calico_look.py` | WP4: the look — sky, sun, contact shadow. |
| `calico_film.py` | WP5: the shot list, the sightline test and the film's bookkeeping. |
| `trail_analysis.py` | WP0: what the scan's trail floor is actually shaped like. |
| `bake_smoke.py` | WP1 smoke test for the splat bindings: loader, Z-up, headless render, `bake_surface`, PhysX trimesh. |
| `look_harness.py` | Iterate on the look without the physics: the scan, a standing Spot, two poses. |
| `lod_bench.py` | A/B the splat LOD policies from the spawn: one level for the whole cloud against one per SSOG node. |
