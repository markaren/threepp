"""Load the Calico Tanks SOG scan, bake its surface, feed it to PhysX.

The WP1 smoke test for the splat bindings: SogLoader -> SplatCloud -> Z-up
transform -> Vulkan headless render -> splats.bake_surface -> a Mesh PhysX
accepts as a static trimesh. Prints bake time and triangle count and writes the
surface as an OBJ next to the asset (NOT into the repo -- it is megabytes).

    set PYTHONPATH=C:/dev/threepp/python
    py -3.14 python/examples/calico/bake_smoke.py [--level 2] [--pose-set interior]

FRAME, decided once (plans/calico-splat-demo.md):
  the scan is Y-DOWN like the COLMAP capture it came from, so the loader's
  coordinates need a half turn about X (rotation.x = pi) -- that is the frame
  every SuperSplat pose is quoted in. The Spot stack on top of it is Z-UP
  (gravity (0,0,-9.81), camera.up = (0,0,1)), so the cloud takes one MORE
  quarter turn about X: rotation.x = pi + pi/2. Bake AFTER that, because
  bake_surface emits WORLD-space vertices.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import threepp as tp

DEFAULT_ASSET = os.environ.get("THREEPP_CALICO_ASSET", "C:/dev/splats/calico_tanks")

ap = argparse.ArgumentParser()
ap.add_argument("--asset", default=DEFAULT_ASSET)
ap.add_argument("--level", type=int, default=2, help="SOG detail level (0 finest)")
ap.add_argument("--pose-set", choices=["orbit", "interior"], default="interior",
                help="the scan is a canyon seen from INSIDE, so interior is the default here")
ap.add_argument("--voxel-size", type=float, default=0.0, help="0 = derive from the cloud's robust fit")
ap.add_argument("--pose-count", type=int, default=26)
ap.add_argument("--obj", default=None, help="where to write the OBJ (default <asset>/bake_l<level>.obj)")
args = ap.parse_args()

W, H = 960, 600

print(f"asset: {args.asset}")
print(f"is_sog: {tp.SogLoader.is_sog(args.asset)}")
info = tp.SogLoader.describe(args.asset)
print(f"describe: lod_levels={info['lod_levels']} sh_degree={info['sh_degree']} counts={info['counts']}")

t0 = time.perf_counter()
data = tp.SogLoader.load(args.asset, level=args.level)
print(f"loaded level {args.level}: {data.count} splats in {time.perf_counter() - t0:.2f}s")

cloud = tp.SplatCloud(data)
# The half turn that undoes the COLMAP Y-down capture, PLUS a quarter turn so
# the world is Z-up like the rest of the Spot stack.
cloud.rotation.x = math.pi + math.pi / 2

scene = tp.Scene()
scene.add(cloud)

camera = tp.PerspectiveCamera(75, W / H, 0.1, 500)
# The plan's establishing pose, quoted in the post-flip Y-up frame, carried into
# Z-up: (x, y, z)_yup -> (x, -z, y)_zup.
camera.up.set(0, 0, 1)
camera.position.set(-6.0, -1.5, 3.6)
camera.look_at(-14.0, -1.7, 1.8)

canvas = tp.Canvas("calico-bake", width=W, height=H, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)

cloud.update(camera)
t0 = time.perf_counter()
renderer.render(scene, camera)
print(f"first render: {time.perf_counter() - t0:.2f}s (includes shader compile)")

pose_set = tp.SplatPoseSet.Interior if args.pose_set == "interior" else tp.SplatPoseSet.Orbit
t0 = time.perf_counter()
surface = tp.bake_surface(renderer, cloud, pose_set=pose_set,
                          voxel_size=args.voxel_size, pose_count=args.pose_count)
bake_s = time.perf_counter() - t0

st = surface.stats
print(f"bake ({args.pose_set}): {surface.triangle_count} triangles, "
      f"{surface.vertex_count} vertices in {bake_s:.2f}s")
print(f"  poses={st['poses']} voxel={st['voxel_size']:.4f} truncation={st['truncation']:.4f} "
      f"max_depth={st['max_depth']:.2f}")
print(f"  depth_samples={st['depth_samples']} observed_voxels={st['observed_voxels']} "
      f"blocks={st['blocks']} refused={st['refused_blocks']}")
print(f"  beyond_centre={st['beyond_centre_samples']} components={st['components']} "
      f"culled={st['culled_components']}")
print(f"  render={st['render_ms']:.0f}ms fuse={st['fuse_ms']:.0f}ms mesh={st['mesh_ms']:.0f}ms")
print(f"  aabb {tuple(round(v, 2) for v in (st['aabb_min'].x, st['aabb_min'].y, st['aabb_min'].z))}"
      f" .. {tuple(round(v, 2) for v in (st['aabb_max'].x, st['aabb_max'].y, st['aabb_max'].z))}")

if surface.triangle_count == 0:
    print("EMPTY BAKE -- see stats above (observed_voxels 0 = no voxel met weight_floor)")
    sys.exit(1)

# The sensor mesh: an ordinary Mesh over the same triangles, on the sensor-only
# layer. It is what the scene adds at the ROOT, and it is also what PhysX takes.
sensor = tp.make_sensor_mesh(surface)
scene.add(sensor)
renderer.set_sensor_only_surfaces(True)
print(f"sensor mesh: {sensor.name} sensor_only_surfaces={renderer.sensor_only_surfaces}")

world = tp.PhysxWorld()
world.set_gravity(tp.Vector3(0, 0, -9.81))
body = world.add_static_trimesh(sensor)
assert body is not None
print("PhysX accepted the sensor mesh as a static trimesh")

# ... and the hand route, for a script that wants its own material/visibility:
mesh = tp.Mesh(surface.to_geometry(), tp.MeshStandardMaterial())
body2 = world.add_static_trimesh(mesh)
assert body2 is not None
print("PhysX accepted surface.to_geometry() -> tp.Mesh too")

obj_path = args.obj or os.path.join(args.asset, f"bake_l{args.level}.obj")
pos, idx = surface.positions, surface.indices
t0 = time.perf_counter()
with open(obj_path, "w") as f:
    f.write(f"# threepp splats.bake_surface, {surface.triangle_count} triangles\n")
    for p in pos:
        f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    for t in idx:
        f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
print(f"wrote {obj_path} in {time.perf_counter() - t0:.1f}s")
