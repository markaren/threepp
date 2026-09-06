"""The sensor-determinism audit, applied to Isaac Sim (4.5, the pip package).

The same question the paper asks of threepp and of Gazebo: does every sensor
stream of one scripted scene replay to the bit across fresh processes? One
standalone headless SimulationApp builds a scene (a ground plane, a static
pillar, a box dropped onto a ramp with an IMU on it, a dropped sphere, a panel
rotated kinematically each step), a camera on the RTX real-time path with
whatever anti-aliasing the shipped defaults use, an RTX lidar, and steps the
world a fixed number of times on the simulation clock, hashing every sensor
frame. Two runs are compared with sensor_audit.py --compare (same manifest
format), and the per-frame hashes are kept so the first differing frame can be
named.

    D:\isaac-venv\Scripts\python.exe isaac_audit.py --frames 120 --out a.json
    D:\isaac-venv\Scripts\python.exe isaac_audit.py --frames 120 --out b.json
    python sensor_audit.py --compare a.json b.json

Rows:
    rgb          the camera's RGBA frame, 640x480, the RTX real-time path as shipped
    rgb.noaa     the same camera with anti-aliasing and DLSS off (/rtx/post/aa/op = 0)
    depth        the camera's distance-to-image-plane annotator (float32)
    lidar        the RTX lidar's point cloud per scan (if the sensor extension loads)
    imu          the IMU on the dropped box (lin_acc, ang_vel, orientation)
    poses        every dynamic body's pose per step (physics truth)

Every clock is the simulation clock: physics at 240 Hz, rendering at 60 Hz,
one rendered frame per world step. Nothing reads wall time.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

FNV_OFFSET, FNV_PRIME, MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1


class Fnv:
    def __init__(self):
        self.h, self.bytes, self.frames, self.per_frame = FNV_OFFSET, 0, 0, []

    def update(self, data: bytes):
        d = hashlib.sha256(data).digest()
        h = self.h
        for b in d:
            h ^= b
            h = (h * FNV_PRIME) & MASK
        self.h = h
        self.bytes += len(data)
        self.frames += 1
        self.per_frame.append(d[:8].hex())

    def row(self):
        return {"fnv": f"{self.h:016x}", "bytes": self.bytes, "frames": self.frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--aa", default="default", choices=["default", "off"],
                    help="anti-aliasing of the rgb row: the shipped default, or off (/rtx/post/aa/op=0)")
    ap.add_argument("--sample", default="", metavar="NPZ",
                    help="also save frame --sample-frame's rgb, depth and lidar arrays, and the imu row, for an element-wise diff")
    ap.add_argument("--sample-frame", type=int, default=60)
    args = ap.parse_args()

    from isaacsim import SimulationApp
    # D3D12, not Vulkan: with driver 595.97 on this machine Kit 106.5's Vulkan
    # path takes an access violation on its first frame; the D3D12 path runs.
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height,
                         "renderer": "RayTracedLighting",
                         "extra_args": ["--/app/vulkan=false",
                                        # synchronous rendering and readback, set before the renderer starts
                                        "--/app/asyncRendering=false", "--/app/asyncRenderingLowLatency=false",
                                        "--/omni/replicator/asyncRendering=false",
                                        "--/app/renderer/waitIdle=true", "--/app/hydraEngine/waitIdle=true"]})
    import numpy as np
    import carb
    import omni
    from pxr import Gf, UsdGeom
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere, FixedCuboid, VisualCuboid
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from isaacsim.sensors.camera import Camera
        from isaacsim.sensors.physics import IMUSensor
    except ImportError:                               # the older namespaces
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, FixedCuboid, VisualCuboid
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.sensor import Camera, IMUSensor

    # Kit swallows stdout at shutdown; everything this script says goes to stderr, flushed.
    def say(*a):
        print(*a, file=sys.stderr, flush=True)

    settings = carb.settings.get_settings()
    # Synchronous rendering and readback: the annotator delivers the frame of
    # THIS step, not whichever frame the GPU finished last. Without these the
    # annotators repeat and skip frames with the host's pacing, and a comparison
    # would be measuring the readback, not the renderer.
    for key, val in (("/app/asyncRendering", False), ("/app/asyncRenderingLowLatency", False),
                     ("/omni/replicator/asyncRendering", False), ("/app/renderer/waitIdle", True),
                     ("/app/hydraEngine/waitIdle", True), ("/app/updateOrder/checkForHydraRenderComplete", 1000)):
        try:
            settings.set(key, val)
        except Exception:
            pass
    aa_before = settings.get("/rtx/post/aa/op")
    if args.aa == "off":
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/post/dlss/execMode", 0)
    render_mode = settings.get("/rtx/rendermode")
    rtx_keys = ["/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/directLighting/sampledLighting/enabled",
                "/rtx/indirectDiffuse/enabled", "/rtx/reflections/enabled", "/rtx/ambientOcclusion/enabled",
                "/rtx/post/dlss/enabled", "/rtx/post/taa/enabled", "/rtx/sceneDb/allowDuplicateAhsInvocation",
                "/app/renderer/resolution/width", "/app/renderer/resolution/height", "/rtx/resolution/width"]
    rtx_state = {k: settings.get(k) for k in rtx_keys}

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    pillar = FixedCuboid("/World/pillar", name="pillar", position=np.array([3.0, 1.5, 1.0]),
                         scale=np.array([0.4, 0.4, 2.0]), color=np.array([0.6, 0.6, 0.65]))
    ramp = FixedCuboid("/World/ramp", name="ramp", position=np.array([4.0, -2.0, 0.35]),
                       orientation=euler_angles_to_quat(np.array([0.0, 0.35, 0.0])),
                       scale=np.array([3.0, 1.2, 0.1]), color=np.array([0.5, 0.45, 0.4]))
    mover = DynamicCuboid("/World/mover", name="mover", position=np.array([5.1, -2.0, 1.0]),
                          orientation=euler_angles_to_quat(np.array([0.0, 0.35, 0.0])),
                          scale=np.array([0.3, 0.3, 0.3]), color=np.array([0.85, 0.15, 0.1]), mass=1.5)
    ball = DynamicSphere("/World/ball", name="ball", position=np.array([2.5, 0.5, 1.5]), radius=0.2,
                         color=np.array([0.15, 0.3, 0.8]), mass=0.8)
    panel = VisualCuboid("/World/panel", name="panel", position=np.array([2.0, -0.8, 1.0]),
                         scale=np.array([0.9, 0.05, 0.3]), color=np.array([0.9, 0.75, 0.2]))
    for o in (pillar, ramp, mover, ball, panel):
        world.scene.add(o)

    cam = Camera(prim_path="/World/rig/camera", position=np.array([0.0, 0.0, 1.2]),
                 orientation=euler_angles_to_quat(np.array([0.0, 0.15, 0.0])),
                 frequency=60, resolution=(args.width, args.height))
    imu = IMUSensor(prim_path="/World/mover/imu", name="imu", frequency=240,
                    translation=np.array([0.0, 0.0, 0.0]))
    world.scene.add(imu)

    lidar = None
    if not args.no_lidar:
        try:
            try:
                from isaacsim.sensors.rtx import LidarRtx
            except ImportError:
                from omni.isaac.sensor import LidarRtx
            lidar = LidarRtx(prim_path="/World/rig/lidar", name="lidar", position=np.array([0.0, 0.0, 1.5]),
                             config_file_name="Example_Rotary")
            world.scene.add(lidar)
        except Exception as e:
            print("lidar unavailable:", e, file=sys.stderr)
            lidar = None

    world.reset()
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    if lidar is not None:
        try:
            lidar.initialize()
            for fn in ("add_point_cloud_data_to_frame", "add_range_data_to_frame", "add_linear_depth_data_to_frame",
                       "add_intensities_data_to_frame"):
                if hasattr(lidar, fn):
                    try:
                        getattr(lidar, fn)()
                    except Exception as e:
                        say(f"lidar {fn} failed: {e}")
        except Exception as e:
            say("lidar init failed:", e)
            lidar = None
    for _ in range(10):                                # let the annotators attach
        world.step(render=True)
    if lidar is not None:
        try:
            fr = lidar.get_current_frame()
            say("lidar frame keys:", {k: (getattr(v, "shape", None) if hasattr(v, "shape") else type(v).__name__) for k, v in fr.items()})
        except Exception as e:
            say("lidar frame read failed:", e)
    world.reset()

    rgb_key = "rgb" if args.aa == "default" else "rgb.noaa"
    rows = {k: Fnv() for k in (rgb_key, "depth", "lidar", "imu", "poses")}
    bodies = [mover, ball]
    sample, imu_log = {}, []
    wall0 = time.perf_counter()
    for f in range(args.frames):
        panel.set_world_pose(orientation=euler_angles_to_quat(np.array([0.0, 0.0, 2.0 * f / 60.0])))
        world.step(render=True)
        frame = cam.get_current_frame()
        rgba = frame.get("rgba")
        if rgba is not None and np.asarray(rgba).size:
            rows[rgb_key].update(np.ascontiguousarray(np.asarray(rgba)).tobytes())
            if args.sample and f == args.sample_frame:
                sample["rgb"] = np.ascontiguousarray(np.asarray(rgba))
        dist = frame.get("distance_to_image_plane")
        if dist is not None and np.asarray(dist).size:
            rows["depth"].update(np.ascontiguousarray(np.asarray(dist, np.float32)).tobytes())
            if args.sample and f == args.sample_frame:
                sample["depth"] = np.ascontiguousarray(np.asarray(dist, np.float32))
        if lidar is not None:
            try:
                fr = lidar.get_current_frame()
                parts = []
                for key in ("point_cloud_data", "range", "linear_depth_data", "intensities_data"):
                    v = fr.get(key)
                    if v is not None and np.asarray(v).size:
                        parts.append(np.ascontiguousarray(np.asarray(v, np.float32)).tobytes())
                        if args.sample and f == args.sample_frame:
                            sample["lidar." + key] = np.ascontiguousarray(np.asarray(v, np.float32))
                if parts:
                    rows["lidar"].update(b"".join(parts))
            except Exception as e:
                if f == 0:
                    say("lidar read failed:", e)
        im = imu.get_current_frame()
        imu_vals = [np.asarray(im[k], np.float64).ravel() for k in ("lin_acc", "ang_vel", "orientation") if k in im]
        rows["imu"].update(b"".join(np.ascontiguousarray(v).tobytes() for v in imu_vals))
        if args.sample:
            imu_log.append(np.concatenate([[f / 60.0], *imu_vals]))
        rows["poses"].update(b"".join(np.ascontiguousarray(np.asarray(x, np.float64)).tobytes()
                                      for b in bodies for x in b.get_world_pose()))
    wall = time.perf_counter() - wall0

    manifest = {
        "meta": {"simulator": "isaacsim", "version": getattr(__import__("isaacsim"), "__version__", "4.5.0"),
                 "renderer": "RayTracedLighting", "rendermode": render_mode, "aa_op_default": aa_before,
                 "aa": args.aa, "rtx": rtx_state,
                 "frames": args.frames, "physics_dt": 1 / 240, "rendering_dt": 1 / 60,
                 "size": [args.width, args.height], "platform": platform.platform(),
                 "python": platform.python_version(), "wall_seconds": round(wall, 2)},
        "rows": {k: v.row() if v.frames else "absent" for k, v in rows.items()},
        "per_frame": {k: v.per_frame for k, v in rows.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=1)
    if args.sample:
        if imu_log:
            sample["imu"] = np.asarray(imu_log, np.float64)
        np.savez(args.sample, **sample)
        say("sample ->", args.sample, {k: tuple(v.shape) for k, v in sample.items()})
    for k, v in manifest["rows"].items():
        say(f"  {k:9s} " + (v if isinstance(v, str) else f"{v['fnv']}  frames={v['frames']}  bytes={v['bytes']}"))
    say(f"isaac: {args.frames} frames, aa={args.aa}, rtx={rtx_state}, {1e3 * wall / max(args.frames, 1):.1f} ms/f -> {args.out}")
    app.close()


if __name__ == "__main__":
    main()
