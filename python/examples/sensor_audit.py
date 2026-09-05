"""Sensor replay audit, in Python, for any machine that can `pip install threepp`.

The C++ harnesses (examples/vulkan/vulkan_aov_audit, vulkan_lidar_audit --sonar,
examples/extras/sensors/replay_audit) are the paper's instruments on the
development machine. This is the same experiment written against the Python
package so it can run where the wheel runs: a Colab T4, a CI box, a laptop
with a different GPU. It builds one scripted scene, drives every clock from
frame index, reads every sensor for N frames, and folds each stream into one
64-bit FNV-1a hash. Two runs are compared row by row:

    python sensor_audit.py --frames 120 --out a.json
    python sensor_audit.py --frames 120 --out b.json      # fresh process
    python sensor_audit.py --compare a.json b.json        # exit 0 = bit-identical

Rows (each a chained hash over all frames):

    prop.imu / prop.contact   proprioceptive streams from PhysX (CPU), IMU with its
                              seeded MEMS noise ON, contact incl. force
    aov.depth / aov.normals / aov.ids / aov.motion / aov.albedo
                              raster G-buffer readbacks, native dtypes
    rgb                       the rendered frame after the temporal resolve
    lidar                     path-traced VLP-16 returns (position, distance,
                              intensity, instance id, return no), 10 Hz
    sonar                     imaging-sonar echogram (SonarSensor when the wheel
                              has it, otherwise the same fan traced through
                              scan_lidar and folded in numpy: max per range bin)

Rows a wheel cannot produce are reported as "absent", never as a match. The
manifest also records GPU, driver, platform and package version, so a
cross-machine compare says what it is comparing.

The scene mirrors the C++ audits: a ground box, a translating+rotating mover,
a spinner, a static pillar, and a CPU-deformed grid that exercises the
per-frame dynamic-geometry path. Every motion is a function of frame index.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time

import numpy as np

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = (1 << 64) - 1


class Fnv:
    """FNV-1a 64, chained across calls: order is part of the contract."""

    def __init__(self):
        self.h = FNV_OFFSET
        self.bytes = 0
        self.frames = 0

    def update(self, data: bytes):
        # Python-level FNV over megabytes per frame would take minutes; use
        # the same chaining semantics on a SHA-256 digest of the bytes instead,
        # folding that digest with FNV so the manifest stays a 64-bit hex row.
        d = hashlib.sha256(data).digest()
        h = self.h
        for b in d:
            h ^= b
            h = (h * FNV_PRIME) & MASK
        self.h = h
        self.bytes += len(data)
        self.frames += 1

    def row(self):
        return {"fnv": f"{self.h:016x}", "bytes": self.bytes, "frames": self.frames}


def arr_bytes(a) -> bytes:
    return np.ascontiguousarray(a).tobytes()


# --------------------------------------------------------------------------
# scene (mirrors examples/vulkan/vulkan_aov_audit.cpp)
# --------------------------------------------------------------------------
def build_scene(tp, renderer, aspect):
    scene = tp.Scene()
    scene.background = tp.Color(0x304050)
    sun = tp.DirectionalLight(tp.Color(0xFFFFFF), 3.0)
    sun.position.set(20, 30, 15)
    scene.add(sun)
    camera = tp.PerspectiveCamera(60, aspect, 0.1, 100)
    camera.position.set(0, 3, 9)
    camera.look_at(0, 1, 0)

    def mesh(geom, color, iid, cid):
        mat = tp.MeshStandardMaterial()
        mat.color = tp.Color(color)
        m = tp.Mesh(geom, mat)
        scene.add(m)
        renderer.set_instance_id(m, iid)
        renderer.set_class_id(m, cid)
        return m

    ground = mesh(tp.BoxGeometry(30, 0.5, 30), 0x556B45, 1, 1)
    ground.position.y = -0.25
    mover = mesh(tp.BoxGeometry(1, 1, 1), 0xC8783C, 2, 2)
    spinner = mesh(tp.SphereGeometry(0.7, 32, 16), 0x3C78C8, 3, 2)
    spinner.position.set(-2.5, 1, 0)
    pillar = mesh(tp.BoxGeometry(0.8, 4, 0.8), 0x808890, 4, 3)
    pillar.position.set(2.5, 2, -1)
    waver = mesh(tp.PlaneGeometry(4, 4, 20, 20), 0xB03A48, 5, 4)
    waver.rotation.x = -math.pi / 2
    waver.position.set(-0.5, 0.6, 3.0)
    rest = np.asarray(waver.geometry.get_attribute("position"), dtype=np.float32).reshape(-1, 3).copy()

    def deform(t):
        x, y = rest[:, 0], rest[:, 1]
        ax = 2.0 * x + np.float32(t) * np.float32(3.0)
        ay = 2.0 * y + np.float32(t) * np.float32(2.0)
        z = np.float32(0.25) * np.sin(ax) * np.cos(ay)
        dzdx = np.float32(0.5) * np.cos(ax) * np.cos(ay)
        dzdy = -np.float32(0.5) * np.sin(ax) * np.sin(ay)
        inv = 1.0 / np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
        pos = rest.copy()
        pos[:, 2] = z
        nrm = np.stack([-dzdx * inv, -dzdy * inv, inv], axis=1).astype(np.float32)
        waver.geometry.update_attribute("position", pos.astype(np.float32))
        waver.geometry.update_attribute("normal", nrm)

    def step(t):
        mover.position.set(2.0 * math.sin(t * 1.3), 1.0, 1.5 * math.cos(t * 0.9))
        mover.rotation.y = t * 1.7
        spinner.rotation.x = t * 2.3
        deform(t)

    return scene, camera, step


# --------------------------------------------------------------------------
# proprioceptive: PhysX CPU, a tumbling stack with an IMU and a contact sensor
# --------------------------------------------------------------------------
def run_proprioceptive(tp, seconds, seed):
    rows = {}
    if not getattr(tp, "HAS_PHYSX", False):
        return {"prop.imu": "absent", "prop.contact": "absent"}
    dt = 1.0 / 240.0
    world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), fixed_timestep=dt)
    floor = tp.Mesh(tp.BoxGeometry(40, 1, 40), tp.MeshStandardMaterial())
    floor.position.set(0, -0.5, 0)
    world.add_static(floor)
    boxes = []
    for i in range(6):
        b = tp.Mesh(tp.BoxGeometry(1, 0.6, 0.8), tp.MeshStandardMaterial())
        b.position.set(0.15 * i, 1.0 + 1.2 * i, 0.1 * (i % 2))
        b.rotation.set(0.1 * i, 0.3 * i, 0.05 * i)
        world.add(b, density=300)
        boxes.append(b)
    mount = tp.Group()
    mount.position.set(0.3, 0.2, 0.1)
    boxes[0].add(mount)
    imu = tp.Imu(mount)
    # Seeded MEMS noise ON: replaying the noise is part of the contract.
    g = imu.gyro_noise
    g.seed = seed
    imu.gyro_noise = g
    a = imu.accel_noise
    a.seed = seed ^ 0x9E3779B97F4A7C15
    imu.accel_noise = a
    world.register_sensor(imu)
    contact = tp.ContactSensor(boxes[0])
    world.register_sensor(contact)

    h_imu, h_con = Fnv(), Fnv()
    steps = int(seconds / dt)
    for _ in range(steps):
        world.step(dt)
        for s in imu.drain():
            h_imu.update(struct.pack("<7d", s.t, s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z,
                                     s.linear_acceleration.x, s.linear_acceleration.y, s.linear_acceleration.z))
        for s in contact.drain():
            h_con.update(struct.pack("<d?3dI", s.t, s.in_contact, s.force.x, s.force.y, s.force.z, s.observed_points))
    rows["prop.imu"] = h_imu.row()
    rows["prop.contact"] = h_con.row()
    return rows


# --------------------------------------------------------------------------
# sonar fold in numpy (the wheel may predate SonarSensor)
# --------------------------------------------------------------------------
def sonar_fan(fov=130.0, beams=256, aperture=20.0, samples=16):
    b = np.arange(beams)
    s = np.arange(samples)
    az = np.deg2rad(-0.5 * fov + fov * (b + 0.5) / beams).astype(np.float32)
    el = np.deg2rad(-0.5 * aperture + aperture * (s + 0.5) / samples).astype(np.float32)
    az = np.repeat(az, samples)
    el = np.tile(el, beams)
    d = np.stack([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], axis=1)
    return d.astype(np.float32)


def sonar_fold(returns, origin, beams, samples, bins, max_range, atten=0.05, floor=0.35):
    img = np.zeros((beams, bins), np.float32)
    rno = returns["return_no"]
    hit = rno > 0
    idx = np.nonzero(hit)[0]
    if idx.size == 0:
        return img
    pos = returns["position"][idx]
    nrm = returns["normal"][idx]
    dist = returns["distance"][idx]
    d = pos - origin
    ln = np.linalg.norm(d, axis=1)
    cos_inc = np.where(ln > 1e-6, np.abs(np.einsum("ij,ij->i", nrm, d) / np.maximum(ln, 1e-6)), 1.0)
    strength = (floor + (1 - floor) * cos_inc) * np.exp(-2.0 * atten * dist)
    ok = (dist > 0) & (dist <= max_range)
    beam = (idx // samples)
    bn = np.minimum(bins - 1, (dist / max_range * bins).astype(np.int64))
    np.maximum.at(img, (beam[ok], bn[ok]), strength[ok].astype(np.float32))
    return img


# --------------------------------------------------------------------------
def run(args):
    import threepp as tp

    manifest = {
        "meta": {
            "threepp": getattr(tp, "__version__", "?"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "frames": args.frames,
            "scan_every": args.scan_every,
            "seconds_prop": args.seconds,
            "seed": f"0x{args.seed:x}",
            "size": [args.width, args.height],
        },
        "rows": {},
    }
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        manifest["meta"]["gpu"] = q.stdout.strip() if q.returncode == 0 else "?"
    except Exception:
        manifest["meta"]["gpu"] = "?"

    manifest["rows"].update(run_proprioceptive(tp, args.seconds, args.seed))

    if not (getattr(tp, "HAS_VULKAN", False) and tp.vulkan_available()):
        for k in ("aov.depth", "aov.normals", "aov.ids", "aov.motion", "aov.albedo", "rgb", "lidar", "sonar"):
            manifest["rows"][k] = "absent"
        return manifest

    canvas = tp.Canvas("sensor_audit", args.width, args.height, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    # Pinned, as in the C++ audits: nothing that adapts from image statistics
    # or from a third-party temporal black box.
    renderer.auto_exposure = False
    pins = {"auto_exposure": False}
    for name in ("fsr", "dlss"):
        if hasattr(renderer, name):
            try:
                setattr(renderer, name, False)
                pins[name] = False
            except Exception as e:  # pragma: no cover
                pins[name] = f"error: {e}"
        else:
            pins[name] = "no binding"
    pins["sim_time"] = hasattr(renderer, "sim_time")
    manifest["meta"]["pins"] = pins

    scene, camera, step = build_scene(tp, renderer, args.width / args.height)

    lidar = tp.PathTracedLidarSensor(tp.LidarModel.vlp16(), 25.0)
    lidar.position.set(0, 1.5, 0)
    lidar.params.detector_threshold = 0.005
    lidar.params.reference_range = 5.0
    scene.add(lidar)

    sonar = None
    fan = None
    if hasattr(tp, "SonarSensor"):
        sonar = tp.SonarSensor(tp.SonarModel.wide130())
        sonar.position.set(0, 1.5, 0)
        scene.add(sonar)
        manifest["meta"]["sonar"] = "SonarSensor"
    elif hasattr(renderer, "scan_lidar"):
        fan = sonar_fan()
        manifest["meta"]["sonar"] = "numpy fold over scan_lidar"
    else:
        manifest["meta"]["sonar"] = "absent (wheel predates scan_lidar)"

    rows = {k: Fnv() for k in ("aov.depth", "aov.normals", "aov.ids", "aov.motion", "aov.albedo", "rgb", "lidar", "sonar")}
    dt = 1.0 / 60.0
    lidar_hits = 0
    sonar_echoes = 0
    t0 = time.time()
    for f in range(args.frames):
        t = f * dt
        if pins["sim_time"]:
            renderer.sim_time = t
        lidar.sim_time = t
        if sonar is not None:
            sonar.sim_time = t
        step(t)
        lidar.rotation.y = t * 0.35
        if sonar is not None:
            sonar.rotation.y = t * 0.35

        aovs = renderer.read_aovs_typed(scene, camera, ["rgb", "depth", "normals", "instance_ids", "motion", "albedo"])
        rows["aov.depth"].update(arr_bytes(aovs["depth"]))
        rows["aov.normals"].update(arr_bytes(aovs["normals"]))
        rows["aov.ids"].update(arr_bytes(aovs["instance_ids"]))
        rows["aov.motion"].update(arr_bytes(aovs["motion"]))
        rows["aov.albedo"].update(arr_bytes(aovs["albedo"]))
        rows["rgb"].update(arr_bytes(aovs["rgb"]))

        if f % args.scan_every != 0:
            continue
        r = lidar.scan(renderer)
        lidar_hits += int((r["return_no"] > 0).sum())
        rows["lidar"].update(b"".join(arr_bytes(r[k]) for k in ("position", "distance", "intensity", "instance_id", "return_no")))

        if sonar is not None:
            img = sonar.scan(renderer).intensity
        elif fan is None:
            continue
        else:
            # The same fan, aimed by the same yaw, traced as a beam table.
            c, s_ = math.cos(t * 0.35), math.sin(t * 0.35)
            rot = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]], np.float32)
            dirs = fan @ rot.T
            origin = np.array([0, 1.5, 0], np.float32)
            p = tp.LidarParams()
            p.max_range = 20.0
            p.detector_threshold = 0.0
            rr = renderer.scan_lidar(np.repeat(origin[None], len(dirs), 0), dirs, p)
            img = sonar_fold(rr, origin, 256, 16, 512, 20.0)
        sonar_echoes += int((img > 0).sum())
        rows["sonar"].update(arr_bytes(img))
    manifest["meta"]["wall_seconds"] = round(time.time() - t0, 2)
    manifest["meta"]["lidar_hits"] = lidar_hits
    manifest["meta"]["sonar_echoes"] = sonar_echoes
    for k, v in rows.items():
        manifest["rows"][k] = v.row() if v.frames else "absent"
    return manifest


def compare(pa, pb):
    a = json.load(open(pa))
    b = json.load(open(pb))
    ma, mb = a["meta"], b["meta"]
    print(f"A: {ma.get('gpu')} | {ma.get('platform')} | threepp {ma.get('threepp')}")
    print(f"B: {mb.get('gpu')} | {mb.get('platform')} | threepp {mb.get('threepp')}")
    same_machine = ma.get("gpu") == mb.get("gpu") and ma.get("platform") == mb.get("platform")
    print("same machine" if same_machine else "CROSS-MACHINE compare")
    exact = absent = diff = 0
    for k in sorted(set(a["rows"]) | set(b["rows"])):
        ra, rb = a["rows"].get(k, "absent"), b["rows"].get(k, "absent")
        if ra == "absent" or rb == "absent":
            print(f"ABSENT   {k}")
            absent += 1
        elif ra["fnv"] == rb["fnv"] and ra["bytes"] == rb["bytes"] and ra["frames"] == rb["frames"]:
            print(f"OK       {k} frames={ra['frames']} bytes={ra['bytes']} fnv={ra['fnv']}")
            exact += 1
        else:
            print(f"DIFF     {k}\n  A: {ra}\n  B: {rb}")
            diff += 1
    print(f"RESULT {exact} bit-identical, {diff} differing, {absent} absent")
    return 0 if diff == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--scan-every", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0, help="proprioceptive run length")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x9E3779B97F4A7C15)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    m = run(args)
    text = json.dumps(m, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    for k, v in m["rows"].items():
        print(f"{k:14s} {v if isinstance(v, str) else v['fnv']}")
    print("meta:", {k: m["meta"][k] for k in ("gpu", "pins", "sonar", "wall_seconds") if k in m["meta"]})


if __name__ == "__main__":
    main()
