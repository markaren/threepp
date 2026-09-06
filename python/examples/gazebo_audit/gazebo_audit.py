"""The sensor-determinism audit, applied to Gazebo (Harmonic).

The paper's question, asked of another simulator with the same instrument:
does every sensor stream of one scripted scene replay to the bit across fresh
processes? This script starts the Gazebo server PAUSED, subscribes to the
streams audit_world.sdf publishes, steps the world a fixed number of 1 ms
iterations through the world-control service, and records every message by its
simulation stamp (a SHA-256 of the serialized protobuf, header included). Two
runs are then compared stamp by stamp:

    python3 gazebo_audit.py --iterations 3000 --seed 42 --out a.json
    python3 gazebo_audit.py --iterations 3000 --seed 42 --out b.json   # fresh process
    python3 gazebo_audit.py --compare a.json b.json                    # exit 0 = bit-identical

A stamp present in both runs with different bytes is a simulation difference.
A stamp present in one run only is a transport loss (gz-transport can drop a
message at start-up or shutdown) and is reported as such, never as a match and
never as a difference. The manifest also carries the whole-stream FNV row in
sensor_audit.py's format, so the two audits read alike.

Rows:
    camera        /camera                  gz.msgs.Image, 640x480 RGB, 30 Hz
    depth         /depth_camera            gz.msgs.Image, 32-bit float depth, 30 Hz
    lidar         /lidar                   gz.msgs.LaserScan, 720 x 16 ranges + intensities, 10 Hz
    lidar.points  /lidar/points            gz.msgs.PointCloudPacked
    imu           /imu                     gz.msgs.IMU, seeded Gaussian noise, 240 Hz
    poses         /world/audit/pose/info   every model pose (Gazebo throttles this
                                           stream by wall time, so its stamp SET varies
                                           between runs; the poses at common stamps
                                           are the physics row)
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time

FNV_OFFSET, FNV_PRIME, MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1
STREAMS = ("camera", "depth", "lidar", "lidar.points", "imu", "poses")


class Stream:
    def __init__(self):
        self.msgs = {}                                  # stamp_ns -> (sha256 hex, bytes)
        self.lock = threading.Lock()
        self.dups = 0

    def add(self, msg):
        data = msg.SerializeToString()
        st = msg.header.stamp
        key = int(st.sec) * 1_000_000_000 + int(st.nsec)
        h = hashlib.sha256(data).hexdigest()
        with self.lock:
            if key in self.msgs:
                self.dups += 1
            self.msgs[key] = (h, len(data))

    def last_stamp(self):
        with self.lock:
            return max(self.msgs) if self.msgs else -1

    def row(self):
        """sensor_audit.py's row: FNV chained over the stamp-ordered message hashes."""
        with self.lock:
            items = sorted(self.msgs.items())
        h, nbytes = FNV_OFFSET, 0
        for _, (hx, n) in items:
            for b in bytes.fromhex(hx):
                h ^= b
                h = (h * FNV_PRIME) & MASK
            nbytes += n
        return {"fnv": f"{h:016x}", "bytes": nbytes, "frames": len(items)}, [[k, v[0][:16]] for k, v in items]


def run(args):
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.laserscan_pb2 import LaserScan
    from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
    from gz.msgs10.imu_pb2 import IMU
    from gz.msgs10.pose_v_pb2 import Pose_V

    streams = {k: Stream() for k in STREAMS}
    node = Node()
    subs = [(Image, "/camera", "camera"), (Image, "/depth_camera", "depth"),
            (LaserScan, "/lidar", "lidar"), (PointCloudPacked, "/lidar/points", "lidar.points"),
            (IMU, "/imu", "imu"), (Pose_V, "/world/audit/pose/info", "poses")]
    cbs = []
    for typ, topic, key in subs:
        cb = (lambda k: (lambda msg: streams[k].add(msg)))(key)
        cbs.append(cb)
        if not node.subscribe(typ, topic, cb):
            print(f"warning: could not subscribe to {topic}", file=sys.stderr)

    # The server starts paused: nothing is published until every subscription
    # has found its publisher, so the first message of every stream is caught.
    cmd = [args.gz, "sim", "-s", "--seed", str(args.seed), args.world]
    if not args.no_headless:
        cmd.insert(3, "--headless-rendering")
    print("starting:", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(args.settle)

    # Step the world by exactly `iterations` steps, then it pauses itself.
    req = f"multi_step: {args.iterations}, pause: true"    # step exactly N, then pause
    step = subprocess.run([args.gz, "service", "-s", "/world/audit/control", "--reqtype", "gz.msgs.WorldControl",
                           "--reptype", "gz.msgs.Boolean", "--timeout", "5000", "--req", req],
                          capture_output=True, text=True)
    if "true" not in step.stdout.lower():
        print("world control:", step.stdout.strip(), step.stderr.strip()[-300:], file=sys.stderr)

    # Wait until the IMU (the fastest stream) has reached the last step, then a grace period.
    target_ns = int(args.iterations * 1_000_000)        # 1 ms steps
    deadline = time.perf_counter() + args.timeout
    while time.perf_counter() < deadline:
        if streams["imu"].last_stamp() >= target_ns - 5_000_000:
            break
        time.sleep(0.1)
    time.sleep(1.5)
    wall = time.perf_counter() - t0
    proc.terminate()
    try:
        out, err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill(); out, err = proc.communicate()

    version = ""
    try:
        version = subprocess.run([args.gz, "sim", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        pass
    rows, stamps = {}, {}
    for k, s in streams.items():
        r, st = s.row()
        rows[k] = r if r["frames"] else "absent"
        stamps[k] = st
    manifest = {
        "meta": {"simulator": "gazebo", "version": version, "world": os.path.basename(args.world),
                 "iterations": args.iterations, "step_s": 0.001, "seed": args.seed,
                 "headless_rendering": not args.no_headless, "render_engine": "ogre2",
                 "platform": platform.platform(), "python": platform.python_version(),
                 "wall_seconds": round(wall, 2), "duplicate_stamps": {k: s.dups for k, s in streams.items()}},
        "rows": rows, "stamps": stamps,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=1)
    for k, v in rows.items():
        span = f"  stamps {stamps[k][0][0] / 1e9:.3f}..{stamps[k][-1][0] / 1e9:.3f} s" if stamps[k] else ""
        print(f"  {k:13s} " + (v if isinstance(v, str) else f"{v['fnv']}  frames={v['frames']}{span}"))
    print(f"gazebo {version!r}, {wall:.1f} s wall -> {args.out}")
    for l in [l for l in (err or "").splitlines() if "Err" in l][-4:]:
        print("  gz:", l[:160])


def compare(pa, pb):
    """Stamp by stamp, inside the stepped window [0, iterations x step]. A stamp
    in one run only is reported as such (a transport loss, or for the pose
    stream Gazebo's wall-time throttle); the verdict is on the common stamps."""
    a, b = json.load(open(pa)), json.load(open(pb))
    print(f"A: {a['meta'].get('version')} seed {a['meta'].get('seed')} | B: {b['meta'].get('version')} seed {b['meta'].get('seed')}")
    win = int(min(a["meta"]["iterations"], b["meta"]["iterations"]) * a["meta"].get("step_s", 0.001) * 1e9)
    identical, differing = 0, 0
    for k in STREAMS:
        sa = {s: h for s, h in a.get("stamps", {}).get(k, []) if s <= win}
        sb = {s: h for s, h in b.get("stamps", {}).get(k, []) if s <= win}
        common = sorted(set(sa) & set(sb))
        only_a, only_b = len(set(sa) - set(sb)), len(set(sb) - set(sa))
        diff = [s for s in common if sa[s] != sb[s]]
        if not common:
            print(f"ABSENT   {k:13s} no common stamps (A {len(sa)}, B {len(sb)})")
            differing += 1
            continue
        note = "throttled by wall time" if k == "poses" else "transport"
        extra = f", only in one run ({note}): A {only_a}, B {only_b}" if (only_a or only_b) else ""
        if diff:
            print(f"DIFF     {k:13s} {len(diff)} of {len(common)} common stamps differ, first at {diff[0] / 1e9:.3f} s{extra}")
            differing += 1
        else:
            print(f"OK       {k:13s} {len(common)} common stamps bit-identical{extra}")
            identical += 1
    print(f"RESULT {identical} bit-identical, {differing} differing (per common stamp, window {win / 1e9:.3f} s)")
    return 0 if differing == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_world.sdf"))
    ap.add_argument("--iterations", type=int, default=3000, help="simulation steps at 1 ms (3000 = 3 s)")
    ap.add_argument("--seed", type=int, default=42, help="Gazebo's random seed (sensor noise)")
    ap.add_argument("--out", default="")
    ap.add_argument("--gz", default="gz")
    ap.add_argument("--no-headless", action="store_true", help="use the display instead of --headless-rendering")
    ap.add_argument("--settle", type=float, default=4.0, help="seconds for discovery before the first step")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for the last step")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    if not args.out:
        ap.error("--out is required for a run")
    run(args)


if __name__ == "__main__":
    main()
