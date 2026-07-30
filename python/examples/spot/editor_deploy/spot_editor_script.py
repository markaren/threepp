"""Run a trained policy bundle on a simulated robot INSIDE the threepp editor.

Attach to a URDF robot that has Simulate ticked, press Play, and the policy drives it. Nothing here
is Spot-specific: everything the controller needs comes from the bundle's spec.json, so pointing
`bundle` at a different export runs a different robot.

Inspector parameters: bundle (folder), vx / vy / wz (the velocity command), settle (ticks of
default-pose hold before the policy takes over), log (CSV path, empty = off).

The editor's world is Y-UP while a robot URDF is typically Z-up, so the robot node carries a -90°
X rotation. The controller never assumes an up axis — it derives one from the world's gravity — so
the observation it assembles is numerically the same as the trainer's despite the different world.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import threepp
from threepp import editor

from policy_runner import PolicyBundle, PolicyController, Adapter


def _quat_to_R(q):
    """[qx,qy,qz,qw] -> body->world rotation matrix (same formula the trainer uses)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


class EditorAdapter(Adapter):
    """The editor's live articulation handle, seen as the runner's simulator interface.

    Ground is the world plane through the origin; a terrain-aware policy would need a scene
    height query here (the editor has no script-visible raycast yet), which is exactly why
    `height_at` is a method on the adapter rather than baked into the runner.
    """

    def __init__(self, art, gravity=(0.0, -1.0, 0.0), ground=0.0):
        self.art = art
        self._g = np.asarray(gravity, float)
        self._ground = float(ground)

    @property
    def joint_names(self):
        return list(self.art.joint_names)

    @property
    def joint_positions(self):
        return np.asarray(self.art.joint_positions, float)

    @property
    def joint_velocities(self):
        return np.asarray(self.art.joint_velocities, float)

    @property
    def root_position(self):
        p = self.art.root_position
        return np.array([p.x, p.y, p.z], float)

    @property
    def root_rotation_matrix(self):
        q = self.art.root_rotation
        return _quat_to_R([q.x, q.y, q.z, q.w])

    @property
    def root_linear_velocity(self):
        v = self.art.root_velocity
        return np.array([v.x, v.y, v.z], float)

    @property
    def root_angular_velocity(self):
        v = self.art.root_angular_velocity
        return np.array([v.x, v.y, v.z], float)

    @property
    def gravity_direction(self):
        return self._g

    def set_drive_targets(self, values):
        self.art.set_drive_targets(values)

    def height_at(self, point):
        return self._ground


def _parse_route(text):
    """"secs:vx,vy,wz|secs:vx,vy,wz" -> [(secs, (vx,vy,wz)), ...]. Empty text -> [].

    Segments are '|'-separated, NOT ';'-separated: scriptFields is itself a flat
    `key=value;...` string, so a ';' inside a value would be eaten by that format.
    """
    out = []
    for seg in (text or "").replace(";", "|").split("|"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            secs, cmd = seg.split(":")
            vx, vy, wz = (float(v) for v in cmd.split(","))
            out.append((float(secs), (vx, vy, wz)))
        except ValueError:
            print(f"[policy] route segment '{seg}' is not secs:vx,vy,wz - ignored")
    return out


class PolicyDrive:

    bundle = ""          # folder holding spec.json + policy.npz (default: ./bundle_spot_steps)
    vx = 0.8             # forward command (m/s, body frame) - used when `route` is empty
    vy = 0.0             # strafe command (m/s, +y = left)
    wz = 0.0             # yaw-rate command (rad/s)
    settle = 100         # control ticks of default-pose hold before the policy takes over
    log = ""             # CSV path for the trace (empty = no file)
    chase = ""           # name of a camera to trail behind the robot (empty = leave cameras alone)
    # Teleop. With keys on, the keyboard is authoritative and nothing held means STAND STILL
    # (which the policy was trained to do on command) — so the robot never wanders off on its
    # own. Turn it off and the `route` below drives instead.
    #   forward  UP / KP8 / W        back      DOWN / KP2 / S
    #   strafe L LEFT / KP4 / A      strafe R  RIGHT / KP6 / D
    #   turn L   N / KP7 / Q         turn R    M / KP9 / E
    keys = 1             # 1 = drive it with the keyboard, 0 = follow `route`
    route = ""           # "secs:vx,vy,wz|..."  used when keys=0; empty = hold (vx, vy, wz)

    def start(self, obj, scene):
        self.obj = obj
        self.scene = scene
        self.ctrl = None
        self.held = 0
        self.t = 0.0
        self.csv = None
        self.status = ""
        art = editor.articulation_from_object(obj)
        if art is None:
            print("[policy] no simulated articulation on this object - "
                  "tick Simulate in the Robot section, and make sure this is a URDF robot")
            return
        folder = self.bundle or os.path.join(_HERE, "bundle_spot_steps")
        if not os.path.isdir(folder):
            print(f"[policy] no bundle at {folder}")
            return
        b = PolicyBundle(folder)
        adapter = EditorAdapter(art)
        try:
            self.ctrl = PolicyController(b, adapter)
        except RuntimeError as e:
            print(f"[policy] {e}")
            return
        self.adapter = adapter
        self.bundle_obj = b
        self.plan = _parse_route(self.route)
        # Older editor builds have no input binding at all; a script that asks politely still
        # runs there, just uncommanded.
        self.keydown = getattr(editor, "is_key_down", None)
        self.cam = None
        if self.chase:
            self.cam = scene.get_object_by_name(self.chase)
            if self.cam is None:
                print(f"[policy] no camera named '{self.chase}' in the scene")
        # The plant this policy wants vs what the scene actually authored. A silent mismatch here
        # is the classic reason a policy that trained fine "doesn't work" after a port.
        want = b.spec.get("plant", {})
        got = self.obj.get_user_data("articulation") or ""
        lines = [
            f"bundle           {os.path.basename(folder)}",
            f"obs/act          {b.obs_dim} / {b.act_dim} at {1.0 / b.dt:.0f} Hz",
            f"editor DOF order {list(art.joint_names)}",
            f"policy order     {b.joints}",
            f"policy->sim map  {list(self.ctrl.p2s)}",
            f"command          vx={self.vx} vy={self.vy} wz={self.wz}, settle={self.settle} ticks",
            f"input            is_key_down {'bound' if self.keydown else 'MISSING (older build)'}"
            f", teleop {'on' if int(self.keys) else 'off'}"
            f", probe W={self.keydown('W') if self.keydown else 'n/a'}",
            f"plant wanted     gains={want.get('gains', {}).get('stiffness')}/"
            f"{want.get('gains', {}).get('damping')} physics_dt={want.get('physics_dt')} "
            f"iters={want.get('solver_position_iterations')}",
            f"scene authored   {got or '(no articulation entry)'}",
        ]
        for line in lines:
            print("[policy] " + line)
        # The editor routes script prints to its own console, so a headless run needs a file to
        # read the contract back from — this is the evidence that the mapping was by NAME.
        if self.log:
            with open(os.path.splitext(self.log)[0] + "_diag.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        if self.log:
            self.csv = open(self.log, "w", buffering=1)
            self.csv.write("tick,t,x,y,z,upright,height,vx_cmd,vx_meas\n")

    def teleop(self):
        """Keyboard command, or None when this build/session offers no input at all."""
        k = self.keydown
        if k is None:
            return None
        held = lambda *names: any(k(n) for n in names)
        vx = (1.2 if held("UP", "KP8", "W") else 0.0) - (0.8 if held("DOWN", "KP2", "S") else 0.0)
        vy = (0.6 if held("LEFT", "KP4", "A") else 0.0) - (0.6 if held("RIGHT", "KP6", "D") else 0.0)
        wz = (1.0 if held("N", "KP7", "Q") else 0.0) - (1.0 if held("M", "KP9", "E") else 0.0)
        return vx, vy, wz

    def command(self):
        """The velocity command for right now: the keyboard, the route's segment, or the fixed one."""
        if int(self.keys):
            cmd = self.teleop()
            # Nothing held is a real command (stand), not "no opinion" — a teleop robot that
            # keeps walking when you let go is a runaway.
            if cmd is not None:
                return cmd
        if not self.plan:
            return self.vx, self.vy, self.wz
        t = self.t
        for secs, cmd in self.plan:
            if t < secs:
                return cmd
            t -= secs
        return self.plan[-1][1]        # route exhausted: hold the last segment

    def follow(self):
        """Trail the chase camera behind the robot, in whatever world up this scene has."""
        R = self.adapter.root_rotation_matrix
        p = self.adapter.root_position
        up = -self.adapter.gravity_direction
        fwd = R[:, 0] - up * float(np.dot(R[:, 0], up))       # heading, flattened into the ground
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        want = p - fwd * 3.0 + up * 1.5
        here = np.array([self.cam.position.x, self.cam.position.y, self.cam.position.z])
        now = here + (want - here) * 0.12                     # lerp, so it trails rather than snaps
        self.cam.position.set(float(now[0]), float(now[1]), float(now[2]))
        at = p + fwd * 0.4 + up * 0.1
        self.cam.look_at(float(at[0]), float(at[1]), float(at[2]))

    def update(self, dt):
        if self.ctrl is None:
            return
        if self.cam is not None:
            self.follow()
        # Hold the authored pose first: the policy expects to start from a settled stand, and the
        # articulation spawns a few centimetres above the ground.
        if self.held < int(self.settle):
            self.held += 1
            self.ctrl.hold_default()
            return
        self.vx, self.vy, self.wz = self.command()
        cmd = np.array([self.vx, self.vy, self.wz], np.float32)
        n = self.ctrl.tick(dt, cmd)
        if n == 0:
            return
        self.t += n * self.bundle_obj.dt
        R = self.adapter.root_rotation_matrix
        p = self.adapter.root_position
        up = -self.adapter.gravity_direction
        upright = float(np.dot(R[:, 2], up))        # the URDF's body up vs the world's
        vel = self.adapter.root_linear_velocity
        vx_meas = float(np.dot(vel, R[:, 0]))       # forward speed in the body frame
        if self.csv:
            self.csv.write(f"{self.ctrl.ticks},{self.t:.3f},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},"
                           f"{upright:.4f},{float(np.dot(p, up)):.4f},{self.vx},{vx_meas:.4f}\n")
        if self.ctrl.ticks % 50 == 0:
            print(f"[policy] t={self.t:5.1f}s  pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})  "
                  f"height={float(np.dot(p, up)):.2f}  upright={upright:+.2f}  "
                  f"vx={vx_meas:+.2f}/{self.vx:.2f}")

    def stop(self):
        if self.csv:
            self.csv.close()
            self.csv = None
        if self.ctrl is not None:
            print(f"[policy] stopped after {self.ctrl.ticks} control ticks ({self.t:.1f}s)")
