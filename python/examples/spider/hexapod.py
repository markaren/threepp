"""A procedural physics hexapod (6-legged spider) built on threepp's PhysX
articulation binding, with an open-loop CPG tripod gait.

Each leg has 2 motorized DOF:
  - coxa : revolute about the vertical (Y) axis  -> swings the leg fore/aft (propulsion)
  - femur: revolute about a horizontal axis       -> lifts/lowers the foot (swing/stance)

A statically-stable tripod gait (two alternating sets of 3 legs) turns a velocity
command (forward, turn) into per-joint drive targets — no training required.

Run headless to validate/tune:  python hexapod.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import threepp as tp

# Leg layout: index -> (fore/aft x, side sign). side -1 = left (-Z), +1 = right (+Z).
# Tripod groups alternate by parity: {0,2,4} move together, {1,3,5} a half-cycle later.
_FORE = (0.18, 0.0, -0.18)


def _orient_to(mesh, p_from, p_to):
    """Orient a Y-aligned mesh (capsule) so its local +Y points from p_from to p_to."""
    d = [p_to[i] - p_from[i] for i in range(3)]
    L = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    dn = [c / L for c in d]
    axis = (dn[2], 0.0, -dn[0])  # (0,1,0) x dn
    amag = math.sqrt(axis[0] ** 2 + axis[2] ** 2)
    if amag > 1e-6:
        ax = tp.Vector3(axis[0] / amag, 0.0, axis[2] / amag)
        mesh.quaternion.set_from_axis_angle(ax, math.acos(max(-1.0, min(1.0, dn[1]))))
    return L


class Hexapod:
    def __init__(self, world, position=(0.0, 0.35, 0.0),
                 body=(0.5, 0.10, 0.28), height=0.35,
                 coxa_len=0.10, foot_out=0.16,
                 coxa_gain=(400.0, 20.0, 120.0), femur_gain=(600.0, 30.0, 180.0),
                 gait_freq=1.6, coxa_amp=0.35, lift_amp=0.55,
                 coxa_sign=-1.0, femur_sign=1.0):
        self.world = world
        self.cmd = [0.0, 0.0]          # (forward [-1..1], turn [-1..1])
        self.psi = 0.0
        self.gait_freq = gait_freq
        self.coxa_amp = coxa_amp
        self.lift_amp = lift_amp

        bx, by, bz = body
        cx, cy, cz = position
        halfW = bz * 0.5
        ck, cd, cf = coxa_gain
        fk, fd, ff = femur_gain

        art = world.create_articulation(fixed_base=False, solver_position_iterations=12,
                                        disable_self_collision=True)
        chassis_mesh = tp.Mesh(tp.BoxGeometry(bx, by, bz), self._mat(0x9a6a3a))
        chassis_mesh.position.set(cx, cy, cz)
        self.chassis = art.add_link(chassis_mesh, density=350)
        self.meshes = [chassis_mesh]
        self.legs = []

        for s in (-1, 1):
            for fx in _FORE:
                A = (cx + fx, cy, cz + s * halfW)
                B = (cx + fx, cy, cz + s * (halfW + coxa_len))
                F = (cx + fx, cy - height, cz + s * (halfW + coxa_len + foot_out))

                coxa_mid = (cx + fx, cy, cz + s * (halfW + coxa_len * 0.5))
                coxa_mesh = tp.Mesh(tp.BoxGeometry(0.05, 0.05, coxa_len), self._mat(0x555555))
                coxa_mesh.position.set(*coxa_mid)
                coxa = art.add_link(coxa_mesh, parent=self.chassis, density=300,
                                    axis=(0, 1, 0), anchor=A,
                                    lower=-0.8, upper=0.8, stiffness=ck, damping=cd, max_force=cf)

                femur_mid = tuple((B[i] + F[i]) * 0.5 for i in range(3))
                femur_mesh = tp.Mesh(tp.CapsuleGeometry(0.035, math.dist(B, F)), self._mat(0x3aa05a))
                femur_mesh.position.set(*femur_mid)
                _orient_to(femur_mesh, B, F)
                femur = art.add_link(femur_mesh, parent=coxa, density=250,
                                     axis=(1, 0, 0), anchor=B,
                                     lower=-1.3, upper=1.3, stiffness=fk, damping=fd, max_force=ff)

                self.meshes += [coxa_mesh, femur_mesh]
                # Both signs flip with the leg's ±Z side: a +rotation about the
                # shared world axis moves left/right feet in opposite directions.
                side_flip = 1.0 if s < 0 else -1.0
                self.legs.append({"coxa": coxa, "femur": femur, "side": s,
                                  "coxa_sign": coxa_sign * side_flip,
                                  "femur_sign": femur_sign * side_flip})
        art.finalize()
        self.art = art

    @staticmethod
    def _mat(color):
        m = tp.MeshStandardMaterial()
        m.color = color
        m.roughness = 0.6
        return m

    def add_to_scene(self, scene):
        for m in self.meshes:
            scene.add(m)

    def set_command(self, forward, turn):
        self.cmd[0] = max(-1.0, min(1.0, forward))
        self.cmd[1] = max(-1.0, min(1.0, turn))

    def update(self, dt, residuals=None):
        """Advance the gait and set joint drive targets. `residuals` (optional, 12
        values: coxa,femur per leg) are added on top of the CPG targets — this is
        the action interface for residual RL (Stage 2)."""
        self.psi += self.gait_freq * 2.0 * math.pi * dt
        fwd, turn = self.cmd
        for i, leg in enumerate(self.legs):
            phase = self.psi + (i % 2) * math.pi
            # +turn = left (CCW): right legs stroke harder than left. (Sign chosen so
            # the conventional +turn maps to a left turn / positive yaw rate.)
            drive = max(-1.0, min(1.0, fwd - turn * (1.0 if leg["side"] < 0 else -1.0)))
            coxa_t = self.coxa_amp * drive * leg["coxa_sign"] * math.cos(phase)
            lift = self.lift_amp * max(0.0, -math.sin(phase))
            femur_t = leg["femur_sign"] * lift
            if residuals is not None:
                coxa_t += residuals[2 * i]
                femur_t += residuals[2 * i + 1]
            leg["coxa"].set_drive_target(coxa_t)
            leg["femur"].set_drive_target(femur_t)

    def reset(self, position=(0.0, 0.40, 0.0)):
        """Teleport back to `position`, upright, joints zeroed; restart the gait."""
        self.art.reset(tp.Vector3(*position))
        self.psi = 0.0
        self.cmd = [0.0, 0.0]

    def joint_states(self):
        """(positions, velocities) for all 12 joints — coxa,femur interleaved per leg."""
        pos, vel = [], []
        for leg in self.legs:
            pos.append(leg["coxa"].joint_position)
            vel.append(leg["coxa"].joint_velocity)
            pos.append(leg["femur"].joint_position)
            vel.append(leg["femur"].joint_velocity)
        return pos, vel

    @property
    def position(self):
        return self.chassis.position

    @property
    def up_y(self):
        # world Y component of the chassis local up axis — ~1.0 when level, <0 if flipped
        q = self.chassis.quaternion
        return 1.0 - 2.0 * (q.x * q.x + q.z * q.z)

    @property
    def yaw(self):
        # heading about world Y (radians)
        q = self.chassis.quaternion
        return math.atan2(2.0 * (q.w * q.y + q.x * q.z), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @property
    def forward(self):
        # World-space forward direction (chassis local +X rotated by its orientation).
        # Use this for a chase camera — it's handedness-correct, unlike rebuilding the
        # direction from a scalar yaw with cos/sin.
        f = tp.Vector3(1, 0, 0)
        f.apply_quaternion(self.chassis.quaternion)
        return f


def _probe():
    """Headless: stand for 1 s, then walk forward 3 s; print the chassis trajectory."""
    world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
    ground = tp.Mesh(tp.BoxGeometry(50, 1, 50), tp.MeshStandardMaterial())
    ground.position.y = -0.5
    world.add_static(ground)

    spider = Hexapod(world, position=(0, 0.40, 0))

    def sample(label):
        p = spider.position
        print("  %-7s x=%+.3f y=%+.3f z=%+.3f  up=%+.2f" % (label, p.x, p.y, p.z, spider.up_y))

    print("settle (no command):")
    for _ in range(90):
        spider.update(1 / 60)
        world.step(1 / 60)
    sample("stand")
    y_stand = spider.position.y

    print("walk forward (cmd=1,0):")
    spider.set_command(1.0, 0.0)
    p0 = (spider.position.x, spider.position.z)
    for k in range(240):
        spider.update(1 / 60)
        world.step(1 / 60)
        if k % 60 == 59:
            sample("t=%.1fs" % ((k + 1) / 60))
    dx = spider.position.x - p0[0]
    dist = math.dist((spider.position.x, spider.position.z), p0)
    print("forward: dx=%+.3f m, |disp|=%.3f m (%.2f m/s)   upright: %s"
          % (dx, dist, dist / 4.0, "yes" if spider.up_y > 0.5 else "NO (fell)"))

    print("turn in place (cmd=0,1):")
    spider.set_command(0.0, 1.0)
    yaw0 = spider.yaw
    for _ in range(240):
        spider.update(1 / 60)
        world.step(1 / 60)
    dyaw = math.degrees(spider.yaw - yaw0)
    print("turn: dyaw=%+.1f deg over 4 s (%.0f deg/s)   upright: %s"
          % (dyaw, dyaw / 4.0, "yes" if spider.up_y > 0.5 else "NO"))


if __name__ == "__main__":
    _probe()
