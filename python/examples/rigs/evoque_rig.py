"""
evoque_rig -- the Range Rover Evoque: the model as a shell on wheels that turn, and PhysX's
vehicle under it.

Three things, and nothing scene-specific:

  * ``Shell``   -- the model on a node whose +z is ahead, four wheel rigs to turn, the tail and
                   reverse lamps on materials of their own, a paint, and on request the head
                   lamps' glass as a skin that can glow. Without the model: a box on four
                   cylinders, so a scene still drives.
  * ``Car``     -- ``tp.PhysxVehicle`` with the Evoque tuning under a Shell: the pose after
                   each physics step, and the drawn pose blended between the last two steps.
  * the helm    -- ``keys()`` (W S A D or the arrows, SPACE brakes: the C++ demo's
                   speed-sensitive steering and its slew) and ``steer_for()`` (the steer
                   command that holds a path curvature).

The model is "2015 Land-Rover Range Rover Evoque Coupe" by Ddiaz Design (Sketchfab),
CC BY-NC-SA 4.0, from threepp-data (``model_path()`` finds it; THREEPP_DATA_DIR wins).

FRAMES
------
PhysX's vehicle: +Z ahead, +Y up, +X to the chassis' left; set_steer(+) turns toward +X. Its
origin is the chassis' middle, which rides ``ride_height(wheel_radius)`` over the road. Wheel
indices 0..3 are front (+x, -x), rear (+x, -x). The model labels its wheels from inside the
car, so its "FL" sits at +x, where PhysX puts wheel 0: TAGS is in PhysX's order.

A Shell's node has its origin on the road under the car's middle until a Car wears it: then
the node is at the chassis' middle and the hull hangs under it (``Shell.hang``), shifted so
the arches stand over PhysX's axles (the model's axles are 3 cm aft of its origin).

TYPICAL USE
-----------
::

    from evoque_rig import Car, Shell, keys, model_path

    shell = Shell(model_path())
    car = Car(world, position, rotation, shell)
    scene.add(shell.node)
    ...
    car.steer, car.throttle, car.brake = keys(canvas, car.veh.forward_speed, car.steer, dt)
    debt += dt
    while debt >= DT:                   # the physics at a fixed step
        debt -= DT
        car.control()
        world.step(DT)
        car.sync()
    car.draw(debt / DT)                 # the shell between the last two steps

A car that is only shown (traffic) is a Shell alone: place ``shell.node`` on the road and
``shell.roll(distance)``.
"""
import math
import os

import numpy as np

import threepp as tp
from demo_common import standard_material

EVOQUE = ("models", "gltf", "2015_land-rover_range_rover_evoque_coupe", "scene.gltf")
TAGS = ("WheelFL", "WheelFR", "WheelBL", "WheelBR")      # the model's wheels in PhysX's order
LENS = "glass_lights"                                    # the model's lamp glass (head and tail in one mesh)

# tp.PhysxVehicle's defaults are the Evoque tuning of the C++ demo (examples/projects/Vehicle):
# a 1.95 x 1.4 x 4.4 m chassis of 1500 kg, four driven wheels, 1500 N m of throttle torque.
WHEELBASE, TRACK, MAX_STEER = 2.66, 1.65, 0.6
# The C++ demo's tyre friction of 2.0 out-grips the geometry: a 1.65 m track under a CoM
# ~0.85 m up rolls at about 1 g, and 2.0 of mu reaches 2, so the car rolls before it slides.
# Measured at 1.1 (warp_mudsnow_drive.py): full lock at 50 km/h on clay peaks at 2.2 degrees of tilt.
TYRE_MU = 1.1
# The chassis' middle over the wheel's hub at rest: the suspension's attachment 0.4 under the
# middle, 0.3 of travel, less the rest jounce.
HUB_TO_CHASSIS = 0.596


def data_dir():
    """threepp_data's checkout. THREEPP_DATA_DIR wins; otherwise the usual places beside the
    repository (the checkout is commonly named with a hyphen)."""
    env = os.environ.get("THREEPP_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    for name in ("threepp-data", "threepp_data"):
        cand = os.path.join(os.path.dirname(repo), name)
        if os.path.isdir(cand):
            return cand
    return ""


def model_path():
    """The Evoque's .gltf in threepp-data, or None where it is not."""
    path = os.path.join(data_dir(), *EVOQUE)
    return path if os.path.isfile(path) else None


def ride_height(wheel_radius):
    """How far PhysX's chassis middle rides over the road on wheels of this radius."""
    return wheel_radius + HUB_TO_CHASSIS


def keys(canvas, forward_speed, steer, dt, slew=2.5):
    """The keyboard: (steer, throttle, brake) from W S A D or the arrows, SPACE brakes too.
    The steer is the C++ demo's: less lock the faster she goes, taken up at `slew` per second."""
    left = canvas.is_key_down("A") or canvas.is_key_down("LEFT")
    right = canvas.is_key_down("D") or canvas.is_key_down("RIGHT")
    want = ((1.0 if left else 0.0) - (1.0 if right else 0.0)) / (1.0 + abs(forward_speed) * 3.6 * 0.015)
    steer += (want - steer) * min(1.0, dt * slew)
    throttle = 1.0 if (canvas.is_key_down("W") or canvas.is_key_down("UP")) else 0.0
    brake = 1.0 if (canvas.is_key_down("S") or canvas.is_key_down("DOWN") or canvas.is_key_down("SPACE")) else 0.0
    return steer, throttle, brake


def steer_for(curvature):
    """The steer command (-1..1) that holds a path of this curvature (1/m, + toward the
    chassis' left): the bicycle model on the wheelbase, over the vehicle's full lock."""
    return float(np.clip(math.atan(WHEELBASE * curvature) / MAX_STEER, -1.0, 1.0))


def in_frame(o):
    """A mesh's vertices in the frame its world matrix leads to (N, 3)."""
    a = np.asarray(o.geometry.get_attribute("position"), float).reshape(-1, 3)
    m = np.array(o.matrix_world.elements(), float).reshape(4, 4).T
    return a @ m[:3, :3].T + m[:3, 3]


class Shell:
    """The Evoque's model on a node whose origin is on the road under the car's middle and whose
    +z is ahead; four wheel rigs to turn. A wheel's place is baked into its geometry, and that
    geometry is Z-up under the model's own turns: the hub is read where the model's nodes put the
    wheel, a rig stands there and the wheel hangs in it, turned as the model turns it.

    path     the model's .gltf (model_path()); None or a missing file gives a box on four cylinders
    paint    linear (r, g, b) for the body, in place of the model's white
    radius   the box's wheel radius (the model's is measured: self.radius)

    node, hull    the node a scene places, and under it what hangs from a chassis (see hang())
    rigs, hubs    the four wheel rigs (PhysX's order) and where the model puts their hubs
    tail, reverse the lamps' materials (the model shares one between them): brake() and
                  reversing() flare them, or a scene sets their look itself"""

    LAMP_REST, LAMP_ON = 1.0, 6.0      # the lamps' emissive intensity, and flared

    def __init__(self, path=None, paint=None, radius=0.4):
        self.node, self.hull = tp.Group(), tp.Group()     # the hull: what hangs under a chassis that rides above the road
        self.node.add(self.hull)
        self.rigs, self.hubs = [], []
        self.tail = self.reverse = self.glass = None
        self.lamps, self.head = [], []
        self._lens = []
        self._braking = self._reversing = False
        self.model = path is not None and os.path.isfile(path)
        if not self.model:
            self._box(radius)
            return
        body = tp.GLTFLoader().load(path).scene
        body.scale.set(100.0, 100.0, 100.0)               # the gltf bakes a 0.01 at its root
        self.hull.add(body)
        self.node.update_matrix_world(True)               # the node is still at the origin: world is the node's own frame
        radii = []
        meshes = []
        body.traverse(lambda o: meshes.append(o) if isinstance(o, tp.Mesh) else None)
        for tag in TAGS:
            group = [o for o in meshes if tag in o.name]
            a = np.concatenate([in_frame(o) for o in group])
            lo, hi = a.min(0), a.max(0)
            hub = 0.5 * (lo + hi)
            radii.append(0.5 * float(np.sort(hi - lo)[1]))
            rig = tp.Group()
            rig.position.set(*map(float, hub))
            self.node.add(rig)
            for o in group:
                c = o.clone()                             # before hiding: a clone takes `visible`
                o.visible = False
                c.visible = True
                p, q, s = o.get_world_position(), o.get_world_quaternion(), o.get_world_scale()
                c.position.set(p.x - float(hub[0]), p.y - float(hub[1]), p.z - float(hub[2]))
                c.quaternion.set(q.x, q.y, q.z, q.w)      # a clone has the mesh's own node alone: what stood over it is put back
                c.scale.set(s.x, s.y, s.z)
                rig.add(c)
            self.rigs.append(rig)
            self.hubs.append(hub)
        self.radius = float(np.mean(radii))
        # The model gives the tail and the reverse lamps their own meshes but ONE shared
        # emissive material: a fresh one each, so they can flare independently.
        self.tail = standard_material(0x201a18, 0.6, emissive=0xff2a12)
        self.reverse = standard_material(0x201a18, 0.6, emissive=0xfff2e0)
        self.tail.emissive_intensity = self.reverse.emissive_intensity = self.LAMP_REST
        pm = None
        if paint is not None:
            pm = tp.MeshStandardMaterial()
            pm.color, pm.roughness, pm.metalness = tp.Color(*paint), 0.32, 0.55

        def finish(o):
            if not isinstance(o, tp.Mesh):
                return
            o.cast_shadow = o.receive_shadow = True
            # The .gltf packs AO and metal-roughness into ONE texture (R = AO, G = rough,
            # B = metal) and never authored R: an aoMap read is zero and the body goes black.
            try:
                o.material.ao_map = None
            except Exception:  # noqa: BLE001 - not every material has one
                pass
            if "lights_position_back" in o.name:
                o.set_material(self.tail)
            elif "lights_reverse" in o.name:
                o.set_material(self.reverse)
            elif pm is not None and o.material.name == "carpaint_color":
                o.set_material(pm)
        self.node.traverse(finish)
        # the head lamps' glass: the front of the model's lamp glass, kept for headlamps()
        for o in [o for o in meshes if LENS in o.name]:
            a = in_frame(o)
            idx = o.geometry.get_index()
            tri = np.arange(len(a)).reshape(-1, 3) if idx is None else np.asarray(idx).reshape(-1, 3)
            tri = tri[(a[tri][:, :, 2] > 0.0).all(1)]     # the same mesh has the tail lamps' glass
            used, inv = np.unique(tri, return_inverse=True)
            self._lens.append((a[used] + (0.0, 0.0, 0.004), inv.reshape(-1)))     # just before the glass

    def _box(self, radius):
        """No model: the chassis' box and four cylinders at PhysX's hubs."""
        self.radius = radius
        box = tp.Mesh(tp.BoxGeometry(1.95, 1.2, 4.4), standard_material(0x7a2f24, 0.5, 0.3))
        box.cast_shadow = True
        box.position.y = ride_height(radius)              # the chassis' middle: hang() takes it there
        self.hull.add(box)
        for sx, sz in ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
            hub = np.array([sx * 0.5 * TRACK, radius, sz * 0.5 * WHEELBASE])
            rig = tp.Group()
            rig.position.set(*map(float, hub))
            w = tp.Mesh(tp.CylinderGeometry(radius, radius, 0.3, 18), standard_material(0x1c1c1e, 0.8))
            w.rotate_z(math.pi / 2)
            w.cast_shadow = True
            rig.add(w)
            self.node.add(rig)
            self.rigs.append(rig)
            self.hubs.append(hub)

    def hang(self, ride):
        """Under a chassis whose middle rides `ride` over the road: the hull that far down, and
        fore or aft by the model's own offset (the model's axles are not about its origin,
        PhysX's are), so the arches stand over the wheels."""
        self.hull.position.y = -ride
        self.hull.position.z = -float(np.mean(self.hubs, axis=0)[2])

    def roll(self, distance):
        """The wheels turned by a way made (a car that is placed, not simulated)."""
        for rig in self.rigs:
            rig.rotation.x = distance / self.radius

    def _lamp(self, mat, on):
        if mat is not None:
            mat.emissive_intensity = self.LAMP_ON if on else self.LAMP_REST
            mat.needs_update()

    def brake(self, on):
        """The tail lamps flared. The material is touched only when the state flips:
        needs_update() re-uploads it."""
        if on != self._braking:
            self._braking = on
            self._lamp(self.tail, on)

    def reversing(self, on):
        """The reverse lamps lit; as brake()."""
        if on != self._reversing:
            self._reversing = on
            self._lamp(self.reverse, on)

    def headlamps(self):
        """The head lamps' glass as a skin of its own just before the model's glass, hidden until
        a scene shows it: self.glass (its material, emissive intensity 0), self.lamps (the skins)
        and self.head (where a lamp's light leaves from, hull frame, one per side). Built once."""
        if self.glass is not None or not self._lens:
            return
        self.glass = tp.MeshStandardMaterial()
        self.glass.color, self.glass.roughness = tp.Color(0.85, 0.85, 0.82), 0.2
        self.glass.emissive, self.glass.emissive_intensity = tp.Color(1.0, 0.96, 0.88), 0.0
        for v, index in self._lens:
            g = tp.BufferGeometry()
            g.set_attribute("position", np.ascontiguousarray(v, np.float32))
            g.set_index(np.ascontiguousarray(index, np.uint32))
            g.compute_vertex_normals()
            m = tp.Mesh(g, self.glass)
            m.visible = False
            self.hull.add(m)
            self.lamps.append(m)
            for side in (v[v[:, 0] > 0.0], v[v[:, 0] < 0.0]):
                c = side.mean(0)
                self.head.append((float(c[0]), float(c[1]), float(side[:, 2].max()) + 0.05))      # a lamp's light leaves from before its glass


class Car:
    """PhysX's vehicle with the Evoque tuning, and a Shell drawn over it.

    world              the tp.PhysxWorld that steps it
    position, rotation where the chassis' middle spawns (tp.Vector3, tp.Quaternion)
    shell              the Shell to wear (or wear() one later)
    wheel_radius       PhysX's wheel (default: the shell's measured radius, else 0.4)
    tire_friction      default TYRE_MU
    settings           any other tp.PhysxVehicle keyword (chassis_mass=, ...)

    A scene writes steer, throttle and brake (keys(), steer_for(), its own autopilot), then per
    physics step: control(), world.step(DT), sync(); per rendered frame: draw(a)."""

    def __init__(self, world, position, rotation, shell=None, wheel_radius=None, tire_friction=TYRE_MU, **settings):
        if wheel_radius is None:
            wheel_radius = shell.radius if shell is not None else 0.4
        self.world = world
        self.ride = ride_height(wheel_radius)
        self.veh = tp.PhysxVehicle(world, wheel_radius=wheel_radius, tire_friction=tire_friction,
                                   position=position, rotation=rotation, **settings)
        self.steer = self.throttle = self.brake = 0.0
        self.shell = None
        self.was = self.now = self.shown = None
        if shell is not None:
            self.wear(shell)

    def wear(self, shell):
        """Hang a Shell under the chassis and draw it where the car is."""
        self.shell = shell
        shell.hang(self.ride)
        self.was = self.now = None
        self.sync()
        self.draw()

    def respawn(self, position, rotation):
        """Back at a pose, at rest, in the forward gear."""
        self.veh.respawn(position, rotation)
        self.veh.gear = tp.PhysxVehicle.Gear.FORWARD
        self.steer = 0.0
        self.was = self.now = None
        self.sync()
        self.draw()

    @property
    def pos(self):
        p = self.veh.position
        return np.array([p.x, p.y, p.z])

    def axes(self):
        """(ahead, up) of the chassis in the world."""
        q = self.veh.quaternion
        x, y, z, w = q.x, q.y, q.z, q.w
        return (np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]),
                np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]))

    def control(self):
        self.veh.set_steer(self.steer)
        self.veh.set_throttle(self.throttle)
        self.veh.set_brake(self.brake)

    def sync(self):
        """After a step of the physics: where the car is now, and where it was a step ago
        (chassis pose, then the four wheels' local poses)."""
        p, q = self.veh.position, self.veh.quaternion
        now = [np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])]
        for k in range(4):
            lp, lq = self.veh.wheel_local_pose(k)
            now += [np.array([lp.x, lp.y, lp.z]), np.array([lq.x, lq.y, lq.z, lq.w])]
        self.was, self.now = self.now, now
        if self.was is None or float(np.linalg.norm(now[0] - self.was[0])) > 5.0:       # the first step, or a respawn
            self.was = now

    def draw(self, a=1.0):
        """The shell between the car's last two steps (a: 0 the one before, 1 the last). The
        physics steps at a fixed rate and a frame takes what time it takes: drawn at its last
        step, the car moves one step in one frame and two in the next and shakes against its
        own camera. Blended by the time the physics still owes (a = debt / DT) it advances a
        little every rendered frame, at the cost of up to one step of visual latency. A frame
        that is a whole number of steps (a film) draws the last step as it is."""
        if self.now is None:
            self.sync()
        if a >= 1.0:
            v = self.now
        else:
            v = []
            for k in range(0, 10, 2):
                # normalised lerp, sign-fixed so it never takes the long way: one step apart
                # (a few degrees at the worst wheelspin) it is the slerp's curve
                q0, q1 = self.was[k + 1], self.now[k + 1]
                q = q0 + ((q1 if float(q0 @ q1) >= 0.0 else -q1) - q0) * a
                v += [self.was[k] + (self.now[k] - self.was[k]) * a, q / max(float(np.linalg.norm(q)), 1e-12)]
        self.shown = (v[0], v[1])
        sh = self.shell
        if sh is None:
            return
        sh.node.position.set(*map(float, v[0]))
        sh.node.quaternion.set(*map(float, v[1]))
        for k in range(4):
            sh.rigs[k].position.set(*map(float, v[2 + 2 * k]))
            sh.rigs[k].quaternion.set(*map(float, v[3 + 2 * k]))

    def view(self):
        """(position, ahead) of the car as it is drawn: what a camera follows."""
        p, (x, y, z, w) = self.shown
        return p, np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
