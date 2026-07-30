// The program that AUTHORS the Hover Arena example, headlessly.
//
// The scene shipped in ExampleScenes.cpp is not hand-written JSON. It is built
// here, through the same editor core the app uses — SceneDocument, the
// PhysicsConfig / SensorConfig / ScriptConfig / GeneratorConfig writes, and one
// generator run driven exactly the way EditorApp::regenerate drives it — and
// then exported with SceneDocument::toJson. So the document is provably
// something the editor could have produced, and changing the example means
// changing code rather than editing a 60 KB literal by hand.
//
//   hover_arena_author scene.json [ExampleScenes.cpp]
//
// With one argument it writes the document, which is what the screenshot loop
// wants (`threepp_editor scene.json --play --screenshot=...` needs no rebuild).
// With two it also writes the translation unit that compiles that document into
// the editor, as chunked raw string literals — MSVC refuses a single string
// literal over 65535 bytes, and one long line is not something anyone should
// have to re-indent by hand either.
//
// Gated on the editor's Python scripting in apps/editor/CMakeLists.txt: the
// arena is a generator, and a generator that cannot run produces no arena.

#include "Scripting.hpp"
#include "ScriptHost.hpp"// runAuthoringSource, exactly as EditorApp::regenerate uses it

#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/geometries/TorusGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // --- the palette ---------------------------------------------------------
    // Three hues and no more: a blue-grey arena, a cyan that means "go here",
    // and an amber that means "you did". Every emissive in the scene is one of
    // the last two, which is what stops a dark scene turning into a fairground.
    constexpr int kNight = 0x0b111a;    // background, and the fog it fades into
    constexpr int kDeck = 0x39485c;     // the floor
    constexpr int kWall = 0x445468;     // perimeter
    constexpr int kPillar = 0x56657c;   // the obstacle field
    constexpr int kHull = 0x6c7c90;     // the drone
    constexpr int kCyan = 0x25d6f0;     // rings, thrusters, kerb
    constexpr int kAmber = 0xff9430;    // beacon, and a ring that has been taken

    std::shared_ptr<MeshStandardMaterial> standard(int color, float roughness, float metalness) {

        auto material = MeshStandardMaterial::create();
        material->color = Color(color);
        material->roughness = roughness;
        material->metalness = metalness;
        return material;
    }

    std::shared_ptr<MeshStandardMaterial> glowing(int color, float intensity) {

        auto material = standard(0x05070a, 0.4f, 0.f);
        material->emissive = Color(color);
        material->emissiveIntensity = intensity;
        return material;
    }

    std::shared_ptr<Mesh> mesh(const std::string& name,
                               const std::shared_ptr<BufferGeometry>& geometry,
                               const std::shared_ptr<Material>& material) {

        auto object = Mesh::create(geometry, material);
        object->name = name;
        object->castShadow = true;
        object->receiveShadow = true;
        return object;
    }

    void writePhysics(Object3D& object, PhysicsConfig::Body body, PhysicsConfig::Shape shape,
                      float mass, float friction, float restitution, bool trigger = false) {

        PhysicsConfig config;
        config.enabled = true;
        config.body = body;
        config.shape = shape;
        config.mass = mass;
        config.friction = friction;
        config.restitution = restitution;
        config.trigger = trigger;
        config.write(object);
    }

    void writeScript(Object3D& object, const std::string& source) {

        ScriptConfig config;
        config.source = source;
        config.write(object);
    }

    // ------------------------------------------------------------------ scripts

    // The arena. Plain module body, run once per Regenerate, and the only thing
    // in the document that knows where a pillar goes.
    const char* kGeneratorSource = R"PY(# Hover Arena - the arena itself.
#
# Everything this builds is DERIVED: press Regenerate and the floor, the
# perimeter, the pillar field and the kerb are rebuilt from SEED alone. The
# drone, the five goal rings and the beacon are hand-placed and are NOT this
# script's output, so a re-run never moves the course.
#
# set_user_data is what makes generated content playable: the flat
# "key=value;..." string is exactly what the Physics section writes.
import math
import random

import threepp
from threepp import editor

SEED = 20260730
HALF = 22.0            # arena half-width, metres
WALL_HEIGHT = 7.0
PILLARS = 15
CRATES = 9

STATIC_BOX = "body=static;shape=box;trigger=0;mass=1;friction=0.6;restitution=0.05"

random.seed(SEED)

# One geometry and one material per role, reused by every instance of it: the
# saved document then carries each of them once.
unit = threepp.BoxGeometry(1, 1, 1)

deck_material = threepp.MeshStandardMaterial()
deck_material.color = threepp.Color(0x39485c)
deck_material.roughness = 0.92
deck_material.metalness = 0.04

wall_material = threepp.MeshStandardMaterial()
wall_material.color = threepp.Color(0x445468)
wall_material.roughness = 0.85
wall_material.metalness = 0.08

pillar_material = threepp.MeshStandardMaterial()
pillar_material.color = threepp.Color(0x56657c)
pillar_material.roughness = 0.7
pillar_material.metalness = 0.18

kerb_material = threepp.MeshStandardMaterial()
kerb_material.color = threepp.Color(0x05070a)
kerb_material.emissive = threepp.Color(0x25d6f0)
kerb_material.emissive_intensity = 1.1
kerb_material.roughness = 0.4

band_material = threepp.MeshStandardMaterial()
band_material.color = threepp.Color(0x05070a)
band_material.emissive = threepp.Color(0x25d6f0)
band_material.emissive_intensity = 0.7
band_material.roughness = 0.4


def block(name, material, position, size, rotation=0.0, physics=STATIC_BOX,
          shadows=True):
    node = threepp.Mesh(unit, material)
    node.name = name
    node.position.set(position[0], position[1], position[2])
    node.scale.set(size[0], size[1], size[2])
    node.rotation.y = rotation
    node.cast_shadow = shadows
    node.receive_shadow = True
    if physics:
        node.set_user_data("physics", physics)
    return editor.add(node)


# --- floor and perimeter ----------------------------------------------------
floor = block("Arena Floor", deck_material, (0.0, -0.5, 0.0),
              (2 * HALF, 1.0, 2 * HALF), shadows=False)

for i, (dx, dz) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
    block("Arena Wall %d" % (i + 1), wall_material,
          (dx * (HALF + 0.4), WALL_HEIGHT * 0.5, dz * (HALF + 0.4)),
          (0.8 if dx else 2 * HALF + 1.6, WALL_HEIGHT, 0.8 if dz else 2 * HALF + 1.6))
    # A lit kerb at the base of each wall. It is the horizon line of the whole
    # scene: without it a dark floor and a dark wall meet in nothing.
    block("Arena Kerb %d" % (i + 1), kerb_material,
          (dx * (HALF - 0.2), 0.09, dz * (HALF - 0.2)),
          (0.35 if dx else 2 * HALF, 0.18, 0.35 if dz else 2 * HALF),
          physics=None, shadows=False)

# --- the pillar field -------------------------------------------------------
# The course is kept clear by rejection: nothing is placed within KEEP of the
# flight line the five rings describe, so a regenerate can never wall the
# course off.
COURSE = ((0.0, 14.0), (0.0, 6.0), (-7.0, -2.0), (-2.0, -12.0),
          (8.0, -8.0), (10.0, 4.0), (0.0, -20.0))
KEEP = 5.0


def clear_of_course(x, z, margin):
    for i in range(len(COURSE) - 1):
        ax, az = COURSE[i]
        bx, bz = COURSE[i + 1]
        dx, dz = bx - ax, bz - az
        span = dx * dx + dz * dz
        t = 0.0 if span <= 0.0 else max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / span))
        px, pz = ax + t * dx, az + t * dz
        if (x - px) ** 2 + (z - pz) ** 2 < margin * margin:
            return False
    return True


def scatter(count, margin, inner, outer):
    placed = []
    tries = 0
    while len(placed) < count and tries < 4000:
        tries += 1
        angle = random.uniform(0.0, 2.0 * math.pi)
        radius = math.sqrt(random.uniform(inner * inner, outer * outer))
        x, z = radius * math.cos(angle), radius * math.sin(angle)
        if not clear_of_course(x, z, margin):
            continue
        if any((x - px) ** 2 + (z - pz) ** 2 < 4.5 * 4.5 for px, pz in placed):
            continue
        placed.append((x, z))
    return placed


for i, (x, z) in enumerate(scatter(PILLARS, KEEP, 6.0, HALF - 2.5)):
    height = random.uniform(3.5, 9.5)
    width = random.uniform(1.1, 2.2)
    depth = random.uniform(1.1, 2.2)
    spin = random.uniform(0.0, math.pi)
    block("Pillar %d" % (i + 1), pillar_material, (x, height * 0.5, z),
          (width, height, depth), spin)
    # A lit band a little BELOW the top, not a lid on it: an emissive top face
    # is a flat slab from every camera that looks down, and looking down is what
    # a camera in an arena does.
    block("Pillar %d Band" % (i + 1), band_material, (x, height - 0.9, z),
          (width * 1.02, 0.1, depth * 1.02), spin, physics=None, shadows=False)

# The pad the drone stands over. Not decoration: a hovering machine with
# nothing beneath it reads as a machine that is falling. Its own material,
# because the kerb's brightness over five square metres is a lightbox.
pad_material = threepp.MeshStandardMaterial()
pad_material.color = threepp.Color(0x0d1620)
pad_material.emissive = threepp.Color(0x25d6f0)
pad_material.emissive_intensity = 0.09
pad_material.roughness = 0.5
block("Start Pad", pad_material, (COURSE[0][0], 0.05, COURSE[0][1]),
      (4.0, 0.1, 4.0), physics=None, shadows=False)

for i, (x, z) in enumerate(scatter(CRATES, KEEP - 1.2, 4.0, HALF - 4.0)):
    size = random.uniform(0.9, 1.7)
    block("Crate %d" % (i + 1), pillar_material, (x, size * 0.5, z),
          (size, size, size * random.uniform(0.8, 1.3)), random.uniform(0.0, math.pi))
)PY";

    // The drone. The controller is the point of the whole example: forces on the
    // physics clock, altitude from a raycast, damping from the NOISY IMU.
    const char* kDroneSource = R"PY(# Hover Arena - the drone.
#
# fixed_update runs once per physics substep with a constant dt, so the same
# gains settle at the same height whatever the frame rate is doing. Altitude is
# a short downward raycast with ignore=self (without it the ray starts inside
# the hull and finds it at distance zero); the rate feedback is the AUTHORED
# IMU's reading, which is noisy, seeded and rate-gated. The small wobble you
# can see is that noise going round the loop - it is not a bug to tune out.
import threepp

editor = getattr(threepp, "editor", None)


def _verb(name):
    # The PhysX-gated names are ABSENT from threepp.editor without the SDK,
    # rather than present and answering None, so ask before reaching.
    return getattr(editor, name, None) if editor is not None else None


class DroneController:

    hover_height = 2.2       # m above whatever is under the drone
    lift_p = 13.0            # thrust per metre of altitude error, per kg
    lift_d = 5.5             # thrust per m/s of climb rate, per kg
    level_p = 2.6            # N*m per radian of tilt
    level_d = 0.55           # N*m per rad/s of body rate  <- reads the IMU
    yaw_p = 0.9              # N*m of yaw command
    yaw_d = 0.25             # N*m per rad/s of yaw rate
    thrust = 5.0             # N of forward / strafe drive
    climb_rate = 1.6         # m/s of commanded climb
    drag = 1.4               # horizontal damping, per kg: a drone is not ice
    probe = 16.0             # m the altimeter looks down
    flash_seconds = 0.35

    def start(self, obj: threepp.Mesh, scene: threepp.Scene):
        self.obj = obj
        self.body = None
        self.imu = None
        self.cast = None
        self.climb_hold = self.hover_height
        self.flash = 0.0
        self.imu_reads = 0
        self.bumps = 0
        material = obj.material
        self.material = material if isinstance(material, threepp.MeshStandardMaterial) else None
        self.rest = self.material.emissive.get_hex() if self.material else 0

        body_of = _verb("rigid_body_from_object")
        self.cast = _verb("raycast")
        imu_of = _verb("imu_from_object")
        if body_of is None or self.cast is None:
            print("Hover Arena needs the PhysX build to fly - the drone will just sit there.",
                  flush=True)
            return

        self.body = body_of(obj)
        if self.body is None:
            print("Hover Arena: no rigid body on", obj.name, "- nothing to fly.", flush=True)
            return
        if imu_of is not None:
            self.imu = imu_of(obj)

        print("Hover Arena - W/S fly, A/D yaw, Q/E strafe, R/F climb. "
              "Fly through all five rings.", flush=True)

    # --- the controller, on the physics clock -------------------------------
    def fixed_update(self, dt: float):
        if self.body is None:
            return

        mass = self.body.mass
        rotation = self.body.rotation
        up = threepp.Vector3(0.0, 1.0, 0.0).apply_quaternion(rotation)
        forward = threepp.Vector3(0.0, 0.0, -1.0).apply_quaternion(rotation)
        right = threepp.Vector3(1.0, 0.0, 0.0).apply_quaternion(rotation)

        # Rate feedback: the sensor if it has spoken, the solver's own number
        # only until the first sample arrives. Both are rad/s; the IMU's is in
        # the sensor frame, so it is rotated into the world the torque is in.
        rate = None
        if self.imu is not None:
            sample = self.imu.latest()
            if sample is not None:
                rate = sample.angular_velocity.clone().apply_quaternion(rotation)
                self.imu_reads += 1
        if rate is None:
            rate = self.body.angular_velocity

        # --- altitude: what is under me, right now --------------------------
        origin = self.body.position
        hit = self.cast(origin, threepp.Vector3(0.0, -1.0, 0.0), self.probe, ignore=self.obj)
        altitude = hit.distance if hit is not None else origin.y

        keys = editor.is_key_down
        climb = (1.0 if keys("R") else 0.0) - (1.0 if keys("F") else 0.0)
        # The stick sets a HEIGHT, not a thrust: let go and it holds the last
        # one, which is what makes threading a ring at 3.8 m possible at all.
        self.climb_hold = max(0.8, min(9.0, self.climb_hold + climb * self.climb_rate * dt))

        error = self.climb_hold - altitude
        lift = mass * (9.81 + self.lift_p * error - self.lift_d * self.body.velocity.y)
        # Along the hull's own up, divided by how much of it still points up:
        # a tilted quad has to work harder to hold height, and that IS the feel.
        lean = max(0.35, up.y)
        lift = max(0.0, min(lift / lean, mass * 9.81 * 3.0))
        self.body.apply_force(up.clone().multiply_scalar(lift))

        # --- attitude: bring the hull's up back to the world's --------------
        tilt = up.clone().cross(threepp.Vector3(0.0, 1.0, 0.0))
        spin = rate.dot(up)
        torque = tilt.multiply_scalar(self.level_p * mass)
        # Damp the TILT rates only; the yaw rate has a command of its own below
        # and would otherwise be fought by its own damper twice.
        tilt_rate = rate.clone().sub(up.clone().multiply_scalar(spin))
        torque.sub(tilt_rate.multiply_scalar(self.level_d * mass))

        yaw = (1.0 if keys("A") else 0.0) - (1.0 if keys("D") else 0.0)
        torque.add(up.clone().multiply_scalar(mass * (self.yaw_p * yaw - self.yaw_d * spin)))
        self.body.apply_torque(torque)

        # --- translation ----------------------------------------------------
        drive = (1.0 if keys("W") else 0.0) - (1.0 if keys("S") else 0.0)
        strafe = (1.0 if keys("E") else 0.0) - (1.0 if keys("Q") else 0.0)
        push = forward.clone().multiply_scalar(drive * self.thrust * mass)
        push.add(right.clone().multiply_scalar(strafe * self.thrust * mass))
        velocity = self.body.velocity
        push.add(threepp.Vector3(velocity.x, 0.0, velocity.z).multiply_scalar(-self.drag * mass))
        self.body.apply_force(push)

    # --- the frame's final word: what you can see ---------------------------
    def update(self, dt: float):
        if self.flash > 0.0 and self.material is not None:
            self.flash = max(0.0, self.flash - dt)
            level = self.flash / self.flash_seconds
            self.material.emissive = threepp.Color(0xff3b1f)
            self.material.emissive_intensity = 2.4 * level
            if self.flash <= 0.0:
                self.material.emissive = threepp.Color(self.rest)
                self.material.emissive_intensity = 1.0

    def on_collision_enter(self, contact):
        self.bumps += 1
        self.flash = self.flash_seconds
        other = contact.other.name if contact.other is not None else "the world"
        impulse = contact.impulse.length()
        if impulse > 1.5:
            print("Hover Arena: clipped %s (%.1f N*s)" % (other, impulse), flush=True)
)PY";

    // One per ring, on the trigger volume — because the volume is the body the
    // trigger report belongs to, and the walk-up lookup finds no body from the
    // Group above it.
    const char* kRingSource = R"PY(# Hover Arena - one goal ring.
#
# This is on the GATE: an invisible thin box spanning the ring's hole, with
# trigger=1 in its physics. A trigger collides with nothing and reports who
# passed through it, which is exactly a goal. The torus is a sibling with no
# physics at all - a ring you can hit is a ring that ends the run.
import math

import threepp

editor = getattr(threepp, "editor", None)


class GoalRing:

    index = 1
    torus = "Ring 1 Torus"
    board = "Beacon"
    flash_seconds = 0.9

    def start(self, obj: threepp.Mesh, scene: threepp.Scene):
        self.obj = obj
        self.taken = False
        self.flash = 0.0
        self.celebrating = False
        self.clock = 0.0
        self.material = None
        ring = scene.get_object_by_name(self.torus)
        if ring is not None:
            material = ring.material
            if isinstance(material, threepp.MeshStandardMaterial):
                self.material = material
        # Resolve the neighbour HERE: every instance exists by now, even though
        # its own start() may not have run yet.
        self.score = None
        if editor is not None:
            self.score = editor.script_from_object(scene.get_object_by_name(self.board))

    def on_trigger_enter(self, other: threepp.Object3D):
        if other is None or other.name != "Drone":
            return
        self.flash = self.flash_seconds
        if self.taken:
            return
        self.taken = True
        if self.score is not None:
            self.score.scored(self.index, other)

    def update(self, dt: float):
        if self.material is None:
            return
        self.clock += dt
        if self.flash > 0.0:
            self.flash = max(0.0, self.flash - dt)
            level = self.flash / self.flash_seconds
            self.material.emissive = threepp.Color(0xffd27a)
            self.material.emissive_intensity = 1.4 + 5.0 * level
            return
        if self.celebrating:
            wave = 0.5 + 0.5 * math.sin(self.clock * 6.0 + self.index)
            self.material.emissive = threepp.Color(0xff9430)
            self.material.emissive_intensity = 1.6 + 3.2 * wave
        elif self.taken:
            self.material.emissive = threepp.Color(0xff9430)
            self.material.emissive_intensity = 1.5
        else:
            wave = 0.5 + 0.5 * math.sin(self.clock * 1.6 + self.index)
            self.material.emissive = threepp.Color(0x25d6f0)
            self.material.emissive_intensity = 1.5 + 0.9 * wave
)PY";

    // The scoreboard, on the beacon. Nothing here needs PhysX: it is reached by
    // script_from_object, which exists in every build that has scripts at all.
    const char* kBeaconSource = R"PY(# Hover Arena - the scoreboard.
#
# The rings call scored() on this instance through
# threepp.editor.script_from_object. There is no event bus: the instance IS
# the API, so a method call is the message.
import math

import threepp

editor = getattr(threepp, "editor", None)


class Scoreboard:

    rings = 5
    light = "Beacon Light"

    def start(self, obj: threepp.Mesh, scene: threepp.Scene):
        self.obj = obj
        self.score = 0
        self.clock = 0.0
        self.done = False
        material = obj.material
        self.material = material if isinstance(material, threepp.MeshStandardMaterial) else None
        self.lamp = scene.get_object_by_name(self.light)
        self.gates = []
        if editor is not None:
            for i in range(self.rings):
                gate = scene.get_object_by_name("Ring %d Gate" % (i + 1))
                instance = editor.script_from_object(gate)
                if instance is not None:
                    self.gates.append(instance)
        print("Hover Arena: 0/%d rings" % self.rings, flush=True)

    def scored(self, index: int, who: threepp.Object3D):
        self.score += 1
        print("Hover Arena: %d/%d rings (ring %d by %s)"
              % (self.score, self.rings, index, who.name if who is not None else "?"),
              flush=True)
        if self.score >= self.rings and not self.done:
            self.done = True
            print("Hover Arena: course complete - every ring taken.", flush=True)
            for gate in self.gates:
                gate.celebrating = True

    def update(self, dt: float):
        self.clock += dt
        share = float(self.score) / float(max(1, self.rings))
        # Cyan while there is work left, gold once there is not. The beacon is
        # the one thing in the scene that reads the score from across the arena.
        cold = threepp.Color(0x25d6f0)
        warm = threepp.Color(0xff9430)
        tint = cold.clone().lerp(warm, share)
        pulse = 0.5 + 0.5 * math.sin(self.clock * (2.0 + 4.0 * share))
        level = (1.4 + 2.2 * share) * (0.75 + 0.45 * pulse)
        if self.material is not None:
            self.material.emissive = tint
            self.material.emissive_intensity = level
        if self.lamp is not None:
            self.lamp.color = tint
            self.lamp.intensity = 6.0 + 26.0 * share * (0.7 + 0.5 * pulse)
)PY";

    // ------------------------------------------------------------------- scene

    struct RingSpec {
        Vector3 centre;
        Vector3 facing;
    };

    void addRing(Scene& scene, int index, const RingSpec& spec) {

        const auto label = std::to_string(index);

        auto group = Group::create();
        group->name = "Ring " + label;
        group->position.copy(spec.centre);
        group->lookAt(Vector3(spec.centre).add(spec.facing));

        // The torus draws the goal and collides with nothing: hitting the rim
        // of a gate you were aiming through is not a game, it is a punishment.
        auto torus = mesh("Ring " + label + " Torus",
                          TorusGeometry::create(1.85f, 0.17f, 16, 64),
                          glowing(kCyan, 1.7f));
        torus->castShadow = false;
        torus->receiveShadow = false;
        group->add(torus);

        // The gate: a thin box across the hole. Inscribed in it (the hole is
        // 3.42 m across, so 2.3 m square clears the rim on the diagonal),
        // invisible, and a trigger — bodies pass through and it reports them.
        auto gate = mesh("Ring " + label + " Gate",
                         BoxGeometry::create(2.3f, 2.3f, 0.3f),
                         standard(kCyan, 0.5f, 0.f));
        gate->visible = false;
        gate->castShadow = false;
        gate->receiveShadow = false;
        writePhysics(*gate, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.5f, 0.f, /*trigger*/ true);

        auto source = std::string(kRingSource);
        // The two per-ring parameters, as authored field values — the same
        // mechanism the inspector writes, so they show up as widgets there.
        ScriptConfig config;
        config.source = source;
        config.setField("index", std::to_string(index));
        config.setField("torus", "Ring " + label + " Torus");
        config.setField("board", "Beacon");
        config.write(*gate);
        group->add(gate);

        scene.add(group);
    }

    std::shared_ptr<Scene> buildScene() {

        auto scene = Scene::create();
        scene->name = "Scene";
        scene->background = Color(kNight);
        // Verified to round-trip: ObjectExporter writes scene.fog and
        // ObjectLoader reads it back, so the depth cue survives a save.
        scene->fog = Fog(Color(kNight), 42.f, 135.f);

        // --- how it opens ----------------------------------------------------
        // The two keys the editor reads when a document is OPENED (see
        // doc/editor.md). Without them the arena is framed automatically, which
        // for a course laid out along -Z means arriving pointed at a wall.
        //
        // The vantage is 7 m behind the drone and 2.4 m over it, looking down
        // the course: the drone in the foreground third above its pad, the first
        // gate right behind it, the rest of the rings receding and the beacon
        // standing at the far end. Two numbers are not free choices — the
        // perimeter wall is at z = 22.4, so a camera much further back is a
        // camera outside the arena photographing masonry, and a camera much
        // higher pitches the far end of the course out of frame.
        //
        // It is also the CHASE offset: Follow keeps whatever offset the camera
        // has when the document opens, so this one placement has to read as an
        // establishing shot AND fly as a chase cam. The target sits ON the drone
        // for the same reason — it is where Follow puts it a fraction of a
        // second later, and a view that lurches as soon as you look at it is
        // not a view anybody authored.
        scene->userData["editorView"] = std::string("0,4.6,21@0,2.2,14");
        scene->userData["editorFollow"] = std::string("Drone");

        // --- lights ---------------------------------------------------------
        // One key light with shadows, and just enough fill to keep the far side
        // of a pillar from being a silhouette.
        auto key = DirectionalLight::create(0xdce7ff, 3.0f);
        key->name = "Key Light";
        key->position.set(16.f, 26.f, 12.f);
        key->castShadow = true;
        // The default shadow frustum is +-1, which in a 44 m arena means no
        // shadows at all. It round-trips (ObjectExporter writes the shadow
        // camera), so widening it here is a property of the document.
        if (auto* shadowCamera = dynamic_cast<OrthographicCamera*>(key->shadow->camera.get())) {
            shadowCamera->left = -30.f;
            shadowCamera->right = 30.f;
            shadowCamera->top = 30.f;
            shadowCamera->bottom = -30.f;
            shadowCamera->nearPlane = 1.f;
            shadowCamera->farPlane = 90.f;
            shadowCamera->updateProjectionMatrix();
        }
        key->shadow->mapSize.set(2048, 2048);
        key->shadow->bias = -0.0008f;
        scene->add(key);

        auto sky = HemisphereLight::create(0x4a6a92, 0x0d1118, 1.15f);
        sky->name = "Sky Fill";
        scene->add(sky);

        auto ambient = AmbientLight::create(0x35435a, 0.55f);
        ambient->name = "Ambient Light";
        scene->add(ambient);

        // --- the drone ------------------------------------------------------
        // A Mesh, not a Group: the collider is then this box and nothing else,
        // so the booms and rotors below can overhang without snagging.
        auto drone = mesh("Drone", BoxGeometry::create(1.f, 0.26f, 1.f),
                          standard(kHull, 0.42f, 0.2f));
        drone->position.set(0.f, 2.2f, 14.f);
        writePhysics(*drone, PhysicsConfig::Body::Dynamic, PhysicsConfig::Shape::Auto,
                     1.4f, 0.4f, 0.05f);
        writeScript(*drone, kDroneSource);
        {
            SensorConfig imu;
            imu.enabled = true;
            imu.type = SensorConfig::Type::Imu;
            imu.rateHz = 200.f;
            imu.seed = 11;
            // Loud on purpose. A spec-sheet MEMS gyro gives a wobble you have to
            // measure; this one gives a wobble you can see, which is the point of
            // closing the loop on the sensor rather than on the solver.
            imu.gyroNoiseDensity = 0.035f;
            imu.gyroRandomWalk = 2e-4f;
            imu.accelNoiseDensity = 0.12f;
            imu.write(*drone);
        }

        auto boomMaterial = standard(0x5d6c80, 0.45f, 0.2f);
        // A RING, not a disc. Four lit discs at close range are four flat
        // lozenges; four lit rings with a dark hub inside them are a
        // quadcopter, from any angle and at any distance.
        auto rotorMaterial = glowing(kCyan, 1.8f);
        auto rotorGeometry = TorusGeometry::create(0.36f, 0.045f, 10, 28);
        auto hubMaterial = standard(0x3b4759, 0.4f, 0.25f);
        auto hubGeometry = CylinderGeometry::create(0.1f, 0.1f, 0.12f, 12);
        auto boomGeometry = BoxGeometry::create(1.f, 1.f, 1.f);
        for (int i = 0; i < 4; ++i) {
            const float x = (i & 1) ? 0.74f : -0.74f;
            const float z = (i & 2) ? 0.74f : -0.74f;

            auto boom = mesh("Drone Boom " + std::to_string(i + 1), boomGeometry, boomMaterial);
            boom->position.set(x * 0.55f, 0.f, z * 0.55f);
            boom->scale.set(1.05f, 0.12f, 0.16f);
            boom->rotation.y = (x * z > 0.f) ? -math::PI / 4.f : math::PI / 4.f;
            drone->add(boom);

            auto rotor = mesh("Drone Rotor " + std::to_string(i + 1), rotorGeometry, rotorMaterial);
            rotor->position.set(x, 0.1f, z);
            rotor->rotation.x = -math::PI / 2.f;
            rotor->castShadow = false;
            drone->add(rotor);

            auto hub = mesh("Drone Hub " + std::to_string(i + 1), hubGeometry, hubMaterial);
            hub->position.set(x, 0.1f, z);
            hub->castShadow = false;
            drone->add(hub);
        }

        auto canopy = mesh("Drone Canopy", SphereGeometry::create(0.32f, 24, 12),
                           standard(0x5a6b86, 0.3f, 0.2f));
        canopy->position.set(0.f, 0.14f, -0.06f);
        canopy->scale.set(1.f, 0.55f, 1.2f);
        drone->add(canopy);

        // Which way is forward. An amber nose against four cyan rotors is the
        // only heading cue a top-down viewer gets, and without it a hovering
        // quad is a disc with no front.
        auto nose = mesh("Drone Nose", BoxGeometry::create(0.22f, 0.1f, 0.5f),
                         glowing(kAmber, 1.8f));
        nose->position.set(0.f, 0.02f, -0.68f);
        nose->castShadow = false;
        drone->add(nose);

        // The LIDAR rides on its own node, because an object carries exactly one
        // sensor entry and the IMU already has the hull's.
        auto lidar = mesh("Drone Lidar", CylinderGeometry::create(0.09f, 0.09f, 0.14f, 16),
                          glowing(kAmber, 1.6f));
        lidar->position.set(0.f, 0.2f, 0.f);
        lidar->castShadow = false;
        {
            SensorConfig scan;
            scan.enabled = true;
            scan.type = SensorConfig::Type::Lidar;
            scan.beams = SensorConfig::Beams::VLP16;
            scan.faceSize = 112;
            scan.rateHz = 10.f;
            scan.nearPlane = 1.2f;
            scan.farPlane = 12.f;
            scan.rangeStddev = 0.015f;
            scan.seed = 23;
            scan.write(*lidar);
        }
        drone->add(lidar);
        scene->add(drone);

        // --- the course -----------------------------------------------------
        static const RingSpec rings[] = {
                {{0.f, 2.4f, 6.f}, {0.f, 0.f, -1.f}},
                {{-7.f, 3.0f, -2.f}, {-0.66f, 0.06f, -0.75f}},
                {{-2.f, 3.8f, -12.f}, {0.45f, 0.07f, -0.89f}},
                {{8.f, 2.6f, -8.f}, {0.93f, -0.1f, 0.36f}},
                {{10.f, 3.4f, 4.f}, {0.16f, 0.06f, 0.99f}},
        };
        for (int i = 0; i < 5; ++i) addRing(*scene, i + 1, rings[i]);

        // --- the beacon -----------------------------------------------------
        auto beacon = mesh("Beacon", CylinderGeometry::create(0.35f, 0.85f, 7.f, 6),
                           glowing(kCyan, 1.6f));
        beacon->position.set(0.f, 3.5f, -20.f);
        writeScript(*beacon, kBeaconSource);
        // The mast it stands on: an emissive pylon with no base reads as a
        // floating prop.
        auto plinth = mesh("Beacon Plinth", CylinderGeometry::create(1.5f, 2.1f, 0.6f, 6),
                           standard(kWall, 0.75f, 0.15f));
        plinth->position.set(0.f, -3.2f, 0.f);
        beacon->add(plinth);

        auto lamp = PointLight::create(Color(kCyan), 8.f, 40.f);
        lamp->name = "Beacon Light";
        lamp->position.set(0.f, 1.2f, 0.f);
        beacon->add(lamp);
        scene->add(beacon);

        return scene;
    }

    // Runs the scene's generator exactly as EditorApp::regenerate does: fill a
    // DETACHED node, attach it only on success.
    bool runGenerator(Scene& scene, std::string& error) {

        const auto config = GeneratorConfig::read(scene);
        if (!config) {
            error = "the scene carries no generator";
            return false;
        }

        auto output = Group::create();
        output->name = "Generated";
        output->userData[GeneratorConfig::generatedKey] = std::string("1");

        {
            scripting::authoringSink() = output.get();
            scripting::authoringScene() = &scene;
            struct SinkGuard {
                ~SinkGuard() {
                    scripting::authoringSink() = nullptr;
                    scripting::authoringScene() = nullptr;
                }
            } guard;
            error = scripting::runAuthoringSource(config->source, "generator");
        }
        if (!error.empty()) return false;
        if (output->children.empty()) {
            error = "the generator added nothing";
            return false;
        }

        scene.add(output);
        return true;
    }

    // --- the translation unit ------------------------------------------------
    // MSVC caps one string literal at 65535 bytes, so the document is emitted as
    // chunks joined at startup. Raw literals with a delimiter JSON cannot
    // contain, so nothing in the document has to be escaped.
    void writeSource(const std::filesystem::path& path, const std::string& json) {

        constexpr std::size_t kChunk = 8000;

        std::ofstream out(path, std::ios::binary);
        out << R"(// The scenes the editor ships, compiled into the binary.
//
// GENERATED — do not edit. Rebuild it from the program that authors the scene:
//
//     cmake --build <build> --target hover_arena_author
//     <build>/bin/hover_arena_author hover_arena.json apps/editor/ExampleScenes.cpp
//
// The document is embedded rather than found on disk for the reason
// ViewportMarkers.cpp embeds its SVG: the editor does not depend on locating
// asset files at runtime. It arrives as chunks because MSVC refuses a single
// string literal over 65535 bytes.

#include "ExampleScenes.hpp"

#include <string>
#include <vector>

namespace {

    // Hover Arena — see apps/editor/tools/HoverArenaAuthor.cpp, which built it.
    const char* const kHoverArena[] = {
)";
        for (std::size_t i = 0; i < json.size(); i += kChunk) {
            out << "            R\"JSON(" << json.substr(i, kChunk) << ")JSON\",\n";
        }
        out << R"(    };

}// namespace

namespace threepp::editor::examples {

    const std::vector<Example>& all() {

        static const std::vector<Example> examples{
                {"hover-arena", "Hover Arena",
                 "a physics drone, five trigger rings and a scoreboard - fly it with W/S/A/D"},
        };
        return examples;
    }

    const Example* find(std::string_view slug) {

        for (const auto& example : all()) {
            if (example.slug == slug) return &example;
        }
        return nullptr;
    }

    std::string json(std::string_view slug) {

        if (slug != "hover-arena") return {};
        std::string document;
        std::size_t size = 0;
        for (const auto* chunk : kHoverArena) size += std::char_traits<char>::length(chunk);
        document.reserve(size);
        for (const auto* chunk : kHoverArena) document += chunk;
        return document;
    }

}// namespace threepp::editor::examples
)";
    }

}// namespace


int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "usage: hover_arena_author <scene.json> [ExampleScenes.cpp]\n";
        return 2;
    }

    std::string error;
    if (!scripting::ensureInterpreter(&error)) {
        std::cerr << "hover_arena_author: no Python interpreter: " << error << std::endl;
        return 1;
    }

    SceneDocument document;
    auto scene = buildScene();

    GeneratorConfig generator;
    generator.source = kGeneratorSource;
    generator.write(*scene);

    document.replaceScene(scene);

    if (!runGenerator(document.scene(), error)) {
        std::cerr << "hover_arena_author: generator failed: " << error << std::endl;
        return 1;
    }

    const auto json = document.toJson(/*prettyPrint*/ true, &error);
    if (json.empty()) {
        std::cerr << "hover_arena_author: export failed: " << error << std::endl;
        return 1;
    }

    {
        std::ofstream out(argv[1], std::ios::binary);
        out << json;
    }
    std::cout << "hover_arena_author: wrote " << argv[1] << " (" << json.size() << " bytes, "
              << [&] {
                     std::size_t objects = 0;
                     document.scene().traverse([&](Object3D&) { ++objects; });
                     return objects;
                 }()
              << " objects)" << std::endl;

    if (argc > 2) {
        writeSource(argv[2], json);
        std::cout << "hover_arena_author: wrote " << argv[2] << std::endl;
    }
    return 0;
}
