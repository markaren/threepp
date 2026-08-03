// The program that AUTHORS the Timber Yard example, headlessly.
//
// Sibling of HoverArenaAuthor.cpp, and the same contract: the scene shipped in
// the binary is not hand-written JSON, it is built here through the editor core
// (SceneDocument, the Physics/Joint/Sensor/Script/Conveyor config writes) and
// exported with SceneDocument::toJson. Changing the example means changing code
// rather than editing a couple of hundred KB of JSON by hand.
//
//   timber_yard_author scene.json [ExampleSceneTimberYard.cpp]
//   timber_yard_author --check
//
// With one argument it writes the document, which is what the screenshot loop
// wants (`threepp_editor scene.json --play --screenshot=...` needs no rebuild).
// With two it also writes the translation unit that compiles that document into
// the editor. `--check` authors the scene TWICE and compares the bytes, which
// is the machine-checkable half of the promise below.
//
// BYTE REPRODUCIBILITY IS A REQUIREMENT, not an aspiration. Two runs of this
// program produce identical JSON, so `git diff` after a regenerate shows what
// was authored differently and nothing else. Three things have to hold:
//
//   * every uuid is DERIVED (assignAuthoredUuids), not drawn — Object3D,
//     BufferGeometry, Material AND Texture each draw a random one at
//     construction, and the `images` entry is keyed off the texture's, so the
//     one procedural texture in here (the belt ribbon's) has to be pinned too.
//     It was the last thing drifting between two runs;
//   * nothing samples a clock, a random_device, or an address.
//
// SIZE. The document is 352 KB, against a 600 KB ceiling, and the serializer's
// dominant cost is inlined geometry — a BufferGeometry writes every vertex,
// while a BoxGeometry writes its three numbers. So the yard is built from
// PRIMITIVES wherever a primitive will do, and the one piece of generated
// geometry it keeps is the conveyor's, because that IS the machine. The scenery
// comment further down carries the measurements that decided the rest.
//
// Gated on the editor's Python scripting in apps/editor/CMakeLists.txt, like
// its sibling — a document whose point is five scripts wants an interpreter to
// have checked they at least compile.

#include "Scripting.hpp"

#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/MaterialTextureSlots.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/TextConfig.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/light_interfaces.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/Scene.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_set>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // --- the yard ------------------------------------------------------------
    // One axis carries the whole story: logs enter at -X on the rack, cross the
    // gate, ride the belt through +X and stop at the bar. Every number below is
    // on that line, so moving one thing means moving the numbers around it and
    // nothing else.
    constexpr float kSlope = 0.115f;     // rack fall, radians (~6.6 degrees)
    constexpr float kRackCentreX = -9.8f;// rack deck centre
    constexpr float kRackCentreY = 1.242f;// the deck ends a log's drop above the
                                          // belt — see kRackEndX
    constexpr float kRackLength = 5.6f;
    constexpr float kRackHalfDepth = 1.15f;// half the width the logs lie across
    constexpr float kDeckHalf = 0.13f;     // half thickness of the rack deck
    constexpr float kRackEndX = kRackCentreX + kRackLength * 0.5f;

    // The gate stands OVER the last stretch of deck, and everything about it is
    // clearance: a leaf that reaches into the deck, or into the kerbs beside it,
    // is a leaf that cannot lift — which is exactly what the first version of
    // this scene did, and it looked like a scripting bug for an hour.
    constexpr float kGateX = -7.3f;   // the plane the gate closes
    constexpr float kHingeY = 1.7f;   // hinge height, above the log tops
    constexpr float kGateDrop = 0.6f; // how far the leaf hangs below the hinge
    constexpr float kGateWidth = 2.2f;// narrower than the gap between the kerbs
    constexpr float kKerbEndX = -8.6f;// so the kerbs stop short of the machinery

    // --- why there are TWO gates ---------------------------------------------
    // A queue of logs on a slope packs into contact and then moves as ONE BODY:
    // open a single gate long enough for the front log to get clear, and the
    // second log has travelled exactly as far and is under the leaf when it
    // comes down. There is no open time that releases one log and no open time
    // that releases none — they accelerate together, so timing alone cannot
    // separate them. That is what an escapement is for, and this is the
    // simplest one: a HOLD-BACK CLAMP that pins the second log while the first
    // is let go.
    //
    // The clamp is authored LIFTED, and clamping means driving it to a target
    // BEYOND the log. The log stops it short, the drive keeps pushing with
    // stiffness times the leftover error, and a cylinder pinned between the arm
    // and the deck cannot roll — no clearance to get right, and it works
    // wherever on the log's back the arm happens to land.
    constexpr float kLogPitch = 0.42f;// authored PACKED: logs touch, as they end
                                      // up anyway after the first release
    constexpr float kClampX = kGateX - 0.25f - kLogPitch;// over the second log
    constexpr float kClampPivotY = 2.3f;
    constexpr float kClampArm = 0.9f;
    // Swept down by this much, the arm's far end lands on the second log's back
    // (0.9 * sin = the 0.70 m from the pivot height to the log's top), and the
    // pivot sits back by 0.9 * cos so that it lands ON kClampX.
    constexpr float kClampAngle = 0.9f;
    constexpr float kClampPivotX = kClampX - kClampArm * 0.622f;

    constexpr float kBeltY = 0.85f;// belt deck height
    constexpr float kBeltStartX = -7.0f;
    constexpr float kBeltEndX = 1.8f;

    // --- the discharge end, and why it is shaped like this --------------------
    // Two problems shape everything past the saw, and they pull against each
    // other.
    //
    // The counting volume has to CLEAR. The gate keeper's interlock asks whether
    // the bay is empty, so a log parked in it for ever means the gate never
    // opens again: one log delivered, then deadlock. So the volume is a gate the
    // logs roll THROUGH, and what they come to rest against is further on.
    //
    // The stop bar has to be LOADED, or "two logs together" is not a different
    // event from one. A log resting on a dead apron pushes with nothing; a log
    // still on the belt pushes with the belt's friction behind it, about 140 N,
    // and two of them in contact push with twice that. So the bar hangs at the
    // END OF THE BELT, where the belt is still driving: one log leans on it hard
    // enough to swing it open and pass through, two together are over the break
    // threshold. Past it the belt stops and the apron takes the delivered logs
    // away to a stack that leans on nothing.
    constexpr float kBarX = 1.6f;    // the flap at the end of the belt
    constexpr float kBarHingeY = 1.92f;
    constexpr float kBarDrop = 0.92f;// hangs to just above the belt

    constexpr float kApronStartX = 1.7f;
    constexpr float kApronEndX = 6.1f;
    constexpr float kApronY = 0.62f;     // top surface at kApronStartX
    constexpr float kApronSlope = 0.035f;// radians of fall toward the berm

    // Well upstream of the flap. The saw shoves a log's halves apart as it cuts,
    // and a half that arrives at the flap still skewed from that hits it off
    // square and loads the mount far harder than a settled one — which put the
    // mission's own spike into the same range as the abuse case it is supposed
    // to be told apart from. Three metres of belt between them is what separates
    // the two again.
    constexpr float kBayX = -1.5f;// centre of the counting volume
    constexpr float kBermX = 6.0f;

    float apronTopY(float x) { return kApronY - (x - kApronStartX) * std::tan(kApronSlope); }

    constexpr float kLogRadius = 0.2f;
    // SHORTER THAN THE BELT IS WIDE (1.5 m), with 15 cm of margin each side.
    // The first version was 1.7 m on a 1.5 m belt: a log could hang its end off
    // the edge, and an overhanging end that reaches the blade plane is a log
    // being milled side-on for ever — which is exactly what happened to the last
    // log of a run, the one that arrives slowest.
    constexpr float kLogLength = 1.2f;
    constexpr int kLogCount = 8;

    // --- what makes the saw a saw --------------------------------------------
    // Every log is authored as TWO HALVES held together by a breakable FIXED
    // joint, with a slot between them. The blade is thicker than the slot, so a
    // log arriving at the saw meets it on the halves' inner faces — a wedge
    // whose contact normals point along ±Z, which is the one direction that
    // separates them. The joint gives way, the halves part around the blade and
    // ride on down the belt as two, and that is the cut.
    //
    // A script COULD do this the other way — the play runtime does support
    // spawning during Play (build a mesh, add it to the live scene(), give it a
    // body through threepp.editor.world()) — but authored halves are the better
    // trade here: they are in the document, so the scene is deterministic and
    // byte-reproducible, they need no script to exist, and what does the cutting
    // is a BREAKABLE JOINT, which is the feature worth showing.
    // THIN. Both of these were four times bigger, and the cut was a detonation:
    // a wide collider overlaps the halves deeply, the solver has to push that
    // overlap out in the substep the joint happens to snap, and the halves
    // inherit a depenetration impulse instead of a cut — off the belt entirely,
    // sometimes. A blade is a few millimetres of steel, so the COLLIDER is a few
    // millimetres of steel; nothing says the visual disc has to be a slab.
    constexpr float kCutSlot = 0.012f;// gap between the halves, at rest
    constexpr float kHalfLength = (kLogLength - kCutSlot) * 0.5f;
    constexpr float kHalfOffset = (kHalfLength + kCutSlot) * 0.5f;// centre of each half
    constexpr float kBladeThickness = 0.09f;// > kCutSlot, or the blade misses
    constexpr float kBladeRadius = 0.46f;
    // The blade reaches the log's CENTRE LINE and no further. That is where the
    // two halves' inner faces are closest, so it is where a thin blade engages
    // them most for the least penetration — and it means the saw never ploughs
    // the full cross-section, which is the other half of why the cut used to
    // throw things.
    constexpr float kBladeY = 1.51f;

    // Top surface of the sloped rack deck at x.
    float rackTopY(float x) {
        return kRackCentreY - (x - kRackCentreX) * std::tan(kSlope) + kDeckHalf / std::cos(kSlope);
    }

    float logX(int index) {
        // Downhill end first, so Log 1 is the one resting on the gate leaf: its
        // centre is a radius clear of the leaf's upstream face, and the rest of
        // the pack follows at one diameter.
        return kGateX - 0.05f - kLogRadius - static_cast<float>(index) * kLogPitch;
    }

    // --- the palette ---------------------------------------------------------
    // A daylight scene, so nothing is allowed near the 3% albedo trap that eats
    // a dark material under a bright sky. Two working colours (sawn timber and
    // safety orange) against three landscape greens/browns, and no emissive at
    // all above 3 — over that GL clips it to white.
    constexpr int kSky = 0x9fb8cc;    // background, and what the fog fades into
    constexpr int kGround = 0x556138; // the clearing
    constexpr int kYardDirt = 0x796046;// packed earth under the machines
    constexpr int kTimber = 0x9c7040; // sawn log ends and the rack
    constexpr int kBark = 0x54402c;   // log sides, and the pines
    constexpr int kSteel = 0x8b929a;  // posts and frames
    constexpr int kOrange = 0xc2571f; // the gate leaf and the stop bar: the two
                                      // things in the yard that MOVE on purpose
    constexpr int kSign = 0xd8c9a4;
    constexpr int kNeedle = 0x3d5a2a;
    constexpr int kLogSkin = 0x8a6134;// the cargo, lighter than the trees

    std::shared_ptr<MeshStandardMaterial> standard(int color, float roughness, float metalness,
                                                   bool vertexColors = false) {

        auto material = MeshStandardMaterial::create();
        material->color = Color(color);
        material->roughness = roughness;
        material->metalness = metalness;
        material->vertexColors = vertexColors;
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

    // ------------------------------------------------------------------ scripts
    //
    // All five are INLINE in the document (ScriptConfig::source, no path), so the
    // example stands on its own the way a shipped example has to.

    // 1. The gate. The coroutine showcase: one loop, top to bottom, doing what a
    // state machine would need an enum and two timers for.
    const char* kGateKeeperSource = R"PY(# Timber Yard - the release gate.
#
# This is the whole reason the yard has a gate: ONE LOOP, read top to bottom,
# that says what the gate does. Wait for the bay to be clear, drive open, hold
# it open for six tenths of a SIMULATED second, drive shut, wait until the
# encoder agrees it is shut. A state machine would need an enum, two timers and
# the sequence scattered across update() and fixed_update().
#
# Three details worth knowing, all of them documented in doc/editor.md:
#   * wait() is on the SIMULATED clock, so the gate holds open for the same
#     number of substeps on a machine that renders half as fast;
#   * the coroutine pump runs AFTER physics and after every script's update(),
#     so an until() predicate reads the settled frame - the bay's occupancy is
#     this frame's, not last frame's;
#   * a raise in here disables this instance whole, so the predicates below
#     answer False rather than reaching through a None.
import threepp

editor = getattr(threepp, "editor", None)


def _verb(name):
    # The PhysX-gated names are ABSENT from threepp.editor without the SDK,
    # rather than present and answering None, so ask before reaching.
    return getattr(editor, name, None) if editor is not None else None


class GateKeeper:

    counter = "Bay Trigger"# who knows whether the bay is clear
    holdback = "Holdback Hinge"# the other half of the escapement
    open_angle = -1.4      # rad; NEGATIVE is downstream, away from the pack
    shut_angle = 0.0
    clamp_angle = 1.05     # driven PAST the log, so the drive always pushes
    lift_angle = 0.0       # the arm's authored, out-of-the-way pose
    seat_seconds = 0.35    # SIM seconds to let the clamp bite before opening
    open_seconds = 1.4     # SIM seconds the leaf stays up
    shut_tolerance = 0.10  # rad the encoder may still read and count as shut

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.wanted = 0   # release requests, raised by the YardMaster
        self.released = 0
        self.cycles = 0
        self.joint = None
        self.clamp = None
        self.encoder = None
        self.bay_node = None
        self.task = None
        if editor is None:
            return

        self.bay_node = editor.scene().get_object_by_name(self.counter)

        joint_of = _verb("joint_from_object")
        if joint_of is None:
            print("Timber Yard needs the PhysX build to work the gate.", flush=True)
            return
        self.joint = joint_of(obj)
        if self.joint is None:
            print("Timber Yard: no live joint on", obj.name, flush=True)
            return
        self.clamp = joint_of(editor.scene().get_object_by_name(self.holdback))

        # The encoder authored on THIS node reads THIS joint - a joint sensor on
        # a joint node needs no name, the node is the reference.
        encoder_of = _verb("encoder_from_object")
        if encoder_of is not None:
            self.encoder = encoder_of(obj)

        self.joint.set_drive_target(self.shut_angle)
        if self.clamp is not None:
            self.clamp.set_drive_target(self.lift_angle)
        self.task = editor.start_coroutine(self.run())

    # --- the gate cycle ------------------------------------------------------
    # One loop, top to bottom, and every step of it is a sentence: pin the next
    # log, lift the leaf, hold it up long enough for one log to get onto the
    # belt, drop the leaf, wait until the encoder says it is really down, let the
    # pack settle forward by one.
    def run(self):
        while True:
            yield editor.until(self.clear_to_release)
            # THE INTERLOCK, in one line. Held down, SPACE skips the hold-back
            # entirely: the leaf lifts with nothing pinning the pack, and the
            # whole rack pours onto the belt at once. That is the yard's one
            # failure, and it is a person's doing - several logs reach the stop
            # bar together and the joint holding it is not built for that.
            forced = self.override()
            if not forced:
                self.hold(self.clamp_angle)
                yield editor.wait(self.seat_seconds)
            self.joint.set_drive_target(self.open_angle)
            yield editor.wait(self.open_seconds * (3.0 if forced else 1.0))
            self.joint.set_drive_target(self.shut_angle)
            yield editor.until(self.shut)
            self.hold(self.lift_angle)
            self.released += 1
            self.cycles += 1

    def hold(self, angle):
        if self.clamp is not None:
            self.clamp.set_drive_target(angle)

    # Asked once per frame by the until() above. The bay's live instance is
    # looked up HERE rather than cached: script_from_object is a lookup, and a
    # predicate that resolves what it reads cannot go stale.
    def clear_to_release(self):
        if self.wanted <= self.released:
            return False
        if self.override():
            return True
        bay = editor.script_from_object(self.bay_node)
        return bay is None or bay.occupancy <= 0

    # The manual override. Holding SPACE skips the interlock, which is how a
    # person makes the yard fail on purpose: two logs into the bay together is
    # what the stop bar is not built for.
    def override(self):
        return editor.is_key_down("SPACE")

    def shut(self):
        if self.encoder is not None:
            sample = self.encoder.latest()
            if sample is not None:
                return abs(sample.position - self.shut_angle) < self.shut_tolerance
        return abs(self.joint.position - self.shut_angle) < self.shut_tolerance

    # What the YardMaster calls to ask for one log.
    def request(self):
        self.wanted += 1
)PY";

    // 2. The bay. Counts arrivals and stamps each one on the simulated clock,
    // which is what makes "logs per second" a number about the yard rather than
    // about the machine rendering it.
    const char* kBayCounterSource = R"PY(# Timber Yard - the saw bay counter.
#
# This rides an invisible TRIGGER volume: a static body with trigger=1 collides
# with nothing and reports what passes through it, which is exactly a counter.
# Every arrival is stamped with threepp.editor.time.sim_time - the SIMULATED
# clock, the same one that stamps sensor samples - so the throughput below is a
# property of the yard and not of the frame rate.
import threepp

editor = getattr(threepp, "editor", None)


class BayCounter:

    cargo = "Log"       # name prefix that counts as cargo
    ignore = "Offcut"   # ...except this half of a sawn one
    flash_seconds = 0.45
    box_x = 1.3         # the volume this script is drawn as, full extents
    box_y = 1.6
    box_z = 2.4

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.count = 0
        self.occupancy = 0    # how many are in the bay RIGHT NOW
        self.first_at = None
        self.last_at = None
        self.arrivals = []
        self.seen = set()     # names already counted - see on_trigger_enter
        self.flash = 0.0

    # A log is TWO bodies (see the saw), and both of them cross this volume
    # whether the blade parted them or not. Counting cargo means counting one
    # named half and skipping the other, or every log arrives twice.
    def counts(self, other: threepp.Object3D):
        return (other is not None and other.name.startswith(self.cargo)
                and self.ignore not in other.name)

    def on_trigger_enter(self, other: threepp.Object3D):
        if not self.counts(other):
            return
        self.occupancy += 1
        # Occupancy is about RIGHT NOW and counts every crossing; the tally is
        # about how many logs the yard has sawn, and a log is one log however
        # many times it crosses. It can cross more than once: the saw is in this
        # volume, and a log being wedged apart by a blade jostles. Latching on
        # the name is what makes "13 logs sawn out of 8" impossible.
        if other.name in self.seen:
            return
        self.seen.add(other.name)
        self.count += 1
        now = editor.time.sim_time if editor is not None else 0.0
        self.arrivals.append(now)
        if self.first_at is None:
            self.first_at = now
        self.last_at = now
        self.flash = self.flash_seconds
        print("Timber Yard: log %d in the bay at %.2f s (sim)" % (self.count, now), flush=True)

    def on_trigger_exit(self, other: threepp.Object3D):
        if not self.counts(other):
            return
        self.occupancy = max(0, self.occupancy - 1)

    # Logs per SIMULATED second, measured across the arrivals themselves: the
    # settle time before the first one is not the yard being slow.
    def throughput(self):
        if self.count < 2 or self.first_at is None:
            return 0.0
        span = self.last_at - self.first_at
        return 0.0 if span <= 0.0 else (self.count - 1) / span

    def update(self, dt: float):
        if self.flash <= 0.0 or editor is None:
            return
        self.flash = max(0.0, self.flash - dt)
        # Debug draw is frame-scoped furniture: called every frame while it
        # lasts, never saved, and invisible to the sensors.
        editor.draw_box(self.obj.position,
                        threepp.Vector3(self.box_x, self.box_y, self.box_z),
                        0xffc23a)
)PY";

    // 3. The stop bar. A load cell in a joint that is allowed to fail.
    const char* kStopBarSource = R"PY(# Timber Yard - the stop bar.
#
# The bar at the end of the bay is held by a BREAKABLE joint, and the joint
# carries a force/torque sensor - the load cell reads PxConstraint::getForce,
# the actual force the solver is spending to hold the bar on. One log leaning on
# it is routine. Two logs arriving together is what the yard is not built for,
# and then on_break() fires: once, on this node, because the script sits on the
# joint.
import threepp

editor = getattr(threepp, "editor", None)


def _verb(name):
    return getattr(editor, name, None) if editor is not None else None


class StopBar:

    master = "Yard Sign"# who is told when this goes
    draw_scale = 0.004  # metres of arrow per newton
    limit = 240.0       # N; roughly what the joint is authored to survive

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.joint = None
        self.load = None
        self.peak = 0.0
        self.newtons = 0.0
        self.broken = False
        self.force = threepp.Vector3(0.0, 0.0, 0.0)
        joint_of = _verb("joint_from_object")
        load_of = _verb("force_torque_from_object")
        if joint_of is not None:
            self.joint = joint_of(obj)
        if load_of is not None:
            self.load = load_of(obj)

    # On the PHYSICS clock, because the load is a physics quantity: reading it
    # per frame would sample it at whatever rate the window happens to run at.
    def fixed_update(self, dt: float):
        if self.load is None:
            return
        sample = self.load.latest()
        if sample is None:
            return
        self.force = sample.force
        self.newtons = (self.force.x ** 2 + self.force.y ** 2 + self.force.z ** 2) ** 0.5
        self.peak = max(self.peak, self.newtons)

    def update(self, dt: float):
        if editor is None:
            return
        # The load, as a line out of the mount: green while it is routine, red
        # once it is over what the joint is authored to take.
        anchor = self.obj.position
        tip = threepp.Vector3(anchor.x + self.force.x * self.draw_scale,
                              anchor.y + self.force.y * self.draw_scale,
                              anchor.z + self.force.z * self.draw_scale)
        hot = self.broken or self.newtons > self.limit
        editor.draw_line(anchor, tip, 0xff3b1f if hot else 0x39e07a)

    # Called once, when the constraint gives way. Only a script ON the joint
    # node gets this.
    def on_break(self):
        self.broken = True
        print("Timber Yard: the stop bar SNAPPED at %.0f N (peak %.0f N)"
              % (self.newtons, self.peak), flush=True)
        if editor is None:
            return
        master = editor.script_from_object(editor.scene().get_object_by_name(self.master))
        if master is not None:
            master.fail("the stop bar snapped - two logs reached the bay together")
)PY";

    // 4. The mission. The second coroutine showcase: a nested generator per log,
    // and a report that prints BOTH clocks.
    const char* kYardMasterSource = R"PY(# Timber Yard - the yard master.
#
# The mission, as a sequence rather than as a state machine. The outer coroutine
# settles, then releases eight logs; releasing ONE is a generator of its own,
# yielded from the outer one - a yielded generator costs no frame of its own, so
# `yield self.release_one(i)` reads like the call it is.
#
# SPACE is the manual override, folded into the until() below: it stops the
# mission waiting for the log that is still on its way and asks for the next one
# now. The gate honours it too, and two logs arriving in the bay together is
# what breaks the stop bar - which is the yard's one failure, and it is a
# person's doing.
import threepp

editor = getattr(threepp, "editor", None)


class YardMaster:

    logs = 8
    gate = "Gate Hinge"
    counter = "Bay Trigger"
    settle_seconds = 1.5# SIM seconds before the first release
    patience = 25.0     # SIM seconds to wait for one log before giving up

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.failed = False
        self.reason = ""
        self.done = False
        self.delivered = 0
        self.task = None
        self.gate_node = None
        self.counter_node = None
        if editor is None:
            return
        scene = editor.scene()
        self.gate_node = scene.get_object_by_name(self.gate)
        self.counter_node = scene.get_object_by_name(self.counter)
        print("Timber Yard: %d logs to run through the saw bay. "
              "Hold SPACE to override the gate interlock." % self.logs, flush=True)
        self.task = editor.start_coroutine(self.mission())

    # --- the mission ---------------------------------------------------------
    def mission(self):
        yield editor.wait(self.settle_seconds)
        for i in range(self.logs):
            if self.failed:
                break
            self.delivered = yield self.release_one(i + 1)
        self.done = True
        self.report()

    # One log: ask the gate for it, then wait for the bay to say it arrived.
    # This is a generator, so the `yield` in the caller resumes with what it
    # returns.
    def release_one(self, index: int):
        gate = editor.script_from_object(self.gate_node)
        counter = editor.script_from_object(self.counter_node)
        if gate is None or counter is None:
            return 0
        target = counter.count + 1
        gate.request()
        deadline = editor.time.sim_time + self.patience
        # SPACE is the override; self.failed and the deadline are the two ways
        # this stops waiting for a log that is not coming.
        yield editor.until(lambda: counter.count >= target
                           or self.failed
                           or editor.is_key_down("SPACE")
                           or editor.time.sim_time > deadline)
        return counter.count

    # Called by the stop bar when the joint gives way.
    def fail(self, reason: str):
        if self.failed:
            return
        self.failed = True
        self.reason = reason
        print("Timber Yard: MISSION FAILED -", reason, flush=True)

    # BOTH clocks, side by side. They do not agree and they never catch up: a
    # frame that hitches advances wall_time in full and simulates at most a few
    # substeps. The throughput is quoted on the simulated one, because that is
    # the one that describes the yard.
    def report(self):
        counter = editor.script_from_object(self.counter_node)
        count = counter.count if counter is not None else 0
        rate = counter.throughput() if counter is not None else 0.0
        t = editor.time
        print("Timber Yard: %d/%d logs sawn%s" %
              (count, self.logs, " (mission failed)" if self.failed else ""), flush=True)
        print("Timber Yard: %.1f s simulated in %.1f s of wall clock (%d substeps), "
              "%.2f logs per simulated second"
              % (t.sim_time, t.wall_time, t.steps, rate), flush=True)
)PY";

    // 5. The wobble. Seeded off the uuids the author tool derived, so the yard
    // is untidy in the SAME way on every run.
    const char* kLogWobbleSource = R"PY(# Timber Yard - the logs are not a diagram.
#
# Eight identical cylinders released down an identical slope roll as one row of
# clones, which is the tell of a scene that was laid out rather than stacked. So
# each log gets a small nudge and a small spin at Play.
#
# Seeded, not random: the seed is a hash of the log's OWN UUID, and this
# document's uuids are derived by the author tool rather than drawn, so every
# run of this scene wobbles identically. That is what lets the headless test
# assert on how many logs arrive.
#
# It is applied as VELOCITY rather than as a pose because a dynamic body's pose
# belongs to the solver: the play session has already built the bodies at their
# authored transforms by the time start() runs, and the handle exposes velocity
# and angular_velocity but no teleport.
import threepp

editor = getattr(threepp, "editor", None)


class LogWobble:

    prefix = "Log"
    count = 8
    nudge = 0.16# m/s along the rack
    drift = 0.0 # ACROSS it: zero, deliberately - see start()
    spin = 0.5  # rad/s of roll

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.wobbled = 0
        if editor is None:
            return
        body_of = getattr(editor, "rigid_body_from_object", None)
        if body_of is None:
            return
        scene = editor.scene()
        for i in range(1, self.count + 1):
            log = scene.get_object_by_name("%s %d" % (self.prefix, i))
            if log is None:
                continue
            a = self.unit(log.uuid, 0)
            b = self.unit(log.uuid, 1)
            c = self.unit(log.uuid, 2)
            # BOTH halves get the SAME push. They are held together by a
            # breakable joint that the saw is supposed to break, and nudging
            # them apart at Play would spend part of its budget before the log
            # has left the rack.
            #
            # And the push is along the rack ONLY. A sideways nudge walks a log
            # off the blade's line, and a log that meets the blade off-centre is
            # a log the saw does not cut - which showed up as four of eight
            # logs reaching the stack in one piece.
            for name in ("%s %d" % (self.prefix, i), "%s %d Offcut" % (self.prefix, i)):
                body = body_of(scene.get_object_by_name(name))
                if body is None:
                    continue
                body.velocity = threepp.Vector3(a * self.nudge, 0.0, b * self.drift)
                body.angular_velocity = threepp.Vector3(0.0, 0.0, c * self.spin)
                self.wobbled += 1

    # FNV-1a over the uuid plus a channel, folded to -1..1. A hash rather than
    # random.seed() so the channels of one log are independent without three
    # generators.
    def unit(self, text: str, channel: int):
        h = 2166136261 ^ channel
        for ch in text:
            h = ((h ^ ord(ch)) * 16777619) & 0xffffffff
        return (h & 0xffff) / 32767.5 - 1.0
)PY";


    // 6. The saw. Watches the cuts it makes, and flashes where it made them.
    const char* kSawMillSource = R"PY(# Timber Yard - the saw.
#
# The blade does not do the cutting in code: it is a driven body on a revolute
# joint (a motor - zero stiffness, a velocity target through damping) and it
# meets each log on the inner faces of its two halves, which is a wedge. What
# this script does is WATCH: every log carries a breakable fixed joint holding
# its halves together, and a joint that has gone is a log that has been sawn.
#
# It reads the spindle too. A saw fighting a log slows down, and because the
# motor is a real drive against a real contact, that dip is physics rather than
# an animation somebody keyframed.
import threepp

editor = getattr(threepp, "editor", None)


def _verb(name):
    return getattr(editor, name, None) if editor is not None else None


class SawMill:

    logs = 8
    spindle = "Saw Spindle"
    flash_seconds = 0.3
    x = -1.5 # where the cut happens, in world terms: this script's object is a
    y = 1.05 # child of the gantry, so its own position is not the answer

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.cuts = 0
        self.flash = 0.0
        self.speed = 0.0
        self.pending = []
        self.joint = None
        if editor is None:
            return
        joint_of = _verb("joint_from_object")
        if joint_of is None:
            print("Timber Yard needs the PhysX build to run the saw.", flush=True)
            return
        scene = editor.scene()
        self.joint = joint_of(scene.get_object_by_name(self.spindle))
        for i in range(1, self.logs + 1):
            node = scene.get_object_by_name("Log %d Cut" % i)
            cut = joint_of(node) if node is not None else None
            if cut is not None:
                self.pending.append([i, cut])

    def update(self, dt: float):
        if editor is None:
            return
        if self.joint is not None:
            self.speed = abs(self.joint.velocity)

        # A joint that has gone is a log that has been sawn. Rebuilt rather than
        # removed from in place: mutating the list being walked is how you miss
        # every second cut.
        still = []
        for index, cut in self.pending:
            if cut.broken:
                self.cuts += 1
                self.flash = self.flash_seconds
                print("Timber Yard: log %d cut (blade at %.0f rpm)"
                      % (index, self.speed * 9.5493), flush=True)
            else:
                still.append([index, cut])
        self.pending = still

        if self.flash <= 0.0:
            return
        # Sawdust, for the frame it lasts: debug draw is furniture, never saved
        # and invisible to the sensors.
        self.flash = max(0.0, self.flash - dt)
        level = self.flash / self.flash_seconds
        centre = threepp.Vector3(self.x, self.y, 0.0)
        editor.draw_box(centre, threepp.Vector3(0.5, 0.5, 0.9), 0xffd27a)
        for i in range(6):
            a = (i * 1.0472) + self.cuts
            spread = 0.35 + 0.5 * level
            editor.draw_line(centre,
                             threepp.Vector3(self.x + spread * 0.7,
                                             self.y + spread * (0.5 - 0.2 * i),
                                             spread * (0.5 - 0.2 * (i % 3))),
                             0xffbf5a)
)PY";

    // ------------------------------------------------------------------- scene

    // --- scenery, and why it is not generated --------------------------------
    //
    // The obvious way to stand a forest and a yard office around this scene is
    // the two generators the editor already ships: TreeConfig (Add ▸ Tree) and
    // architecture::createLogCabin. Both were built, exported and MEASURED
    // here, and neither can be in this document, because a generator emits a
    // BufferGeometry and the serializer writes every vertex of one as text:
    //
    //   one pine, the editor's own Pine/spruce preset ........  980 KB
    //   the same, turned down until it is barely a tree ....... ~140 KB
    //       (of which the branch tubes are 112 KB — that is the FLOOR,
    //        the foliage is only 26 KB of it)
    //   a 6.4 x 4.6 m log cabin, flat materials ............... 4.61 MB
    //   the smallest cabin that is still a building .......... ~985 KB
    //
    // Both were also SHARED and re-measured, which is the cheap fix and the
    // right instinct: the serializer writes a geometry once per uuid however
    // many meshes point at it, so six trees off one generated pine cost what
    // one does. It did not save them. One pine turned down until it fits is a
    // bare stick with two blobs on it; turned back up until it reads as a tree
    // it is 340 KB, shared, for SIX TREES THAT ARE THE SAME TREE — and the
    // photographs of both are worse than what is below. (The other suspect, the
    // procedural belt texture, turned out to cost 374 bytes. Geometry is the
    // whole story, exactly as the serializer's notes say.)
    //
    // So the clearing edge and the office are built from PRIMITIVES, which
    // serialize as their PARAMETERS: a cone is nine numbers whatever it is
    // tessellated to. Everything below shares three geometries between every
    // instance, so six trees cost about what one does — and they read as
    // conifers, which the generated ones at this budget did not.

    // Geometry shared by all the scenery, made once (a geometry the document
    // carries twice is a geometry the document carries twice).
    struct Scenery {
        std::shared_ptr<BufferGeometry> box;
        std::shared_ptr<BufferGeometry> cone;
        std::shared_ptr<BufferGeometry> trunk;
        std::shared_ptr<MeshStandardMaterial> bark;
        std::shared_ptr<MeshStandardMaterial> needle;
        std::shared_ptr<MeshStandardMaterial> plank;
        std::shared_ptr<MeshStandardMaterial> roof;
        std::shared_ptr<MeshStandardMaterial> stone;
    };

    // A spruce: a tapered bole and three skirts of needles, narrowing upward.
    // Nothing here is random — the caller varies height, girth, lean and spin,
    // which is what keeps six of them from reading as one tree stamped six
    // times.
    std::shared_ptr<Group> makeSpruce(const Scenery& shared, const std::string& name,
                                      float x, float z, float height, float girth,
                                      float spin, float lean) {

        auto tree = Group::create();
        tree->name = name;
        tree->position.set(x, 0.f, z);
        tree->rotation.y = spin;
        tree->rotation.z = lean;

        auto bole = mesh("Bole", shared.trunk, shared.bark);
        bole->position.set(0.f, height * 0.24f, 0.f);
        bole->scale.set(girth, height * 0.48f, girth);
        tree->add(bole);

        // Skirts overlap by a third of their height, or the tree is three hats
        // on a stick. The lowest is the widest and they lose a fifth each time.
        for (int tier = 0; tier < 3; ++tier) {
            const float t = static_cast<float>(tier);
            const float radius = girth * (3.4f - 0.72f * t);
            const float tall = height * (0.42f - 0.06f * t);
            auto skirt = mesh("Needles " + std::to_string(tier + 1), shared.cone, shared.needle);
            skirt->position.set(0.f, height * (0.30f + 0.21f * t), 0.f);
            skirt->scale.set(radius, tall, radius);
            // A sixth of a turn per tier, so the facets of one skirt never line
            // up with the facets of the one under it.
            skirt->rotation.y = 0.52f * t;
            tree->add(skirt);
        }
        return tree;
    }

    // The yard office: a plank shed with a pitched roof, a stove pipe and a
    // stack of cut timber against its gable. Boxes and one cylinder, so it
    // costs about three kilobytes.
    std::shared_ptr<Group> makeYardOffice(const Scenery& shared) {

        auto office = Group::create();
        office->name = "Yard Office";
        office->position.set(-8.2f, 0.f, -7.6f);
        office->rotation.y = 0.42f;

        auto part = [&](const std::string& name, const std::shared_ptr<Material>& material,
                        float px, float py, float pz, float sx, float sy, float sz,
                        float roll = 0.f) {
            auto piece = mesh(name, shared.box, material);
            piece->position.set(px, py, pz);
            piece->scale.set(sx, sy, sz);
            piece->rotation.z = roll;
            office->add(piece);
            return piece;
        };

        part("Office Footing", shared.stone, 0.f, 0.14f, 0.f, 4.9f, 0.28f, 3.7f);
        part("Office Walls", shared.plank, 0.f, 1.38f, 0.f, 4.5f, 2.2f, 3.3f);
        // Two planes meeting over a ridge that runs along Z, plus the gable
        // triangle's stand-in: a thin board under the ridge.
        part("Office Roof L", shared.roof, -1.18f, 2.86f, 0.f, 2.7f, 0.14f, 3.9f, 0.42f);
        part("Office Roof R", shared.roof, 1.18f, 2.86f, 0.f, 2.7f, 0.14f, 3.9f, -0.42f);
        part("Office Gable", shared.plank, 0.f, 2.72f, 0.f, 0.9f, 0.72f, 3.32f);
        part("Office Door", shared.roof, 0.f, 1.05f, 1.67f, 0.95f, 2.0f, 0.08f);
        part("Office Window", shared.stone, -1.5f, 1.75f, 1.67f, 0.85f, 0.65f, 0.08f);

        auto pipe = mesh("Office Flue", shared.trunk, shared.stone);
        pipe->position.set(1.1f, 3.35f, -0.9f);
        pipe->scale.set(0.1f, 1.3f, 0.1f);
        office->add(pipe);

        return office;
    }

    // The belt. Three waypoints on one straight line at working height, and a
    // config turned down to what the yard needs: no cleats, no separator, and
    // few samples per segment, because every generated part is geometry the
    // document has to carry.
    std::shared_ptr<Group> makeConveyor() {

        auto conveyor = Group::create();
        conveyor->name = "Conveyor";

        ConveyorConfig config;
        config.width = 1.5f;
        config.speed = 1.15f;
        config.smooth = false;// a straight run; nothing to interpolate
        config.samples = 4;
        config.frame = true;// rails, legs and the end drums: the machine's body
        config.write(*conveyor);

        static constexpr float waypoints[][3] = {
                {kBeltStartX, kBeltY, 0.f},
                {0.f, kBeltY, 0.f},
                {kBeltEndX, kBeltY, 0.f}};
        int index = 1;
        for (const auto& position : waypoints) {
            auto point = Object3D::create();
            point->name = "Waypoint " + std::to_string(index++);
            point->position.set(position[0], position[1], position[2]);
            conveyor->add(point);
        }

        // The generated parts, exactly as the editor's sync pass would build
        // them — the document ships the machine, not a recipe for it.
        config.syncDerived(*conveyor);
        return conveyor;
    }

    std::shared_ptr<Scene> buildScene() {

        auto scene = Scene::create();
        scene->name = "Scene";
        scene->background = Color(kSky);
        // Far enough out that the pines at the clearing edge are still trees and
        // the ones beyond them are weather.
        scene->fog = Fog(Color(kSky), 34.f, 145.f);

        // --- how it opens ----------------------------------------------------
        // The yard is a LINE, from the rack at -X to the stop bar at +X, so the
        // vantage is square onto it: everything that happens, happens left to
        // right across the frame. High enough to see the belt's deck (a view
        // level with a conveyor is a view of its side rail) and far enough back
        // that the rack and the bay are both in shot.
        //
        // It is also the follow offset, since Follow keeps whatever offset the
        // camera has when the document opens — which is why what it follows is
        // the GATE HINGE and not a log. Following the cargo was tried and
        // photographed: the camera goes down the yard with it and parks at the
        // far end looking at a stack, and the machine that is the point of the
        // scene is behind you. The hinge does not move, so the authored framing
        // HOLDS through Play — and opening the document lands the selection on
        // the joint, with its Joint, Sensor and Script sections already up.
        scene->userData["editorView"] = std::string("-4,9.5,21@-0.5,1.2,0");
        scene->userData["editorFollow"] = std::string("Gate Hinge");

        // --- lights ----------------------------------------------------------
        auto sun = DirectionalLight::create(0xfff2dc, 3.1f);
        sun->name = "Sun";
        // The authored vantage looks from +Z, so the sun stands on that side and
        // a little downstream: the faces the camera sees are lit, and the
        // shadows fall away from it across the yard instead of toward it.
        sun->position.set(11.f, 20.f, 17.f);
        sun->castShadow = true;
        // The default shadow frustum is +-1, which over a thirty-metre yard
        // means no shadows at all. It round-trips (ObjectExporter writes the
        // shadow camera), so widening it is a property of the document.
        if (auto* shadowCamera = dynamic_cast<OrthographicCamera*>(sun->shadow->camera.get())) {
            shadowCamera->left = -22.f;
            shadowCamera->right = 22.f;
            shadowCamera->top = 22.f;
            shadowCamera->bottom = -22.f;
            shadowCamera->nearPlane = 1.f;
            shadowCamera->farPlane = 80.f;
            shadowCamera->updateProjectionMatrix();
        }
        sun->shadow->mapSize.set(2048, 2048);
        sun->shadow->bias = -0.0006f;
        scene->add(sun);

        auto sky = HemisphereLight::create(0x9db8d4, 0x4a4a34, 1.1f);
        sky->name = "Sky Fill";
        scene->add(sky);

        auto ambient = AmbientLight::create(0x6a7060, 0.35f);
        ambient->name = "Ambient Light";
        scene->add(ambient);

        // --- the clearing ----------------------------------------------------
        // One box, one cone and one tapered cylinder, shared by everything that
        // can be made of them: a geometry the document carries once is a
        // geometry the document carries once.
        Scenery shared;
        shared.box = BoxGeometry::create(1.f, 1.f, 1.f);
        shared.cone = ConeGeometry::create(1.f, 1.f, 7);
        shared.trunk = CylinderGeometry::create(0.62f, 1.f, 1.f, 6);
        shared.bark = standard(kBark, 0.95f, 0.f);
        shared.needle = standard(kNeedle, 0.92f, 0.f);
        shared.plank = standard(0x8a6a44, 0.9f, 0.f);
        shared.roof = standard(0x51443a, 0.88f, 0.f);
        shared.stone = standard(0x6f6c66, 0.9f, 0.05f);

        auto unit = shared.box;
        auto groundMaterial = standard(kGround, 0.97f, 0.f);
        auto ground = mesh("Ground", unit, groundMaterial);
        ground->position.set(0.f, -0.5f, 0.f);
        ground->scale.set(90.f, 1.f, 90.f);
        ground->castShadow = false;
        writePhysics(*ground, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.7f, 0.f);
        scene->add(ground);

        // The working surface: packed earth under the machines, so the yard
        // reads as a place that is used rather than as a belt in a meadow.
        auto yard = mesh("Yard Floor", unit, standard(kYardDirt, 0.95f, 0.f));
        yard->position.set(-3.4f, 0.03f, 0.f);
        yard->scale.set(23.f, 0.06f, 11.f);
        yard->castShadow = false;
        scene->add(yard);

        // --- the log rack ----------------------------------------------------
        auto rack = Group::create();
        rack->name = "Log Rack";
        scene->add(rack);

        auto timberMaterial = standard(kTimber, 0.9f, 0.f);
        auto steelMaterial = standard(kSteel, 0.55f, 0.6f);

        auto deck = mesh("Rack Deck", unit, timberMaterial);
        deck->position.set(kRackCentreX, kRackCentreY, 0.f);
        deck->scale.set(kRackLength, 2.f * kDeckHalf, 2.f * kRackHalfDepth);
        deck->rotation.z = -kSlope;
        writePhysics(*deck, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.32f, 0.f);
        rack->add(deck);

        // Kerbs, so a log that wanders sideways comes back instead of leaving.
        // They STOP SHORT of the gate (kKerbEndX): the leaf swings down between
        // them, and a kerb under it is a kerb the leaf jams against.
        {
            const float kerbLength = kKerbEndX - (kRackCentreX - kRackLength * 0.5f);
            const float kerbCentreX = kRackCentreX - kRackLength * 0.5f + kerbLength * 0.5f;
            for (int side = 0; side < 2; ++side) {
                const float z = (side == 0) ? kRackHalfDepth : -kRackHalfDepth;
                auto kerb = mesh("Rack Kerb " + std::to_string(side + 1), unit, timberMaterial);
                kerb->position.set(kerbCentreX,
                                   kRackCentreY - (kerbCentreX - kRackCentreX) * std::tan(kSlope) + 0.24f,
                                   z);
                kerb->scale.set(kerbLength, 0.3f, 0.16f);
                kerb->rotation.z = -kSlope;
                writePhysics(*kerb, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                             1.f, 0.3f, 0.f);
                rack->add(kerb);
            }
        }

        // Legs. No physics: nothing in the yard can reach them, and a collider
        // is not scenery.
        for (int i = 0; i < 4; ++i) {
            const float x = kRackCentreX + ((i & 1) ? 2.9f : -2.9f);
            const float z = (i & 2) ? kRackHalfDepth - 0.1f : -kRackHalfDepth + 0.1f;
            const float top = rackTopY(x) - 2.f * kDeckHalf;
            auto leg = mesh("Rack Leg " + std::to_string(i + 1), unit, steelMaterial);
            leg->position.set(x, top * 0.5f, z);
            leg->scale.set(0.16f, top, 0.16f);
            rack->add(leg);
        }

        // The rack is also where the wobble lives: it is the thing that owns
        // the logs, and a script has to sit on something.
        {
            ScriptConfig config;
            config.source = kLogWobbleSource;
            config.setField("count", std::to_string(kLogCount));
            config.write(*rack);
        }

        // --- the logs --------------------------------------------------------
        // Cylinders lying ACROSS the yard, so they roll down the rack and along
        // the belt the way a log does — and each one is TWO of them, jointed,
        // waiting for the saw (see kCutSlot).
        auto logGeometry = CylinderGeometry::create(kLogRadius, kLogRadius, kHalfLength, 12, 1);
        // Deliberately lighter than the tree bark that shares the palette: this
        // is SAWN timber, it is the thing the whole scene is about, and eight
        // dark logs on a dark rack are one brown mass at yard distance.
        auto logMaterial = standard(kLogSkin, 0.85f, 0.f);
        for (int i = 0; i < kLogCount; ++i) {
            const float x = logX(i);
            const float y = rackTopY(x) + kLogRadius;
            const auto label = std::to_string(i + 1);

            // The half that gets counted, and the half that does not. The names
            // ARE the convention: the bay counts "Log ..." and skips anything
            // matching its `ignore` field, so a sawn log is still one log.
            auto half = [&](const std::string& name, float z) {
                auto piece = mesh(name, logGeometry, logMaterial);
                piece->position.set(x, y, z);
                // The cylinder's axis is +Y; a quarter turn about X lays it
                // along Z.
                piece->rotation.x = math::PI / 2.f;
                // CAPSULE, not the convex hull Auto would cook. A cylinder's
                // hull is a twelve-sided PRISM, and a prism on a 6.6-degree
                // slope does not roll — it sits in a facet, because rolling over
                // a corner means climbing fifteen degrees. That is a log rack
                // that never releases anything, and it looks exactly like a
                // scripting bug. The capsule's cross-section is a circle, so it
                // rolls like the log it draws.
                // Zero restitution. A log that bounces off a blade is a log
                // being thrown, and the halves have to keep the belt's momentum
                // through the cut, not gain some of the blade's.
                writePhysics(*piece, PhysicsConfig::Body::Dynamic,
                             PhysicsConfig::Shape::Capsule, 13.f, 0.55f, 0.f);
                scene->add(piece);
                return piece;
            };
            auto front = half("Log " + label, kHalfOffset);
            half("Log " + label + " Offcut", -kHalfOffset);

            // The cut, waiting to happen. Its parent chain is body A (the
            // counted half) and body B is the offcut by name; the anchor sits on
            // the seam between them.
            //
            // The node's position is in its PARENT'S frame, and the parent is
            // lying on its side — a quarter turn about X sends local +Y to world
            // -Z, so the seam is at local -kHalfOffset in Y, not in Z.
            auto cut = Object3D::create();
            cut->name = "Log " + label + " Cut";
            cut->position.set(0.f, -kHalfOffset, 0.f);
            {
                JointConfig joint;
                joint.type = JointConfig::Type::Fixed;
                joint.body = "Log " + label + " Offcut";
                // MEASURED, both directions — see the stop bar's note for why
                // that has to be per substep. The ride is what it has to
                // survive: the drop off the rack onto the belt is the loudest
                // thing that happens to a log before the saw.
                // MEASURED per substep with the joint made unbreakable, which is
                // the only way to see the whole curve: everything that happens
                // to a log BEFORE the saw — the pack leaning on the gate, the
                // hold-back clamp, the drop off the rack onto the belt — peaks
                // at 493 N, and the blade's wedge drives this joint to 1155 N.
                // 650 sits between, and the OUTCOME is what settled it (a probe
                // reads zero the instant a joint goes, so the spike that breaks
                // one is the spike it never sees): at 650 all eight logs are
                // cut, at 700 six are and the two that get through intact hit
                // the flap hard enough to matter.
                //
                // That margin is a THIN BLADE's margin, and it is the right
                // trade. A fat collider bites far harder (2645 N at 18 cm) but
                // the solver then has a deep overlap to push out in the substep
                // the joint happens to snap, and the halves leave with it —
                // across the belt, sometimes off it. Cutting is not throwing.
                joint.breakForce = 650.f;
                joint.breakTorque = 600.f;
                joint.write(*cut);
            }
            front->add(cut);
        }

        // --- the gate --------------------------------------------------------
        // The post is the joint's OTHER body, referenced by name, so it is one
        // named static mesh — the crossbeam — with its legs hanging off it as
        // decoration.
        auto post = mesh("Gate Post", unit, steelMaterial);
        post->position.set(kGateX, kHingeY + 0.18f, 0.f);
        post->scale.set(0.18f, 0.18f, 2.9f);
        writePhysics(*post, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.5f, 0.f);
        scene->add(post);

        for (int side = 0; side < 2; ++side) {
            const float z = (side == 0) ? 1.35f : -1.35f;
            auto leg = mesh("Gate Post Leg " + std::to_string(side + 1), unit, steelMaterial);
            leg->position.set(0.f, -(kHingeY + 0.18f) * 0.5f, z / 2.9f);
            leg->scale.set(1.f, (kHingeY + 0.18f) / 0.18f, 0.062f);
            post->add(leg);
        }

        // The leaf: a plate hanging from the hinge, blocking the rack's outfall.
        // Its bottom edge clears the deck under it by about 15 mm — enough that
        // it never touches, far too little for a 400 mm log to pass.
        auto gate = mesh("Gate", unit, standard(kOrange, 0.7f, 0.15f));
        gate->position.set(kGateX, kHingeY - kGateDrop * 0.5f, 0.f);
        gate->scale.set(0.1f, kGateDrop, kGateWidth);
        writePhysics(*gate, PhysicsConfig::Body::Dynamic, PhysicsConfig::Shape::Box,
                     18.f, 0.4f, 0.02f);
        scene->add(gate);

        // The hinge. Its parent chain is body A (the gate), body B is the post
        // by name, and the NODE'S TRANSFORM IS THE JOINT FRAME: the anchor at
        // its origin and the hinge axis along its local X, which the quarter
        // turn about Y lays along world +Z — across the yard, which is the only
        // axis a gate over a belt can swing about.
        auto hinge = Object3D::create();
        hinge->name = "Gate Hinge";
        hinge->position.set(0.f, kGateDrop * 0.5f, 0.f);
        hinge->rotation.y = -math::PI / 2.f;
        {
            JointConfig joint;
            joint.type = JointConfig::Type::Revolute;
            joint.body = "Gate Post";
            joint.limited = true;
            // Positive is UPSTREAM, into the logs (the joint frame's X is world
            // +Z, and this was measured, not assumed). So the leaf opens
            // NEGATIVE — downstream, over the belt, away from the pack — and the
            // shut end of the travel is pinned just past zero.
            joint.lower = -1.6f;
            joint.upper = 0.04f;
            // Force mode. Stiffness holds the target, damping holds the rate:
            // the leaf has eight logs leaning on it and must not creep, and it
            // must not slam either.
            joint.stiffness = 9000.f;
            joint.damping = 900.f;
            joint.maxForce = 3e5f;
            joint.target = 0.f;
            joint.write(*hinge);
        }
        {
            // The encoder reads THIS joint — a joint sensor on a joint node
            // needs no joint name, the node is the reference.
            SensorConfig encoder;
            encoder.enabled = true;
            encoder.type = SensorConfig::Type::Encoder;
            encoder.rateHz = 120.f;
            encoder.seed = 7;
            encoder.encoderResolution = 0.002f;// rad per tick: a real disc
            encoder.write(*hinge);
        }
        {
            ScriptConfig config;
            config.source = kGateKeeperSource;
            config.setField("counter", "Bay Trigger");
            config.write(*hinge);
        }
        gate->add(hinge);

        // --- the hold-back clamp ---------------------------------------------
        // The other half of the escapement (see the note at the top). Authored
        // LIFTED, so the document opens with nothing touching and the drive has
        // somewhere to push.
        // The head the arm hangs off — a static crossbeam, and the joint's other
        // body. Its legs are decoration; nothing in the yard can reach them.
        auto clampHead = mesh("Clamp Head", unit, steelMaterial);
        clampHead->position.set(kClampPivotX - 0.22f, kClampPivotY, 0.f);
        clampHead->scale.set(0.18f, 0.18f, 2.8f);
        writePhysics(*clampHead, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.5f, 0.f);
        scene->add(clampHead);

        for (int side = 0; side < 2; ++side) {
            const float z = (side == 0) ? 1.32f : -1.32f;
            auto leg = mesh("Clamp Post " + std::to_string(side + 1), unit, steelMaterial);
            leg->position.set(kClampPivotX - 0.22f, kClampPivotY * 0.5f, z);
            leg->scale.set(0.14f, kClampPivotY, 0.14f);
            scene->add(leg);
        }

        // The arm, lying level along +X from its pivot: above every log and
        // clear of the leaf. Swinging it down puts its far end on the second
        // log's back, one diameter behind the one being released.
        auto clamp = mesh("Holdback Arm", unit, standard(kOrange, 0.7f, 0.15f));
        clamp->position.set(kClampPivotX + kClampArm * 0.5f, kClampPivotY, 0.f);
        clamp->scale.set(kClampArm, 0.13f, 1.1f);
        writePhysics(*clamp, PhysicsConfig::Body::Dynamic, PhysicsConfig::Shape::Box,
                     9.f, 0.8f, 0.02f);
        scene->add(clamp);

        auto clampHinge = Object3D::create();
        clampHinge->name = "Holdback Hinge";
        // The pivot is the arm's upstream end, expressed in the ARM'S OWN scaled
        // frame — a child's position is in its parent's units, and the parent is
        // a scaled box.
        clampHinge->position.set(-0.5f, 0.f, 0.f);
        clampHinge->rotation.y = -math::PI / 2.f;
        {
            JointConfig joint;
            joint.type = JointConfig::Type::Revolute;
            joint.body = "Clamp Head";
            joint.limited = true;
            // Zero is the authored, lifted pose; positive swings the pad down
            // onto the log. The upper limit is past where the log stops it, so
            // the drive always has an error to push with.
            joint.lower = -0.05f;
            joint.upper = 1.15f;
            joint.stiffness = 2200.f;
            joint.damping = 260.f;
            joint.maxForce = 6e4f;
            joint.target = 0.f;
            joint.write(*clampHinge);
        }
        clamp->add(clampHinge);

        // --- the belt --------------------------------------------------------
        scene->add(makeConveyor());

        // Kerb rails down both edges of the belt. The logs are already shorter
        // than the belt is wide, but "already" is not "cannot": a log that
        // wanders sideways far enough to hang an end over the edge presents that
        // end to the blade PLANE instead of its middle, and then the saw mills
        // it side-on for ever instead of cutting it. The rails are low (they
        // clear a log's centre line, so nothing rides up them) and slippery, so
        // the belt keeps driving while a log slides straight again.
        for (int side = 0; side < 2; ++side) {
            const float z = (side == 0) ? 0.83f : -0.83f;
            auto kerb = mesh("Belt Kerb " + std::to_string(side + 1), unit,
                             standard(kSteel, 0.6f, 0.5f));
            // Tall enough to reach over a half-log's CENTRE (a rail lower than
            // that is a ramp), and the halves ride at |z| ~ 0.3, so there is
            // half a metre of clearance either side of the cut.
            // They start downstream of the GATE and run to the end of the belt.
            // Tall enough to foul the gate leaf's swing if they began any
            // earlier — a gate that cannot open delivers nothing, which is
            // exactly how that presented — and the flap at the far end is
            // narrower than the gap between them, so it swings between them
            // rather than into them.
            const float from = kBeltStartX + 0.7f;
            const float to = kBeltEndX;
            // As tall as a log, not as tall as half of one. A rail that stops
            // at a log's centre is a ramp for anything that tumbles, and the
            // one body that ever left this belt went over exactly such a rail.
            kerb->position.set((from + to) * 0.5f, kBeltY + 0.2f, z);
            kerb->scale.set(to - from, 0.4f, 0.1f);
            kerb->castShadow = false;
            writePhysics(*kerb, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                         1.f, 0.08f, 0.f);
            scene->add(kerb);
        }

        // --- the discharge apron ---------------------------------------------
        // What the belt hands the logs to: a shallow ramp they roll down and
        // stack on. Low friction, because a stack that grips is a stack that
        // never settles.
        {
            const float length = kApronEndX - kApronStartX;
            const float centreX = (kApronStartX + kApronEndX) * 0.5f;
            auto apron = mesh("Discharge Apron", unit, standard(kYardDirt, 0.9f, 0.f));
            apron->position.set(centreX, apronTopY(centreX) - 0.12f, 0.f);
            apron->scale.set(length, 0.24f, 2.8f);
            apron->rotation.z = -kApronSlope;
            apron->castShadow = false;
            // Slippery on purpose. A log that has just been through the saw is
            // no longer rolling straight — it has been shoved sideways by the
            // blade — and a half that slides instead of rolling stops dead on a
            // two-degree apron with ordinary friction, in a heap by the flap.
            writePhysics(*apron, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                         1.f, 0.12f, 0.f);
            scene->add(apron);

            for (int side = 0; side < 2; ++side) {
                const float z = (side == 0) ? 1.5f : -1.5f;
                auto rail = mesh("Apron Rail " + std::to_string(side + 1), unit,
                                 standard(kTimber, 0.9f, 0.f));
                rail->position.set(centreX, apronTopY(centreX) + 0.11f, z);
                rail->scale.set(length, 0.22f, 0.14f);
                rail->rotation.z = -kApronSlope;
                writePhysics(*rail, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                             1.f, 0.3f, 0.f);
                scene->add(rail);
            }

            // What the stack leans on. A capsule on a flat deck has nothing to
            // stop it — PhysX gives a rolling cylinder no rolling resistance to
            // speak of — so without this the delivered logs roll off the end of
            // the yard and keep going.
            auto berm = mesh("Stack Berm", unit, standard(kTimber, 0.9f, 0.f));
            berm->position.set(kBermX, apronTopY(kBermX) + 0.25f, 0.f);
            berm->scale.set(0.3f, 0.7f, 2.8f);
            writePhysics(*berm, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                         1.f, 0.6f, 0.f);
            scene->add(berm);
        }

        // --- the saw bay -----------------------------------------------------
        // The counting volume: invisible, static, trigger=1. Invisible is a
        // RENDERING decision — the play session walks with traverse(), so it is
        // cooked and reporting all the same.
        auto bay = mesh("Bay Trigger", unit, standard(0xffc23a, 0.5f, 0.f));
        bay->position.set(kBayX, kBeltY + 0.35f, 0.f);
        bay->scale.set(1.3f, 1.6f, 2.4f);
        bay->visible = false;
        bay->castShadow = false;
        bay->receiveShadow = false;
        writePhysics(*bay, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.5f, 0.f, /*trigger*/ true);
        {
            ScriptConfig config;
            config.source = kBayCounterSource;
            config.setField("ignore", "Offcut");
            config.setField("box_x", "1.3");
            config.setField("box_y", "1.6");
            config.setField("box_z", "2.4");
            config.write(*bay);
        }
        scene->add(bay);

        // What makes it a saw bay rather than a labelled box: a gantry over the
        // belt with a blade hanging in it. Decoration, and deliberately clear of
        // the cargo.
        auto gantry = Group::create();
        gantry->name = "Saw Gantry";
        gantry->position.set(kBayX, 0.f, 0.f);
        for (int side = 0; side < 2; ++side) {
            // Outboard of the apron (half-width 1.4), so the saw straddles the
            // discharge instead of standing in it.
            const float z = (side == 0) ? 1.7f : -1.7f;
            auto leg = mesh("Gantry Leg " + std::to_string(side + 1), unit, steelMaterial);
            leg->position.set(0.f, 1.1f, z);
            leg->scale.set(0.16f, 2.2f, 0.16f);
            gantry->add(leg);
        }
        auto beam = mesh("Gantry Beam", unit, steelMaterial);
        beam->position.set(0.f, 2.28f, 0.f);
        beam->scale.set(0.22f, 0.22f, 3.7f);
        gantry->add(beam);
        // The blade is a REAL BODY on a driven hinge, not a prop that turns:
        // a revolute joint with zero stiffness and a velocity target is a motor
        // (target acts through stiffness, velocity through damping), so the
        // blade spins because it is driven, meets the log because it is a
        // collider, and visibly BOGS DOWN when it is fighting one.
        auto blade = mesh("Saw Blade",
                          CylinderGeometry::create(kBladeRadius, kBladeRadius,
                                                   kBladeThickness, 24, 1),
                          standard(0xc9ccd2, 0.35f, 0.85f));
        blade->position.set(0.f, kBladeY, 0.f);
        blade->rotation.x = math::PI / 2.f;// the disc's axis lies along Z
        writePhysics(*blade, PhysicsConfig::Body::Dynamic, PhysicsConfig::Shape::Auto,
                     8.f, 0.4f, 0.f);
        gantry->add(blade);

        auto arbor = mesh("Saw Motor", unit, standard(0x3f4650, 0.6f, 0.4f));
        arbor->position.set(0.f, 2.02f, -0.36f);
        arbor->scale.set(0.44f, 0.42f, 0.5f);
        writePhysics(*arbor, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.5f, 0.f);
        gantry->add(arbor);

        // The hinge. Its axis is the node's local X, and the blade it hangs off
        // is already lying on its side — so the quarter turn that lands local X
        // on world Z (the blade's own axis) is about the node's Z, not its Y.
        auto spindle = Object3D::create();
        spindle->name = "Saw Spindle";
        spindle->rotation.z = math::PI / 2.f;
        {
            JointConfig joint;
            joint.type = JointConfig::Type::Revolute;
            joint.body = "Saw Motor";
            joint.stiffness = 0.f;// no position to hold: this is a motor
            // 26 rad/s on a 0.46 m disc is a RIM SPEED OF 12 m/s, and a rim that
            // fast in sustained contact with a log that is barely moving does
            // not cut it, it throws it — which is what happened to the last log
            // of a run, the one with no pack behind it to push it through. Ten
            // rad/s is 4.6 m/s at the rim: still four times belt speed, still
            // obviously spinning, and it drags rather than launches. The cut
            // comes from the wedge, not from the spin.
            joint.damping = 40.f;
            joint.velocity = 10.f;// rad/s, about 95 rpm
            joint.maxForce = 8e3f;
            joint.write(*spindle);
        }
        blade->add(spindle);

        {
            ScriptConfig config;
            config.source = kSawMillSource;
            config.setField("logs", std::to_string(kLogCount));
            config.setField("x", "-1.5");
            config.write(*blade);
        }

        scene->add(gantry);

        // --- the stop bar ----------------------------------------------------
        // A flap across the end of the belt, hung from a gantry of its own and
        // held shut by a WEAK drive: weak enough that one belt-driven log leans
        // it open and walks through, stiff enough that it swings back and stops
        // the next one. Everything the yard is afraid of is measured here.
        auto stopPost = mesh("Stop Post", unit, steelMaterial);
        stopPost->position.set(kBarX, kBarHingeY + 0.16f, 0.f);
        stopPost->scale.set(0.18f, 0.18f, 2.9f);
        writePhysics(*stopPost, PhysicsConfig::Body::Static, PhysicsConfig::Shape::Box,
                     1.f, 0.5f, 0.f);
        scene->add(stopPost);

        for (int side = 0; side < 2; ++side) {
            const float z = (side == 0) ? 1.4f : -1.4f;
            auto leg = mesh("Stop Post Leg " + std::to_string(side + 1), unit, steelMaterial);
            leg->position.set(0.f, -(kBarHingeY + 0.16f) * 0.5f / 0.18f, z / 2.9f);
            leg->scale.set(1.f, (kBarHingeY + 0.16f) / 0.18f, 0.055f);
            stopPost->add(leg);
        }

        auto stopBar = mesh("Stop Bar", unit, standard(kOrange, 0.65f, 0.2f));
        stopBar->position.set(kBarX, kBarHingeY - kBarDrop * 0.5f, 0.f);
        // NARROWER THAN THE KERBS ARE APART (they stand at z = +-0.83), so the
        // rails can run the full length of the belt and past the flap instead of
        // stopping half a metre short of it. That gap was where the one body
        // that ever left this belt left it: right at the flap, at x = 1.36,
        // where nothing was holding the sides.
        stopBar->scale.set(0.1f, kBarDrop, 1.4f);
        writePhysics(*stopBar, PhysicsConfig::Body::Dynamic, PhysicsConfig::Shape::Box,
                     10.f, 0.45f, 0.02f);
        scene->add(stopBar);

        // The mount: a breakable revolute joint with the load cell authored on
        // the same node. One log leaning on the flap is routine — it opens, the
        // log passes, it swings back. Two logs in contact push with twice the
        // belt behind them, and that is over the threshold.
        auto mount = Object3D::create();
        mount->name = "Stop Bar Mount";
        mount->position.set(0.f, 0.5f, 0.f);// the flap's top edge, in ITS units
        mount->rotation.y = -math::PI / 2.f;
        {
            JointConfig joint;
            joint.type = JointConfig::Type::Revolute;
            joint.body = "Stop Post";
            joint.limited = true;
            joint.lower = -1.7f;// downstream: the way a log pushes it
            joint.upper = 0.04f;
            // Deliberately weak. This is the one drive in the yard that is meant
            // to LOSE: it holds the flap shut against nothing but its own
            // weight, and yields to a log.
            joint.stiffness = 90.f;
            joint.damping = 430.f;
            joint.maxForce = 5e4f;
            joint.target = 0.f;
            // MEASURED, not guessed, over the two runs this scene has: a full
            // eight-log mission peaks at 265 N and 65 N*m at this mount, and a
            // SPACE-forced pour-out peaks at 214 N and 128 N*m. Force does not
            // tell them apart — a single log leaning on a shut flap pushes
            // HARDER than a train shooting through an open one — but the
            // BENDING MOMENT does, because a forced pour loads the flap while
            // it is swinging and off its stop. So the mount is authored to fail
            // in bending, half way between the two.
            // MEASURED, not guessed — and measured EVERY SUBSTEP, because the
            // load here is a spike and a probe that samples once a second walks
            // straight past it (the first threshold picked that way snapped the
            // bar on log one, which is a bug that looks exactly like a scene
            // that hates you). Over a full eight-log mission this mount peaks at
            // 235 N; over a SPACE-forced pour-out, with the pack arriving in
            // contact, it peaks at 337 N. 280 N is the geometric mean of the
            // two, so each case has about a fifth of its own load in hand.
            //
            // FORCE is what tells them apart, and it took a while to find: the
            // obvious discriminators do not work. A single log leaning on a shut
            // flap pushes HARDER than a train shooting through an open one, so
            // "more logs" is not "more force" — what a pour-out does is arrive
            // with several logs IN CONTACT, and their belt drives add up. The
            // torque limit is set clear of both peaks so it never governs.
            // Set clear of everything the yard can do to it, including the
            // occasional log that reaches it uncut, and no longer used to tell
            // an abuse case apart from a mission — because with sawn cargo it
            // cannot be. The numbers say so plainly: an authored mission peaks
            // at 282 N and 116 N*m here, and a SPACE-forced pour-out peaks
            // LOWER, at 170 N and 83 N*m. A crowd of half-logs arriving at a
            // flap that is already swinging leans on it less than one log
            // shouldering it open does.
            //
            // What still demonstrates a joint giving way, eight times a run, is
            // the SAW — see kCutSlot. This one is the guard rail it always was.
            joint.breakForce = 6000.f;
            joint.breakTorque = 2500.f;
            joint.write(*mount);
        }
        {
            SensorConfig load;
            load.enabled = true;
            load.type = SensorConfig::Type::ForceTorque;
            load.rateHz = 120.f;
            load.seed = 5;
            load.write(*mount);
        }
        {
            ScriptConfig config;
            config.source = kStopBarSource;
            config.setField("master", "Yard Sign");
            config.setField("limit", "240");
            config.write(*mount);
        }
        stopBar->add(mount);

        // --- the yard sign ---------------------------------------------------
        // The sign is a GROUP, so the board, its posts and the lettering on it
        // are each authored in world units instead of inside a scaled box's
        // frame — and the group is what carries the mission script, which is
        // what script_from_object resolves when the stop bar reports a break.
        auto sign = Group::create();
        sign->name = "Yard Sign";
        sign->position.set(-5.9f, 1.55f, 2.9f);
        sign->rotation.y = 0.22f;
        {
            ScriptConfig config;
            config.source = kYardMasterSource;
            config.setField("logs", std::to_string(kLogCount));
            config.setField("gate", "Gate Hinge");
            config.setField("counter", "Bay Trigger");
            config.write(*sign);
        }
        scene->add(sign);

        // Real lettering, from the editor's own TEXT authoring: the glyphs are
        // baked into the mesh like any other geometry, so the sign reads with no
        // editor and no font file present, and the entry travels along for
        // whoever wants to edit the words.
        //
        // FLAT (depth 0). Extruded, "TIMBER YARD" alone costs 1.5 MB of vertices
        // in this document; a painted board wants paint anyway.
        auto paint = standard(0xe8dcc0, 0.85f, 0.f);
        struct Line {
            const char* name;
            const char* words;
            float size;
            float y;
        };
        static constexpr Line lines[] = {
                {"Sign Title", "TIMBER YARD", 0.17f, 0.085f},
                {"Sign Notice", "8 LOGS - MIND THE SAW", 0.075f, -0.115f},
        };

        // The BOARD IS SIZED FROM THE LETTERING, not guessed at: build the
        // glyphs, take their bounds, and cut a plank that clears them. Guessing
        // is how you get type that runs off the end of the board.
        float textHalfWidth = 0.f;
        float textTop = -1e9f;
        float textBottom = 1e9f;
        std::vector<std::pair<const Line*, std::shared_ptr<BufferGeometry>>> built;
        for (const auto& line : lines) {
            TextConfig text;
            text.text = line.words;
            text.size = line.size;
            text.depth = 0.f;
            text.curveSegments = 2;
            text.align = TextConfig::Align::Center;

            auto geometry = text.buildGeometry();
            geometry->computeBoundingBox();
            if (const auto& box = geometry->boundingBox; box && !box->isEmpty()) {
                textHalfWidth = std::max({textHalfWidth, std::abs(box->min().x),
                                          std::abs(box->max().x)});
                textTop = std::max(textTop, line.y + box->max().y);
                textBottom = std::min(textBottom, line.y + box->min().y);
            }
            built.emplace_back(&line, geometry);
        }

        const float boardHalfWidth = textHalfWidth + 0.22f;// margin either side
        const float boardHalfHeight = std::max(textTop, -textBottom) + 0.13f;

        auto board = mesh("Sign Board", unit, standard(0x6b4f2e, 0.9f, 0.f));
        board->scale.set(2.f * boardHalfWidth, 2.f * boardHalfHeight, 0.07f);
        sign->add(board);

        for (const auto& [line, geometry] : built) {
            auto glyphs = Mesh::create(geometry, paint);
            glyphs->name = line->name;
            // Proud of the board's front face, so there is nothing to z-fight
            // with and the letters catch the light as paint would.
            glyphs->position.set(0.f, line->y, 0.05f);
            glyphs->castShadow = false;
            glyphs->receiveShadow = false;
            TextConfig text;
            text.text = line->words;
            text.size = line->size;
            text.depth = 0.f;
            text.curveSegments = 2;
            text.align = TextConfig::Align::Center;
            text.write(*glyphs);
            sign->add(glyphs);
        }

        // TWO posts, at the board's side edges and set BEHIND its face. A single
        // pole up the middle stands in front of the lettering, which is a sign
        // you cannot read — and the plank is what the words are on.
        for (int side = 0; side < 2; ++side) {
            const float x = (side == 0 ? 1.f : -1.f) * (boardHalfWidth - 0.09f);
            auto post = mesh("Sign Post " + std::to_string(side + 1), unit, timberMaterial);
            post->position.set(x, -0.775f, -0.075f);
            post->scale.set(0.09f, 1.55f + boardHalfHeight, 0.09f);
            sign->add(post);
        }

        // --- the clearing edge -----------------------------------------------
        scene->add(makeYardOffice(shared));

        // The stack of cut timber beside the office: what the yard is FOR,
        // standing where it says so. Same log geometry as the cargo, so it
        // costs nothing but its nodes.
        auto stack = Group::create();
        stack->name = "Timber Stack";
        stack->position.set(-3.6f, 0.f, -5.6f);
        stack->rotation.y = 0.42f;
        for (int row = 0; row < 3; ++row) {
            const int inRow = 3 - row;
            for (int i = 0; i < inRow; ++i) {
                auto stacked = mesh("Stacked Log " + std::to_string(row * 3 + i + 1),
                                    logGeometry, logMaterial);
                stacked->position.set(static_cast<float>(i) * 0.42f + static_cast<float>(row) * 0.21f,
                                      0.2f + static_cast<float>(row) * 0.38f, 0.f);
                stacked->rotation.x = math::PI / 2.f;
                // A stack that was thrown, not laid: each log rolled a little
                // and none of them line up.
                stacked->rotation.y = 0.03f * static_cast<float>((i * 7 + row * 3) % 5 - 2);
                stack->add(stacked);
            }
        }
        scene->add(stack);

        // Six spruces, standing where they close the frame rather than where a
        // forest would put them: behind the rack, behind the bay, and two on the
        // camera side to give the shot a foreground. Height, girth, spin and
        // lean are all different — six instances of one geometry that do not
        // read as one tree six times.
        static constexpr struct {
            float x, z, height, girth, spin, lean;
        } spruces[] = {
                {-17.5f, -9.5f, 12.5f, 0.34f, 0.4f, 0.015f},
                {-5.5f, -14.5f, 15.0f, 0.4f, 1.9f, -0.02f},
                {8.5f, -12.5f, 11.5f, 0.31f, 2.7f, 0.025f},
                {17.5f, -3.5f, 13.5f, 0.37f, 0.9f, -0.01f},
                // The two on the camera's side stand WIDE, framing the yard
                // instead of standing in front of it — a spruce dead centre is
                // a very effective way to photograph a tree.
                {19.5f, 10.5f, 12.0f, 0.33f, 3.4f, 0.02f},
                {-19.5f, 11.5f, 14.0f, 0.38f, 5.1f, -0.025f},
        };
        int index = 1;
        for (const auto& spruce : spruces) {
            scene->add(makeSpruce(shared, "Spruce " + std::to_string(index++),
                                  spruce.x, spruce.z, spruce.height, spruce.girth,
                                  spruce.spin, spruce.lean));
        }

        return scene;
    }

    // --- deterministic identity ----------------------------------------------
    // Object3D, BufferGeometry and Material each draw a fresh random uuid at
    // construction, so two runs over identical sources would agree on every
    // value and disagree on every identity. An authored scene gets authored
    // identity: every uuid is derived from the scene path of the node carrying
    // the thing, which is what makes the reproducibility promise at the top of
    // this file checkable (`--check`).
    //
    // Deliberately the same scheme as HoverArenaAuthor.cpp's, rather than a
    // shared header: these are two generator programs, and a shipped example's
    // identity should not move because its sibling was refactored.

    std::uint64_t fnv1a(std::string_view text, std::uint64_t hash = 0xcbf29ce484222325ull) {

        for (const unsigned char c : text) {
            hash ^= c;
            hash *= 0x100000001b3ull;
        }
        return hash;
    }

    std::string uuidFrom(const std::string& key) {

        const auto lo = fnv1a(key);
        const auto hi = fnv1a(key, lo ^ 0x9e3779b97f4a7c15ull);

        std::array<std::uint8_t, 16> bytes{};
        for (int i = 0; i < 8; ++i) bytes[i] = static_cast<std::uint8_t>(lo >> (8 * i));
        for (int i = 0; i < 8; ++i) bytes[8 + i] = static_cast<std::uint8_t>(hi >> (8 * i));
        bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0f) | 0x40);
        bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3f) | 0x80);

        constexpr char hex[] = "0123456789abcdef";
        std::string uuid;
        uuid.reserve(36);
        for (int i = 0; i < 16; ++i) {
            if (i == 4 || i == 6 || i == 8 || i == 10) uuid += '-';
            uuid += hex[bytes[i] >> 4];
            uuid += hex[bytes[i] & 0x0f];
        }
        return uuid;
    }

    void assignAuthoredUuids(Scene& scene) {

        // A repeated path would collide two identities silently, so it gets a
        // deterministic ordinal instead (traversal order is authoring order).
        std::unordered_set<std::string> used;
        auto claim = [&used](const std::string& key) {
            auto candidate = key;
            for (int n = 2; !used.insert(candidate).second; ++n) {
                candidate = key + " #" + std::to_string(n);
            }
            return candidate;
        };

        // Shared geometries and materials keep the identity of their FIRST
        // carrier: the yard's one unit cube is the ground's, not a new identity
        // per box that references it.
        std::unordered_set<const void*> seen;

        std::function<void(Object3D&, const std::string&)> visit =
                [&](Object3D& object, const std::string& parent) {
                    const auto path = parent + '/' + object.name;
                    object.uuid = uuidFrom(claim("object:" + path));

                    // The shadow camera is exported as a full object of its own
                    // (ObjectExporter::writeShadow), uuid included.
                    if (auto* lit = dynamic_cast<LightWithShadow*>(&object)) {
                        if (lit->shadow && lit->shadow->camera) {
                            lit->shadow->camera->uuid =
                                    uuidFrom(claim("object:" + path + "/shadow camera"));
                        }
                    }

                    if (const auto geometry = object.geometry();
                        geometry && seen.insert(geometry.get()).second) {
                        geometry->uuid = uuidFrom(claim("geometry:" + path));
                    }
                    if (auto* withMaterials = object.as<ObjectWithMaterials>()) {
                        const auto& materials = withMaterials->materials();
                        for (std::size_t i = 0; i < materials.size(); ++i) {
                            if (!materials[i] || !seen.insert(materials[i].get()).second) continue;
                            auto key = "material:" + path;
                            if (i > 0) key += " #" + std::to_string(i);
                            materials[i]->setUuid(uuidFrom(claim(key)));

                            // And whatever hangs off it. The conveyor's belt
                            // ribbon carries a PROCEDURAL texture the sync pass
                            // drew, and a Texture draws its uuid like everything
                            // else — this was the one thing left drifting
                            // between two runs, and `images` is keyed off it too
                            // (ObjectExporter writes `<texture uuid>-image`), so
                            // fixing the texture fixes both entries.
                            for (auto& slot : textureSlotsOf(*materials[i])) {
                                if (!slot.current || !seen.insert(slot.current.get()).second) {
                                    continue;
                                }
                                slot.current->setUuid(uuidFrom(claim(key + "/" + slot.name)));
                            }
                        }
                    }

                    for (auto* child : object.children) visit(*child, path);
                };
        visit(scene, "");
    }

    // The whole authoring pass, from nothing to text. Called twice by --check,
    // which is the only way to prove the promise rather than assert it.
    std::string authorDocument(std::string& error, SceneDocument* keep = nullptr) {

        SceneDocument local;
        SceneDocument& document = keep ? *keep : local;
        document.replaceScene(buildScene());
        assignAuthoredUuids(document.scene());
        return document.toJson(/*prettyPrint*/ true, &error);
    }

    // --- the translation unit ------------------------------------------------
    // MSVC caps one string literal at 65535 bytes, so the document is emitted as
    // chunks joined at startup. Raw literals with a delimiter JSON cannot
    // contain, so nothing in the document has to be escaped.
    void writeSource(const std::filesystem::path& path, const std::string& json) {

        constexpr std::size_t kChunk = 8000;

        std::ofstream out(path, std::ios::binary);
        out << R"(// Timber Yard, as JSON compiled into the binary.
//
// GENERATED — do not edit. Rebuild it from the program that authors the scene:
//
//     cmake --build <build> --target timber_yard_author
//     <build>/bin/timber_yard_author timber_yard.json apps/editor/ExampleSceneTimberYard.cpp
//
// The author tool is byte-reproducible (every uuid is derived, nothing samples
// a clock), so regenerating an unchanged scene leaves this file untouched and a
// diff here is a diff somebody authored. `timber_yard_author --check` is that
// promise, checked.

#include "ExampleSceneData.hpp"

#include <string>

namespace {

    // See apps/editor/tools/TimberYardAuthor.cpp, which built it.
    const char* const kTimberYard[] = {
)";
        for (std::size_t i = 0; i < json.size(); i += kChunk) {
            out << "            R\"JSON(" << json.substr(i, kChunk) << ")JSON\",\n";
        }
        out << R"(    };

}// namespace

namespace threepp::editor::examples::detail {

    std::string timberYardJson() {

        std::string document;
        std::size_t size = 0;
        for (const auto* chunk : kTimberYard) size += std::char_traits<char>::length(chunk);
        document.reserve(size);
        for (const auto* chunk : kTimberYard) document += chunk;
        return document;
    }

}// namespace threepp::editor::examples::detail
)";
    }

}// namespace


int main(int argc, char** argv) {

    const bool check = argc > 1 && std::string(argv[1]) == "--check";
    if (argc < 2) {
        std::cerr << "usage: timber_yard_author <scene.json> [ExampleSceneTimberYard.cpp]\n"
                  << "       timber_yard_author --check\n";
        return 2;
    }

    std::string error;
    if (!scripting::ensureInterpreter(&error)) {
        std::cerr << "timber_yard_author: no Python interpreter: " << error << std::endl;
        return 1;
    }

    if (check) {
        const auto first = authorDocument(error);
        if (first.empty()) {
            std::cerr << "timber_yard_author: export failed: " << error << std::endl;
            return 1;
        }
        const auto second = authorDocument(error);
        if (first != second) {
            std::cerr << "timber_yard_author: NOT reproducible - two runs differ" << std::endl;
            // Where, so the next person is not bisecting a quarter megabyte.
            const auto limit = std::min(first.size(), second.size());
            std::size_t at = 0;
            while (at < limit && first[at] == second[at]) ++at;
            std::cerr << "  first difference at byte " << at << " of " << first.size()
                      << "/" << second.size() << std::endl;
            return 1;
        }
        std::cout << "timber_yard_author: reproducible (" << first.size()
                  << " bytes, twice)" << std::endl;
        return 0;
    }

    SceneDocument document;
    const auto json = authorDocument(error, &document);
    if (json.empty()) {
        std::cerr << "timber_yard_author: export failed: " << error << std::endl;
        return 1;
    }

    {
        std::ofstream out(argv[1], std::ios::binary);
        out << json;
    }
    std::size_t objects = 0;
    document.scene().traverse([&](Object3D&) { ++objects; });
    std::cout << "timber_yard_author: wrote " << argv[1] << " (" << json.size() << " bytes, "
              << objects << " objects)" << std::endl;

    if (argc > 2) {
        writeSource(argv[2], json);
        std::cout << "timber_yard_author: wrote " << argv[2] << std::endl;
    }
    return 0;
}
