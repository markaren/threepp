// A drop-in ambient bird flock: boids that fly, perch, walk about and lift off
// on their own. Not the centre-point of a scene — the whole design is about
// being cheap and unobtrusive while surviving being looked at directly, because
// someone always will.
//
// Three lines in the host:
//
//     auto birds = Flock::create();
//     scene->add(birds);
//     canvas.animate([&] { birds->update(clock.getDelta()); renderer->render(*scene, *camera); });
//
// Add `birds->bakePerches(*scene);` and they will find surfaces to land on.
//
// RENDERING: one merged BufferGeometry rebaked on the CPU every frame, drawn as
// a single Mesh with a stock MeshStandardMaterial. This is the one animated-mesh
// shape both backends already agree on — the host picks GL or Vulkan at a
// runtime prompt (RendererFactory.cpp:16-59), and the Vulkan backend has no
// generic ShaderMaterial path, so anything shader-driven renders there as a
// flat grey non-flapping blob with no warning at all. `flatShading` is likewise
// never touched: it has zero occurrences in the Vulkan backend, so setting it
// would make the birds look materially different depending on which renderer
// the user picked at the prompt. Every crease in this mesh comes from split
// vertices, which work identically on both.
//
// THE FLOCK NODE SHOULD STAY AT IDENTITY. The simulation runs in world space
// and vertices are written through a cached inverse of *matrixWorld, so a
// transformed node is correct — but with ONE FRAME OF MATRIX LAG, because
// update() runs before the renderer refreshes matrices. The error is a fraction
// of a bird-length even under motion, and it is stated here rather than fixed:
// the alternative is for this subsystem to call updateMatrixWorld() on someone
// else's node, which is a far worse thing to do to a host scene than being one
// frame stale. A parent with NON-UNIFORM scale additionally shears the birds,
// because the body basis is orthonormalised on the way into local space.
//
// TOPOLOGY IS IMMUTABLE FOR THE OBJECT'S LIFETIME. birdCount is fixed at
// construction; there is no setCount(), no capacity/live split and no parked
// birds. That is what makes the DrawUsage::Dynamic hint safe to set exactly
// once (gl/GLAttributes.cpp:38-56 captures it at glBufferData time, so an
// attribute REPLACED later silently reverts to Static), and it is why there is
// no degenerate parked vertex anywhere in the system to poison a normal.
//
// DETERMINISM: bit-identical for the same binary, the same seed, and the same
// dt sequence. Cross-toolchain agreement is ~1e-4 — sin/cos/atan2/pow are not
// bit-identical across libm implementations. -ffast-math breaks it entirely.
//
// FOUR PLACES WHERE THIS FILE DOES NOT DO WHAT THE DESIGN LITERALLY SAID, each
// because doing so was measured and did not work. They are called out at their
// own sites too; this is the index.
//   · The approach HANDS OFF from the gate to the perch (goalTarget()). Steering
//     at the gate for the whole approach converges on a point 2–3 m short of
//     every perch, and the Approach → Flare test — which measures range to the
//     SPOT — then never fires. Measured: 0 landings in 20 000 steps.
//   · abortChance is rolled ONCE per approach, not once per decision tick. An
//     approach spans six to eight ticks, so per-tick it turns 0.12 into a 64 %
//     abort rate.
//   · The territory force carries a RADIAL DAMPER and the home drift is 0.35 ×
//     roamRadius in TOTAL rather than per axis. A pure position spring overshoots
//     by v²/2a whatever its gain; without both the flock reaches 1.9 × roamRadius
//     from home and keeps going.
//   · The altitude spring is switched OFF while a bird is committed to a landing.
//     A ground perch sits a full cruiseAltitude below the preferred height, so
//     otherwise the altitude force out-pulls the goal force and no bird can
//     descend to the ground at all.
//
// Known limitations, stated rather than fixed:
//   · Neighbour search is O(N²) and birdCount is hard-clamped to 256. A uniform
//     grid is deliberately not in v1: it is the largest single source of
//     divergence for a subsystem that will never run 500 birds. 256² distance
//     tests is ≈ 0.25 ms; the default of 18 is 324 tests.
//   · The obstacle field is 2 m cells (PerchIndex's default), so a bird may
//     clip a bare twig or a wire. The ground floor comes from the heightfield
//     and is much finer.
//   · The perch table is a SNAPSHOT. Move a perched object and the bird floats.
//     Call bakePerches() again — there is no dirty tracking, deliberately,
//     because the bake's output holds no pointer into the scene.
//   · No soaring, no thermalling, no V-formations, no foot IK, no knee.
//
// Header-only, dependency-free beyond threepp core.

#ifndef THREEPP_FLOCK_HPP
#define THREEPP_FLOCK_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/fauna/BirdGeometry.hpp"
#include "threepp/extras/fauna/PerchIndex.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Sphere.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace threepp {

    class Flock: public Mesh {

    public:
        enum class BirdState : std::uint8_t {
            Cruise = 0,  // airborne, boids steering
            Approach = 1,// committed to a claimed perch, steering to the gate
            Flare = 2,   // last 1.2 m, landing controller
            Perched = 3, // feet planted; hops/walks when the spot is walkable
            Launch = 4,  // crouch, leg impulse, three boosted beats
            Evade = 5,   // startled, overriding steering
        };

        enum class BirdRole : std::uint8_t { Follower = 0,
                                             Leader = 1,
                                             Loner = 2 };

        enum class Gait : std::uint8_t { Hop = 0,
                                         Walk = 1 };

        struct Params {
            unsigned int seed = 1337u;

            // ── Population ───────────────────────────────────────────────
            // 18 birds over a 42 m radius reads as "a place where birds live".
            // 200 reads as "a bird simulation" and drags the eye to exactly
            // where this subsystem is not supposed to put it. Raise it knowing
            // that. Hard-clamped to [0, 256]: neighbour search is O(N²) and a
            // uniform grid is deliberately not in v1.
            int birdCount = 18;

            // ── Territory (world, metres) ────────────────────────────────
            Vector3 home{0.f, 14.f, 0.f};// centre of the loiter volume; DRIFTS at runtime
            float roamRadius = 42.f;     // soft; the bounds force is zero inside 0.75× this
            float cruiseAltitude = 14.f; // m above the baked ground under `home`
            float altitudeSpread = 0.35f;// ± fraction, per bird — breaks the flock plane
            float homeDriftRate = 0.012f;// Hz; the territory migrates so it is never a bowl

            // ── Species ──────────────────────────────────────────────────
            float massKg = 0.078f;  // drives wingbeatHz allometrically
            float wingbeatHz = 0.f; // 0 ⇒ 8.5 · (massKg/0.078)^(-1/3)
            bool nyquistGuard = true;// clamp effective beat to 1/(6·dtSmoothed)
            Gait gait = Gait::Walk; // starlings/crows/pigeons walk; finches/sparrows hop

            // ── Flight ───────────────────────────────────────────────────
            float cruiseSpeed = 9.0f;    // m/s
            float minSpeed = 4.5f;       // m/s; below this the bird stall-flares
            float maxSpeed = 17.0f;      // m/s
            float maxAccelAlong = 3.0f;  // m/s² — birds accelerate slowly
            float maxAccelLateral = 9.0f;// m/s² — and turn hard. The asymmetry matters.
            float speedDrag = 1.8f;      // 1/s toward target speed; a soft law, never a clamp
            float maxTurnRate = 3.2f;    // rad/s, heading slew cap
            float maxBank = 1.05f;       // rad (60°)
            float bankGain = 1.25f;      // birds over-bank a coordinated turn
            float bankTau = 0.18f;       // s, roll response time constant
            float pitchTau = 0.20f;      // s

            // ── Wingbeat kinematics (radians / cycles) ───────────────────
            float strokeDown = 0.88f;    // rad below horizontal (50°)
            float strokeUp = 0.58f;      // rad above horizontal (33°)
            float downstrokeFrac = 0.42f;// fraction of the cycle spent going down
            float spanLagCycles = 0.13f; // root→tip travelling-wave lag
            float twistAmp = 0.34f;      // rad tip twist — the scintillation
            float wristFlex = 1.00f;     // rad hand fold at mid-upstroke
            float glideDihedral = 0.10f; // rad, static dihedral held while gliding
            float perchSweep = 1.35f;    // rad, folded-wing sweep
            float perchLift = 0.20f;     // rad, folded-wing lift

            // ── Boids ────────────────────────────────────────────────────
            // Topological, not metric (Ballerini et al., PNAS 2008): a fixed
            // NUMBER of nearest neighbours, so the flock keeps cohesion whether
            // it compresses or spreads. That density independence is what lets
            // ONE set of weights work in a scene whose bird count and volume
            // the host chose, not us.
            int neighbourCount = 7;
            float neighbourRadius = 8.0f;   // m, candidate gather only
            float rearBlindAngle = 0.52f;   // rad, half-cone behind that is ignored
            float separationDistance = 1.6f;// m
            float cohesionDeadZone = 1.2f;  // m; without it a bird at the centroid jitters
            float wSeparation = 1.80f;
            float wAlignment = 0.60f;
            float wCohesion = 0.45f;
            float wWander = 0.80f;
            float wBounds = 1.20f;
            float wAltitude = 0.60f;
            float wObstacle = 3.50f;
            float wGround = 4.00f;
            float wGoal = 1.20f;
            float wObserver = 0.15f;    // soft repulsion from setObserver()'s camera
            float leaderFraction = 0.08f;
            float lonerFraction = 0.20f;
            float leaderReassign = 7.0f;// s; leadership is a POSITION, not a trait

            // ── Obstacles ────────────────────────────────────────────────
            float lookaheadTime = 0.55f;    // s, obstacle field sample distance
            float obstacleMargin = 6.0f;    // m, clearance below which the field repels
            float minGroundClearance = 1.2f;// m above heightAt()

            // ── Perching ─────────────────────────────────────────────────
            bool perching = true;
            float perchSearchRadius = 32.f;  // m
            float perchIntervalMin = 25.f;   // s aloft before a bird wants down
            float perchIntervalMax = 90.f;
            float restIntervalMin = 12.f;    // s perched before it wants up
            float restIntervalMax = 70.f;
            float maxPerchedFraction = 0.55f;// 0..1; the sky is never allowed to empty
            float perchContagion = 0.35f;    // 0..1, land-because-others-landed
            float launchContagion = 0.55f;   // 0..1, leave-because-a-neighbour-left
            float contagionRadius = 4.0f;    // m
            float flareDistance = 1.2f;      // m
            float settleTime = 0.35f;        // s, contact → wings fully folded
            float abortChance = 0.12f;       // approaches that go around — a FEATURE
            float groundBias = 0.30f;        // 0..1 chance a walkable perch gets walked about
            fauna::PerchIndex::Params perch{};

            // ── Disturbance ──────────────────────────────────────────────
            float flightInitiationDistance = 6.0f;// m, × per-bird boldness
            float startleWaveSpeed = 25.f;        // m/s, contagion propagation
            float postFlushCalmMin = 20.f;        // s before a flushed flock will land again
            float postFlushCalmMax = 60.f;

            // ── Look ─────────────────────────────────────────────────────
            fauna::BirdShape shape{};
            fauna::BirdPlumage plumage{};
            float sizeVariation = 0.10f;  // ± fraction, per bird
            bool birdsCastShadow = false; // correct when enabled — the deformation is in the verts
            float lodFarDistance = 60.f;  // m; beyond this a bird is re-baked every 2nd frame
            Vector2 wind{0.7f, 0.7f};     // world XZ; perched birds face into it

            bool operator==(const Params&) const = default;
        };

        explicit Flock(const Params& params)
            : Mesh(BufferGeometry::create(), defaultMaterial()),
              params_(sanitise(params)),
              tmpl_(fauna::makeBirdTemplate(params_.shape)) {

            build();
        }

        // Two overloads, NOT a defaulted argument: GCC rejects a default
        // argument naming a nested class's defaults before the outer class is
        // complete (cf. FireEffect.hpp:348-363).
        static std::shared_ptr<Flock> create(const Params& params) {

            return std::make_shared<Flock>(params);
        }

        static std::shared_ptr<Flock> create() {

            return create(Params{});
        }

        [[nodiscard]] std::string type() const override {

            return "Flock";
        }

        // Advance the simulation and rebake the geometry. CALL ONCE PER FRAME.
        //
        // dt is clamped internally to [0, 0.05] s. A debugger pause or a loading
        // hitch otherwise teleports the whole flock through the scene in one
        // step, and at 0.05 s a 17 m/s bird still moves less than half an
        // obstacle cell (0.85 m against the 2 m default), so it cannot tunnel
        // through geometry either. There is no sub-stepping: a hitch costs the
        // flock a little travel, which nobody can see, rather than a variable
        // number of integrations, which nobody can reproduce.
        // dt <= 0 is a no-op that increments stalledUpdates().
        void update(float dt) {

            if (!(dt > 0.f) || !std::isfinite(dt)) {
                ++stalled_;
                return;
            }
            stalled_ = 0;

            dt = std::min(dt, kMaxStep);
            lastDt_ = dt;
            time_ += dt;
            ++frame_;
            ++updates_;

            // A FIXED EMA, not a wall-clock measurement: the Nyquist guard reads
            // it, so it has to be a pure function of the dt sequence or the
            // wingbeat frequency — and therefore every wing vertex — would
            // depend on how fast the machine happened to be running.
            dtSmoothed_ = 0.9f * dtSmoothed_ + 0.1f * dt;

            if (bakeRequested_ && !perch_.complete()) {
                if (perch_.step()) {
                    bakeRequested_ = false;
                    hasField_ = true;
                    syncClaims();
                }
            }

            if (birds_.empty()) return;

            refreshWorldInverse();
            updateAggregates();

            for (int i = 0; i < count_; ++i) gatherNeighbours(i);
            for (int i = 0; i < count_; ++i) decide(i, dt);
            for (int i = 0; i < count_; ++i) accel_[static_cast<std::size_t>(i)] = steer(i);
            for (int i = 0; i < count_; ++i) integrate(i, dt);

            // DOUBLE BUFFERED. Steps 4, 6 and 7 above all read prev_ and only
            // the integrator writes next_. Without this, bird 5 would see bird
            // 0's new position and bird 6's old one — still deterministic, but
            // it makes the neighbour relation asymmetric in a way nobody can
            // reason about, and it silently couples the result to loop order in
            // every future refactor. This is not an optimisation to undo.
            std::swap(prev_, next_);

            bakeVertices();
        }

        // One-time scene scan: perch spots, obstacle field, ground heightfield.
        // Amortised over frames at Params::perch.bakeWorkPerFrame; until it
        // completes the birds simply fly. Excludes this Flock automatically, so
        // add() order does not matter.
        //
        // THE HOST MUST CALL THIS AGAIN AFTER CHANGING THE SCENE. There is no
        // dirty tracking, deliberately: the bake's output holds no pointer into
        // the scene, so a stale bake leaves a bird perched in mid-air rather
        // than dereferencing freed geometry.
        void bakePerches(Object3D& sceneRoot) {

            releaseAllClaims();
            perch_.begin(sceneRoot, params_.perch, this, filter_);
            bakeRequested_ = true;
            hasField_ = false;
            syncClaims();
        }

        void bakePerchesBlocking(Object3D& sceneRoot) {

            releaseAllClaims();
            perch_.bakeBlocking(sceneRoot, params_.perch, this, filter_);
            bakeRequested_ = false;
            hasField_ = true;
            syncClaims();
        }

        [[nodiscard]] bool bakeComplete() const {

            return perch_.complete();
        }

        [[nodiscard]] float bakeProgress() const {

            return perch_.progress();
        }

        [[nodiscard]] std::size_t perchCount() const {

            return perch_.spots().size();
        }

        // Kept OUT of Params on purpose: std::function has no operator==, so a
        // std::function member would silently DELETE the defaulted operator==
        // the house style requires — and the deletion is only diagnosed at the
        // first use site, far from the cause.
        void setPerchFilter(std::function<bool(const Mesh&)> filter) {

            filter_ = std::move(filter);
        }

        // Authored perches — the escape hatch for a huge scene, or a designer
        // who knows exactly which three railings matter. Call instead of
        // bakePerches(); the obstacle field and heightfield stay empty, so
        // birds get no obstacle avoidance and a flat floor at home.y - 100.
        void setPerches(std::vector<fauna::PerchSpot> spots) {

            releaseAllClaims();
            perch_.setSpots(std::move(spots));
            bakeRequested_ = false;
            hasField_ = false;
            syncClaims();
        }

        void addPerch(const Vector3& worldPos, const Vector3& worldNormal, bool walkable) {

            fauna::PerchSpot spot;
            spot.position = worldPos;
            spot.normal = fauna::detail::safeNormalized(worldNormal, {0, 1, 0});
            if (spot.normal.y < 0.f) spot.normal.negate();
            spot.walkable = walkable;
            spot.ground = false;

            releaseAllClaims();
            perch_.addSpot(spot);
            bakeRequested_ = false;
            syncClaims();
        }

        // ── Events ───────────────────────────────────────────────────────
        // A wave of agitation at `epicentre`, propagating outward at
        // Params::startleWaveSpeed. Perched birds launch as it reaches them;
        // flying birds break away. Twelve lines for one of the most
        // recognisable behaviours in nature — and the wave is the whole point:
        // a flock that flushes on one frame reads as a scripted cut, while the
        // same flock flushing over 0.4 s reads as alarm spreading through it.
        //
        // Safe to call from outside update(): it only queues per-bird reaction
        // TIMES and touches nothing the steering or integration steps read.
        void startle(const Vector3& epicentre, float radius = 1e9f, float strength = 1.f) {

            if (birds_.empty() || !fauna::detail::isFinite(epicentre)) return;

            const float r2 = radius * radius;
            const float waveSpeed = std::max(params_.startleWaveSpeed, 0.1f);

            for (int i = 0; i < count_; ++i) {

                Bird& b = birds_[static_cast<std::size_t>(i)];
                const float d2 = prev_[static_cast<std::size_t>(i)].pos.distanceToSquared(epicentre);
                if (d2 > r2) continue;

                // Even the bypass gets a per-bird latency. A wave that arrives
                // at the speed of sound and is acted on in the same frame is
                // still a simultaneous flush wherever the birds are close
                // together, which is exactly where a row of them is.
                const float latency = 0.05f + 0.20f * roll(b, i);
                const float at = time_ + std::sqrt(d2) / waveSpeed + latency;

                if (b.reactAt < 0.f || at < b.reactAt) {
                    b.reactAt = at;
                    b.startleFrom = epicentre;
                    b.startleStrength = strength;
                }
            }
        }

        // Non-owning; perched birds flush when it enters their personal flight-
        // initiation distance. Pass nullptr to clear. THE CALLER MUST CLEAR IT
        // BEFORE DESTROYING THE NODE — it is dereferenced once per bird per frame.
        void setDisturbanceSource(const Object3D* source) {

            disturbance_ = source;
        }

        // Non-owning; enables the distance LOD and camera shyness. nullptr clears.
        // Same lifetime rule as above, and the same reason: the dereference is in
        // the hot path, so a dangling pointer here is a crash in the render loop
        // rather than a stale frame.
        void setObserver(const Camera* camera) {

            observer_ = camera;
        }

        void setWind(const Vector2& dirXZ) {

            params_.wind = dirXZ;
        }

        // ── Query ────────────────────────────────────────────────────────
        [[nodiscard]] int birdCount() const {

            return count_;
        }

        [[nodiscard]] int perchedCount() const {

            return perchedNow_;
        }

        [[nodiscard]] int flyingCount() const {

            return count_ - perchedNow_;
        }

        [[nodiscard]] BirdState stateOf(int i) const {

            if (i < 0 || i >= count_) return BirdState::Cruise;
            return birds_[static_cast<std::size_t>(i)].state;
        }

        [[nodiscard]] BirdRole roleOf(int i) const {

            if (i < 0 || i >= count_) return BirdRole::Follower;
            return birds_[static_cast<std::size_t>(i)].role;
        }

        [[nodiscard]] const Vector3& birdPosition(int i) const {

            if (i < 0 || i >= count_) return zero_;
            return prev_[static_cast<std::size_t>(i)].pos;
        }

        [[nodiscard]] const Vector3& birdVelocity(int i) const {

            if (i < 0 || i >= count_) return zero_;
            return prev_[static_cast<std::size_t>(i)].vel;
        }

        [[nodiscard]] const fauna::PerchIndex& perchIndex() const {

            return perch_;
        }

        [[nodiscard]] const Params& params() const {

            return params_;
        }

        // Diagnostics. These turn the two bug reports this subsystem will
        // actually generate — "my birds don't move" and "my birds never land" —
        // into a ten-second answer instead of a support thread.
        [[nodiscard]] std::uint64_t updateCount() const {

            return updates_;
        }

        [[nodiscard]] std::uint64_t stalledUpdates() const {

            return stalled_;
        }

        // The material this Flock was built with. vertexColors is true and
        // color is white; if you swap the material, keep both, or every bird
        // renders black (a vertexColors material with no colour attribute gets
        // the generic attribute default 0,0,0,1 — and this one HAS the
        // attribute, so the failure is the other way round: drop vertexColors
        // and every bird turns into an untextured white dart).
        // MESHSTANDARDMATERIAL, NOT MESHPHONGMATERIAL, AND THE REASON IS THE
        // SAME ONE THAT BANS flatShading HERE.
        //
        // The host chooses its backend at runtime, so anything this material
        // says has to survive both. The Vulkan backend builds its material
        // record by asking for exactly four interfaces — MaterialWithColor,
        // MaterialWithRoughness, MaterialWithMetalness, MaterialWithEmissive
        // (VulkanCoreScene.cpp materialFromMesh). MeshPhongMaterial implements
        // MaterialWithSpecular and none of the middle two, so a Phong flock
        // keeps its albedo and its vertex colours and then silently falls back
        // to the default roughness 0.5 / metalness 0: every bit of shininess
        // and specular tuning is dropped on the floor, with no warning, and the
        // plumage reads differently depending on which renderer the host
        // happened to pick. MeshStandardMaterial's roughness and metalness are
        // read by BOTH backends, so what is tuned here is what is seen.
        //
        // vertexColors is true and color is white; if you swap the material,
        // keep both. Dropping vertexColors turns every bird into an untextured
        // white dart.
        [[nodiscard]] static std::shared_ptr<MeshStandardMaterial> defaultMaterial() {

            auto m = MeshStandardMaterial::create();
            m->color = Color(0xffffff);// the vertex colour IS the albedo
            m->vertexColors = true;
            // Every part is a closed solid except two buried holes per bird
            // (wing root, hip ring) that sit inside the opaque body, so the
            // cheap side is also the correct one.
            m->side = Side::Front;
            // Matte plumage. Feathers are not glossy, and a low roughness here
            // puts a moving highlight on every bird that draws the eye straight
            // to the thing that is meant to stay in the background.
            m->roughness = 0.65f;
            m->metalness = 0.f;
            m->name = "flockPlumage";
            return m;
        }

        ~Flock() override = default;

    private:
        // ── Tuning that is not a knob ────────────────────────────────────
        static constexpr float kMaxStep = 0.05f;    // s, the dt ceiling (see update())
        static constexpr float kGravity = 9.81f;    // m/s², used by bounds/bounding/leaps only
        static constexpr int kMaxBirds = 256;       // O(N²) neighbours; see the banner
        static constexpr int kMaxNeighbours = 24;
        static constexpr float kStepTime = 0.13f;   // s, one walking step's swing phase
        static constexpr float kStepRate = 3.0f;    // Hz
        static constexpr float kGroundSpeed = 0.38f;// m/s, walking
        static constexpr float kHopCycle = 0.38f;   // s
        static constexpr float kSaccade = 0.045f;   // s — the head SNAPS; see below
        static constexpr float kBalanceHold = 0.12f;// s, wings held half-raised after contact
        static constexpr float kLiftOmega = 22.f;   // rad/s, landing spring
        static constexpr float kLiftZeta = 0.55f;

        // Position and velocity, and nothing else: this is the only state that
        // is double-buffered, because it is the only state one bird reads off
        // another.
        struct Dyn {
            Vector3 pos{};
            Vector3 vel{};
        };

        struct Bird {
            // ── Personality, drawn once (§5.4) ───────────────────────────
            float beatRate = 1.f;
            float beatAmp = 1.f;
            float size = 1.f;
            float speedScale = 1.f;
            float boldness = 1.f;
            float sociability = 1.f;
            float restlessness = 1.f;
            float yPrefFrac = 1.f;
            float rollTrim = 0.f;
            float wingAsym = 0.f;
            float decisionPeriod = 0.5f;
            Vector3 wanderAxis{0, 1, 0};
            Vector3 wanderDir{0, 0, 1};
            float gaitPhase = 0.f;
            std::array<float, 4> idlePhase{};

            // ── Runtime ──────────────────────────────────────────────────
            BirdState state = BirdState::Cruise;
            BirdRole role = BirdRole::Follower;
            std::uint32_t seq = 0;// event sequence; advanced only inside decision ticks
            float stateTime = 0.f;
            float nextDecision = 0.f;
            bool baked = false;

            Vector3 fwd{0, 0, 1};// heading, unit
            float bank = 0.f;
            float pitch = 0.f;
            float targetSpeed = 9.f;

            float cyclePos = 0.f;
            float flapWeight = 1.f;
            float perchFold = 0.f;
            float tailSpread = 0.10f;
            float tailPitch = 0.f;
            float tailRoll = 0.f;
            float legExtend = 0.f;
            bool feetPlanted = false;
            std::array<Vector3, 2> footWorld{};

            float bodyLift = 0.f;
            float liftX = 0.f;// landing-spring displacement
            float liftV = 0.f;// landing-spring velocity

            // Head. Gaze is stabilised by the NECK, not by eye movement — the
            // head snaps to a target and holds it.
            float headYaw = 0.f, headPitch = 0.f, headRoll = 0.f, headLead = 0.f;
            float headYawFrom = 0.f, headPitchFrom = 0.f;
            float headYawTo = 0.f, headPitchTo = 0.f;
            float saccadeStart = -1.f;
            float saccadeNext = 0.f;

            // Idle timers (four incommensurate periods per bird, so two perched
            // birds essentially never twitch together).
            float flickAt = 0.f, flickStart = -1.f;
            float shuffleAt = 0.f, shuffleStart = -1.f;
            float preenAt = 0.f, preenUntil = -1.f;

            // Perching
            float perchUrge = 0.f, restUrge = 0.f;
            float perchInterval = 50.f, restInterval = 30.f;
            float perchSuppress = 0.f;// s of enforced airtime
            int claim = -1;           // index into perch_.spots(), -1 = none
            Vector3 spotPos{}, spotNormal{0, 1, 0}, gate{};
            bool spotWalkable = false;

            // Perched stance
            Vector3 anchor{};// world foot-contact point
            Vector3 stagger{};
            Vector3 groundGoal{};
            bool walker = false;
            bool leaped = false;
            bool abortRolled = false;
            bool gatePassed = false;
            float pauseUntil = 0.f;
            float stepStart = -1.f;
            float gaitPitch = 0.f;
            float gaitLift = 0.f;
            int swingFoot = 0;
            Vector3 stepFrom{}, stepTo{};

            // Flap-bounding
            bool bounding = false;
            float boundT = 0.f;
            float beatsLeft = 5.f;

            // Startle
            float reactAt = -1.f;
            Vector3 startleFrom{};
            float startleStrength = 0.f;
            Vector3 evadeAway{1, 0, 0};
            float evadeUntil = 0.f;
        };

        // ── Deterministic stateless hash ─────────────────────────────────
        //
        // Copied from FireEffect.cpp:15-34. It lives here as private statics
        // rather than in an anonymous namespace, which is what the spec's prose
        // says: an anonymous namespace inside a HEADER gives every translation
        // unit its own copy, and an inline member function that referenced one
        // would be an ODR violation the linker is free not to diagnose.
        //
        // A decision stream is a pure function of (seed, bird, sequence) and is
        // therefore independent of frame ordering, of how many frames a bake
        // took, and of wall time — none of which are reproducible.
        [[nodiscard]] static std::uint32_t hashU(std::uint32_t x) {

            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return x;
        }

        [[nodiscard]] static float rnd01(std::uint32_t seed, std::uint32_t slot, std::uint32_t stream) {

            const std::uint32_t h = hashU(slot * 0x9e3779b9u + stream * 0x85ebca6bu + seed);
            // 24 bits -> [0,1). Exact in fp32, and never reaches 1.0.
            return static_cast<float>(h >> 8) * (1.f / 16777216.f);
        }

        // One draw for one bird, advancing that bird's own sequence. Never call
        // it twice inside a single expression: argument evaluation order is
        // unspecified, and two draws in one argument list is exactly how
        // "deterministic for a fixed seed" comes to mean different things on
        // different toolchains.
        float roll(Bird& b, int i) const {

            return rnd01(params_.seed, static_cast<std::uint32_t>(i), b.seq++);
        }

        float rollRange(Bird& b, int i, float lo, float hi) const {

            return lo + (hi - lo) * roll(b, i);
        }

        // ── Parameter hygiene ────────────────────────────────────────────
        //
        // Clamped rather than trusted, because three of these turn a typo into
        // something that does not look like a typo: a birdCount of 5000 turns an
        // ambient background into a 6-second frame, a zero perchInterval makes
        // every bird want down on the first tick, and a negative separation
        // distance inverts the boids force so the flock implodes into one point.
        [[nodiscard]] static Params sanitise(const Params& in) {

            Params p = in;

            p.birdCount = std::clamp(p.birdCount, 0, kMaxBirds);
            p.roamRadius = std::max(p.roamRadius, 1.f);
            p.cruiseAltitude = std::max(p.cruiseAltitude, 0.5f);
            p.altitudeSpread = std::clamp(p.altitudeSpread, 0.f, 0.9f);
            p.massKg = std::max(p.massKg, 1e-3f);

            p.maxSpeed = std::max(p.maxSpeed, 1.f);
            p.cruiseSpeed = std::clamp(p.cruiseSpeed, 0.5f, p.maxSpeed);
            p.minSpeed = std::clamp(p.minSpeed, 0.1f, p.cruiseSpeed);
            p.maxAccelAlong = std::max(p.maxAccelAlong, 0.1f);
            p.maxAccelLateral = std::max(p.maxAccelLateral, 0.1f);
            p.maxTurnRate = std::max(p.maxTurnRate, 0.05f);
            p.maxBank = std::clamp(p.maxBank, 0.f, 1.4f);
            p.bankTau = std::max(p.bankTau, 1e-3f);
            p.pitchTau = std::max(p.pitchTau, 1e-3f);

            p.neighbourCount = std::clamp(p.neighbourCount, 0, kMaxNeighbours);
            p.neighbourRadius = std::max(p.neighbourRadius, 0.f);
            p.rearBlindAngle = std::clamp(p.rearBlindAngle, 0.f, math::PI);
            p.separationDistance = std::max(p.separationDistance, 1e-2f);
            p.cohesionDeadZone = std::max(p.cohesionDeadZone, 0.f);
            p.leaderFraction = std::clamp(p.leaderFraction, 0.f, 1.f);
            p.lonerFraction = std::clamp(p.lonerFraction, 0.f, 1.f - p.leaderFraction);
            p.leaderReassign = std::max(p.leaderReassign, 0.5f);

            p.obstacleMargin = std::max(p.obstacleMargin, 1e-2f);
            p.minGroundClearance = std::max(p.minGroundClearance, 0.f);

            p.perchIntervalMin = std::max(p.perchIntervalMin, 0.5f);
            p.perchIntervalMax = std::max(p.perchIntervalMax, p.perchIntervalMin);
            p.restIntervalMin = std::max(p.restIntervalMin, 0.5f);
            p.restIntervalMax = std::max(p.restIntervalMax, p.restIntervalMin);
            p.maxPerchedFraction = std::clamp(p.maxPerchedFraction, 0.f, 1.f);
            p.perchContagion = std::clamp(p.perchContagion, 0.f, 1.f);
            p.launchContagion = std::clamp(p.launchContagion, 0.f, 1.f);
            p.contagionRadius = std::max(p.contagionRadius, 1e-2f);
            p.flareDistance = std::max(p.flareDistance, 0.05f);
            p.settleTime = std::max(p.settleTime, 1e-2f);
            p.abortChance = std::clamp(p.abortChance, 0.f, 1.f);
            p.groundBias = std::clamp(p.groundBias, 0.f, 1.f);

            p.postFlushCalmMin = std::max(p.postFlushCalmMin, 0.f);
            p.postFlushCalmMax = std::max(p.postFlushCalmMax, p.postFlushCalmMin);

            p.sizeVariation = std::clamp(p.sizeVariation, 0.f, 0.5f);
            p.lodFarDistance = std::max(p.lodFarDistance, 1.f);

            return p;
        }

        // ── Construction ─────────────────────────────────────────────────
        void build() {

            count_ = params_.birdCount;
            seedRng();
            buildGeometry();
            buildBirds();
            bakeVertices();
        }

        void buildGeometry() {

            // The nine live wingbeat knobs are routed through BirdGeometry's
            // six-argument poseBird(). Without this the five-argument form would
            // quietly use the stock constants and Params' whole "Wingbeat
            // kinematics" block would be dead code that still autocompletes.
            kin_.strokeDown = params_.strokeDown;
            kin_.strokeUp = params_.strokeUp;
            kin_.downstrokeFrac = params_.downstrokeFrac;
            kin_.spanLagCycles = params_.spanLagCycles;
            kin_.twistAmp = params_.twistAmp;
            kin_.wristFlex = params_.wristFlex;
            kin_.glideDihedral = params_.glideDihedral;
            kin_.perchSweep = params_.perchSweep;
            kin_.perchLift = params_.perchLift;

            const auto verts = static_cast<std::size_t>(count_) * fauna::kVertsPerBird;

            std::vector<float> positions(verts * 3u, 0.f);
            std::vector<float> normals(verts * 3u, 0.f);
            // A zero normal on an unwritten vertex is the only safe default: it
            // shades black rather than NaN, and every vertex is overwritten by
            // the first bakeVertices() call in the constructor anyway.
            for (std::size_t v = 0; v < verts; ++v) normals[v * 3u + 1u] = 1.f;

            auto geo = geometry_;
            geo->setAttribute("position", FloatBufferAttribute::create(std::move(positions), 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(std::move(normals), 3));
            geo->setAttribute("color", FloatBufferAttribute::create(
                                               fauna::makeFlockColors(tmpl_, params_.plumage, count_, params_.seed), 3));
            geo->setIndex(fauna::makeFlockIndices(count_));

            posAttr_ = geo->getAttribute<float>("position");
            nrmAttr_ = geo->getAttribute<float>("normal");

            // SET BEFORE THE FIRST RENDER, ONCE. gl/GLAttributes.cpp:38-56
            // captures the usage hint at glBufferData time, so a hint set after
            // the first upload does nothing at all. Topology is immutable
            // (§0.5), so these attributes are never replaced and this call never
            // needs repeating — which is precisely why topology is immutable.
            posAttr_->setUsage(DrawUsage::Dynamic);
            nrmAttr_->setUsage(DrawUsage::Dynamic);

            // A geometry with no explicit bounding sphere is frustum-culled
            // against an empty optional that NOTHING recomputes
            // (BufferGeometry.hpp:38; Frustum.cpp:68-72). Populate it here so
            // even a never-updated Flock renders, and refresh it every frame so
            // frustumCulled can stay TRUE — which is better than the usual
            // workaround of switching culling off and paying for the draw
            // whichever way the camera is facing.
            geo->boundingSphere = Sphere(Vector3{}, 0.f);

            castShadow = params_.birdsCastShadow;
            receiveShadow = false;
            name = "Flock";
        }

        void seedRng() {

            std::mt19937 rng(params_.seed ? params_.seed : 1u);

            // [0,1) by hand rather than through uniform_real_distribution, whose
            // mapping is implementation-defined even though mt19937's output
            // sequence is not. Same reasoning as makeFlockColors().
            auto u01 = [&rng] { return static_cast<float>(rng() >> 8) * (1.f / 16777216.f); };

            for (int m = 0; m < 3; ++m) {
                // Every draw in its own named const. Three of them in one set()
                // call would make the flock's territory drift differ per
                // toolchain, and it would take a very long afternoon to find.
                const float ax = u01() * 2.f - 1.f;
                const float ay = (u01() * 2.f - 1.f) * 0.25f;
                const float az = u01() * 2.f - 1.f;
                const float ph = u01() * math::TWO_PI;
                driftAxis_[static_cast<std::size_t>(m)] =
                        fauna::detail::safeNormalized({ax, ay, az}, {1, 0, 0});
                driftPhase_[static_cast<std::size_t>(m)] = ph;
            }

            rngDraws_.clear();
            rngDraws_.reserve(static_cast<std::size_t>(count_) * 24u);
            for (int i = 0; i < count_ * 24; ++i) rngDraws_.push_back(u01());
        }

        void buildBirds() {

            birds_.assign(static_cast<std::size_t>(count_), Bird{});
            prev_.assign(static_cast<std::size_t>(count_), Dyn{});
            next_.assign(static_cast<std::size_t>(count_), Dyn{});
            accel_.assign(static_cast<std::size_t>(count_), Vector3{});
            nbrIdx_.assign(static_cast<std::size_t>(count_) * kMaxNeighbours, -1);
            nbrCount_.assign(static_cast<std::size_t>(count_), 0);

            // ROLES BY DETERMINISTIC INDEX RATIO, not by RNG: the counts must be
            // exactly reproducible even when the personality draws are not being
            // consumed in the same order (they are, but this removes the
            // question).
            leaderCount_ = static_cast<int>(std::lround(static_cast<double>(count_) * params_.leaderFraction));
            lonerCount_ = static_cast<int>(std::lround(static_cast<double>(count_) * params_.lonerFraction));
            leaderCount_ = std::clamp(leaderCount_, 0, count_);
            lonerCount_ = std::clamp(lonerCount_, 0, count_ - leaderCount_);

            for (int i = 0; i < count_; ++i) {

                Bird& b = birds_[static_cast<std::size_t>(i)];
                const std::size_t d = static_cast<std::size_t>(i) * 24u;

                // phase01 — THE SINGLE CHEAPEST ANTI-TELL IN THE DESIGN.
                // Without it every bird beats in lockstep and the flock reads as
                // one organism cloned N times, which is the first thing anyone
                // notices and the last thing they can name.
                b.cyclePos = draw(d + 0);
                // Two birds at 8.5 and 9.2 Hz drift a half-cycle apart in 0.7 s,
                // so even a genuine synchronising event decorrelates within ~2 s
                // with no special handling anywhere.
                b.beatRate = 0.88f + 0.24f * draw(d + 1);
                b.beatAmp = 0.92f + 0.16f * draw(d + 2);
                b.size = 1.f + (draw(d + 3) * 2.f - 1.f) * params_.sizeVariation;
                b.speedScale = 0.88f + 0.24f * draw(d + 4);
                b.boldness = 0.30f + 0.70f * draw(d + 5);
                b.sociability = 0.40f + 0.90f * draw(d + 6);
                b.restlessness = 0.50f + 1.10f * draw(d + 7);
                // The axis most implementations forget, because separation is
                // always tuned looking down from above: without an altitude
                // spread the flock is a disc.
                b.yPrefFrac = 1.f + (draw(d + 8) * 2.f - 1.f) * params_.altitudeSpread;
                b.rollTrim = (draw(d + 9) * 2.f - 1.f) * 0.03f;
                b.wingAsym = (draw(d + 10) * 2.f - 1.f) * 0.03f;
                b.decisionPeriod = 0.25f + 0.65f * draw(d + 11);
                b.gaitPhase = draw(d + 12);
                b.idlePhase[0] = draw(d + 13);
                b.idlePhase[1] = draw(d + 14);
                b.idlePhase[2] = draw(d + 15);
                b.idlePhase[3] = draw(d + 16);

                const float wx = draw(d + 17) * 2.f - 1.f;
                const float wy = draw(d + 18) * 2.f - 1.f;
                const float wz = draw(d + 19) * 2.f - 1.f;
                b.wanderAxis = fauna::detail::safeNormalized({wx, wy, wz}, {0, 1, 0});

                const float hx = draw(d + 20) * 2.f - 1.f;
                const float hz = draw(d + 21) * 2.f - 1.f;
                b.fwd = fauna::detail::safeNormalized({hx, 0.f, hz}, {0, 0, 1});
                b.wanderDir = b.fwd;

                // Two birds therefore react to the same stimulus 0–0.9 s apart.
                // That stagger IS what a flock's ripple looks like, it costs
                // nothing, and it is the cheapest general answer to "N agents
                // reacting on the same frame".
                b.nextDecision = draw(d + 22) * b.decisionPeriod;

                b.role = (i < leaderCount_)                  ? BirdRole::Leader
                         : (i < leaderCount_ + lonerCount_)  ? BirdRole::Loner
                                                             : BirdRole::Follower;

                b.perchInterval = math::lerp(params_.perchIntervalMin, params_.perchIntervalMax,
                                             std::clamp(1.6f - b.restlessness, 0.f, 1.f) / 1.1f);
                b.restInterval = math::lerp(params_.restIntervalMin, params_.restIntervalMax,
                                            std::clamp(b.restlessness, 0.f, 1.6f) / 1.6f);
                b.perchUrge = draw(d + 23) * 0.6f;
                b.targetSpeed = params_.cruiseSpeed * b.speedScale;

                // Initial placement: a shell around home, so the flock is
                // already a flock on frame 0 rather than a point that explodes.
                const float ang = draw(d + 0) * math::TWO_PI;
                const float rad = params_.roamRadius * (0.15f + 0.45f * draw(d + 4));
                Dyn& dyn = prev_[static_cast<std::size_t>(i)];
                dyn.pos.set(params_.home.x + std::cos(ang) * rad,
                            params_.home.y + (draw(d + 8) * 2.f - 1.f) * params_.cruiseAltitude * 0.25f,
                            params_.home.z + std::sin(ang) * rad);
                dyn.vel = b.fwd;
                dyn.vel.multiplyScalar(b.targetSpeed);
                next_[static_cast<std::size_t>(i)] = dyn;

                // Four incommensurate idle periods per bird, so two perched
                // birds essentially never coincide and a row on a ridge never
                // twitches in unison.
                b.saccadeNext = b.idlePhase[1] * 2.f;
                b.flickAt = 2.f + b.idlePhase[1] * 4.f;
                b.shuffleAt = 8.f + b.idlePhase[2] * 12.f;
                b.preenAt = 15.f + b.idlePhase[3] * 45.f;
            }
        }

        [[nodiscard]] float draw(std::size_t k) const {

            return k < rngDraws_.size() ? rngDraws_[k] : 0.5f;
        }

        [[nodiscard]] float baseBeatHz() const {

            if (params_.wingbeatHz > 0.f) return params_.wingbeatHz;
            // Allometric: a heavier bird beats slower, as f ∝ m^(-1/3). It is one
            // pow() at construction and it means "make them pigeons" is a mass,
            // not a frequency nobody knows the value of.
            return 8.5f * std::pow(params_.massKg / 0.078f, -1.f / 3.f);
        }

        // ── World ↔ local ────────────────────────────────────────────────
        //
        // The simulation is world-space; the vertices have to land in the node's
        // own space. Rather than transforming 94 positions and 94 normals per
        // bird, the BirdPose itself is carried into local space — four vectors
        // and a scalar — and poseBird() then writes local coordinates directly.
        // For a rigid or uniformly-scaled parent the two are identical; under
        // non-uniform scale the basis is re-orthonormalised by
        // transformDirection() and the birds shear, which is documented in the
        // banner and not worth 188 transforms a bird to fix.
        void refreshWorldInverse() {

            const Matrix4& w = *matrixWorld;

            identityWorld_ = true;
            for (unsigned int k = 0; k < 16u; ++k) {
                const float want = (k % 5u == 0u) ? 1.f : 0.f;
                if (std::abs(w.elements[k] - want) > 1e-6f) {
                    identityWorld_ = false;
                    break;
                }
            }
            if (identityWorld_) {
                invScale_ = 1.f;
                return;
            }

            invWorld_.copy(w).invert();

            Vector3 s;
            s.setFromMatrixScale(invWorld_);
            invScale_ = (s.x + s.y + s.z) / 3.f;
            if (!std::isfinite(invScale_) || invScale_ <= 0.f) invScale_ = 1.f;
        }

        void toLocal(fauna::BirdPose& pose) const {

            if (identityWorld_) return;

            pose.pos.applyMatrix4(invWorld_);
            pose.bx.transformDirection(invWorld_);
            pose.by.transformDirection(invWorld_);
            pose.bz.transformDirection(invWorld_);
            pose.footWorld[0].applyMatrix4(invWorld_);
            pose.footWorld[1].applyMatrix4(invWorld_);
            pose.scale *= invScale_;
        }

        // ── Scene queries with an empty-index answer ─────────────────────
        //
        // A never-baked index answers heightAt() with 0, which for a scene whose
        // ground is at y = 0 is right by accident and for one at y = 40 is a
        // floor forty metres below the birds. Substituting home.y - 100 instead
        // makes the ground force provably inert until there is a real
        // heightfield to consult — "no data" has to mean "no force", never
        // "force toward zero".
        [[nodiscard]] float groundAt(float x, float z) const {

            if (!hasField_) return params_.home.y - 100.f;
            return perch_.heightAt(x, z);
        }

        // The floor a given bird is entitled to, in world Y. ONE definition,
        // consulted by both the ground force in steer() and the hard clamp in
        // the integrators — when those two disagreed, the force pushed up from
        // one height while the clamp held at another and birds sat vibrating on
        // the seam.
        //
        // A committing bird gets a LOWER floor: its perch itself, with 10 cm of
        // slack for the flare. Nothing may stop the descent above the perch;
        // below it, everything does.
        [[nodiscard]] float floorFor(const Bird& b, float x, float z) const {

            const float g = groundAt(x, z);
            if (b.state == BirdState::Approach || b.state == BirdState::Flare) {
                return std::min(g, b.spotPos.y) - 0.10f;
            }
            return g;
        }

        // THE FLOOR IS A CONSTRAINT AND A FORCE CANNOT ENFORCE ONE. The ground
        // spring in steer() recovers a bird descending at 7 m/s over roughly a
        // fifth of a second, and in that fifth of a second it is 0.8 m under the
        // mesh with its body visibly through the floor — which is one of the
        // exact tells this whole subsystem exists to avoid.
        //
        // So after integrating, a flying bird is placed back on the floor and
        // its DOWNWARD velocity is removed (the horizontal component is kept, so
        // it skims rather than stopping dead, and a bird that was climbing is
        // untouched). Above the floor this is a no-op, so ordinary flight never
        // sees it.
        void clampToFloor(const Bird& b, Dyn& q) const {

            if (!hasField_) return;

            const float floorY = floorFor(b, q.pos.x, q.pos.z);
            if (q.pos.y >= floorY) return;

            q.pos.y = floorY;
            if (q.vel.y < 0.f) q.vel.y = 0.f;
        }

        // THE ALTITUDE PREFERENCE NEEDS ITS OWN REFERENCE, AND THIS IS NOT A
        // TIDINESS ARGUMENT. groundAt() answers "how low may I go", so with no
        // heightfield it answers home.y − 100 precisely in order to make the
        // ground force inert. Feeding that same number to the altitude spring
        // makes the preferred height home.y − 100 + cruiseAltitude, and the
        // whole flock calmly descends eighty metres into the basement and
        // loiters there — no NaN, no warning, nothing in the logs, and the
        // symptom is "my birds vanished". Without a bake the notional ground is
        // one cruiseAltitude below `home`, so the flock loiters at `home`.
        [[nodiscard]] float altitudeRef(float x, float z) const {

            if (!hasField_) return params_.home.y - params_.cruiseAltitude;
            return perch_.heightAt(x, z);
        }

        // ── Per-frame aggregates ─────────────────────────────────────────
        void updateAggregates() {

            centroid_.set(0, 0, 0);
            meanVel_.set(0, 0, 0);
            perchedNow_ = 0;
            committed_ = 0;

            for (int i = 0; i < count_; ++i) {
                const Dyn& d = prev_[static_cast<std::size_t>(i)];
                centroid_.add(d.pos);
                meanVel_.add(d.vel);
                const BirdState s = birds_[static_cast<std::size_t>(i)].state;
                if (s == BirdState::Perched) ++perchedNow_;
                if (s == BirdState::Perched || s == BirdState::Approach || s == BirdState::Flare) ++committed_;
            }
            const float inv = 1.f / static_cast<float>(count_);
            centroid_.multiplyScalar(inv);
            meanVel_.multiplyScalar(inv);

            // HOME DRIFT. Three incommensurate sinusoids on three fixed axes.
            // Without it the flock orbits one point for ever and the territory
            // reads as a bowl someone drew; with it the whole population
            // migrates slowly across the scene the way a real one does.
            //
            // 0.35 · roamRadius is the amplitude of the SUM, not of each term.
            // Three independent 0.35 R sinusoids peak at 1.05 R of drift, which
            // stacks on top of the birds' own ~1.0 R spread around the drifted
            // centre and puts the outliers past 1.9 R from `home` — outside the
            // 1.5 R containment the flock is supposed to honour, and far enough
            // to walk a background flock out of the scene it was placed in.
            homeDrifted_ = params_.home;
            static constexpr std::array<float, 3> rate{1.0f, 0.43f, 0.27f};
            for (int m = 0; m < 3; ++m) {
                const std::size_t mi = static_cast<std::size_t>(m);
                const float phase = math::TWO_PI * params_.homeDriftRate * time_ * rate[mi] + driftPhase_[mi];
                homeDrifted_.addScaledVector(driftAxis_[mi],
                                             (0.35f / 3.f) * params_.roamRadius * std::sin(phase));
            }

            // Decayed ONCE per frame, not once per bird: a per-bird decay makes
            // the contagion window N times shorter and therefore makes the
            // behaviour depend on bird count, which is the one thing this
            // subsystem's parameters are meant not to.
            if (launchedRecently_ > 0.f) launchedRecently_ = std::max(0.f, launchedRecently_ - lastDt_);

            if (time_ >= nextLeaderVote_) {
                nextLeaderVote_ = time_ + params_.leaderReassign;
                reassignLeaders();
            }

            if (claimedBy_.size() != perch_.spots().size()) syncClaims();
        }

        // LEADERSHIP IS A POSITION, NOT A TRAIT. A fixed Leader personality
        // gives the flock a permanent visible boss that the eye picks up within
        // a minute of watching; re-electing whoever is furthest forward along
        // the mean velocity produces the same steering benefit and none of that.
        void reassignLeaders() {

            if (leaderCount_ <= 0) return;

            Vector3 dir = fauna::detail::safeNormalized(meanVel_, {0, 0, 1});

            for (int i = leaderCount_ + lonerCount_; i < count_; ++i) {
                birds_[static_cast<std::size_t>(i)].role = BirdRole::Follower;
            }
            for (int i = 0; i < leaderCount_; ++i) {
                birds_[static_cast<std::size_t>(i)].role = BirdRole::Follower;
            }

            for (int k = 0; k < leaderCount_; ++k) {

                int best = -1;
                float bestScore = -std::numeric_limits<float>::infinity();

                for (int i = 0; i < count_; ++i) {
                    Bird& b = birds_[static_cast<std::size_t>(i)];
                    if (b.role != BirdRole::Follower) continue;// Loners never lead; already-picked leaders skip
                    if (b.state == BirdState::Perched) continue;

                    Vector3 rel = prev_[static_cast<std::size_t>(i)].pos;
                    rel.sub(centroid_);
                    const float score = rel.dot(dir);
                    // Ties break on the LOWEST index, which is why this is a
                    // strict >: the first bird to reach a score keeps it.
                    if (score > bestScore) {
                        bestScore = score;
                        best = i;
                    }
                }
                if (best < 0) break;
                birds_[static_cast<std::size_t>(best)].role = BirdRole::Leader;
            }
        }

        // ── Neighbours (§5.3, from prev_) ────────────────────────────────
        void gatherNeighbours(int i) {

            const int k = params_.neighbourCount;
            nbrCount_[static_cast<std::size_t>(i)] = 0;
            if (k <= 0) return;

            const Dyn& di = prev_[static_cast<std::size_t>(i)];
            const Bird& bi = birds_[static_cast<std::size_t>(i)];
            const float r2 = params_.neighbourRadius * params_.neighbourRadius;
            const float blind = -std::cos(params_.rearBlindAngle);

            std::array<float, kMaxNeighbours> bestD{};
            int n = 0;
            const std::size_t base = static_cast<std::size_t>(i) * kMaxNeighbours;

            for (int j = 0; j < count_; ++j) {

                if (j == i) continue;

                Vector3 off = prev_[static_cast<std::size_t>(j)].pos;
                off.sub(di.pos);
                const float d2 = off.lengthSq();
                if (d2 > r2 || d2 < 1e-12f) continue;

                // One dot product, and besides matching what a bird can actually
                // see it kills the artificial conga lines that metric boids form
                // when a follower locks onto the tail of the bird in front.
                const float inv = 1.f / std::sqrt(d2);
                if ((off.x * bi.fwd.x + off.y * bi.fwd.y + off.z * bi.fwd.z) * inv < blind) continue;

                // Fixed-size insertion sort, ties broken by ascending bird index
                // — which falls out of `<` rather than `<=` on an ascending scan.
                int slot = n;
                while (slot > 0 && d2 < bestD[static_cast<std::size_t>(slot - 1)]) --slot;
                if (slot >= k) continue;

                const int last = std::min(n, k - 1);
                for (int m = last; m > slot; --m) {
                    bestD[static_cast<std::size_t>(m)] = bestD[static_cast<std::size_t>(m - 1)];
                    nbrIdx_[base + static_cast<std::size_t>(m)] = nbrIdx_[base + static_cast<std::size_t>(m - 1)];
                }
                bestD[static_cast<std::size_t>(slot)] = d2;
                nbrIdx_[base + static_cast<std::size_t>(slot)] = j;
                if (n < k) ++n;
            }

            nbrCount_[static_cast<std::size_t>(i)] = n;
        }

        // ── Steering (§5.3) ──────────────────────────────────────────────
        //
        // The ten forces are accumulated in the numbered order, each into its
        // own named intermediate. Reordering them changes float accumulation and
        // breaks the determinism contract for no benefit whatsoever.
        [[nodiscard]] Vector3 steer(int i) {

            const Bird& b = birds_[static_cast<std::size_t>(i)];
            const Dyn& d = prev_[static_cast<std::size_t>(i)];

            Vector3 a{0, 0, 0};

            if (b.state == BirdState::Perched) return a;

            const std::size_t base = static_cast<std::size_t>(i) * kMaxNeighbours;
            const int n = nbrCount_[static_cast<std::size_t>(i)];

            // A COMMITTED BIRD IS NO LONGER FLOCKING, AND THE ALTITUDE SPRING IS
            // THE ONE THAT HAS TO GO. A perch on the ground sits a full
            // cruiseAltitude below the preferred height, so at the default
            // weights the altitude force is 5.4 m/s² UP against the goal force's
            // 4.8 m/s² down: every approach is flown, every approach is aborted,
            // perchedCount() reads 0 for ever, and nothing anywhere logs a word.
            // The remaining flock forces are attenuated rather than cut, so a
            // landing bird still avoids its neighbours on the way in.
            const bool committing = (b.state == BirdState::Approach || b.state == BirdState::Flare);
            const float flockScale = committing ? 0.3f : 1.f;

            // 1 — separation
            Vector3 fSep{0, 0, 0};
            for (int m = 0; m < n; ++m) {
                const int j = nbrIdx_[base + static_cast<std::size_t>(m)];
                Vector3 off = d.pos;
                off.sub(prev_[static_cast<std::size_t>(j)].pos);
                const float dist = off.length();
                if (dist >= params_.separationDistance || dist < 1e-4f) continue;
                const float t = 1.f - dist / params_.separationDistance;
                fSep.addScaledVector(off, t * t / std::max(dist, 0.05f));
            }
            a.addScaledVector(fSep, params_.wSeparation);

            // 2 — alignment
            Vector3 fAli{0, 0, 0};
            if (n > 0 && b.role != BirdRole::Loner) {
                for (int m = 0; m < n; ++m) {
                    fAli.add(prev_[static_cast<std::size_t>(nbrIdx_[base + static_cast<std::size_t>(m)])].vel);
                }
                fAli.multiplyScalar(1.f / static_cast<float>(n));
                fAli.sub(d.vel);
                fAli.multiplyScalar(b.sociability);
            }
            a.addScaledVector(fAli, params_.wAlignment * flockScale);

            // 3 — cohesion. Leaders skip it entirely (they are the front, and a
            // front that is pulled back into the centroid is not a front).
            Vector3 fCoh{0, 0, 0};
            if (n > 0 && b.role != BirdRole::Leader) {
                Vector3 c{0, 0, 0};
                float wsum = 0.f;
                for (int m = 0; m < n; ++m) {
                    const int j = nbrIdx_[base + static_cast<std::size_t>(m)];
                    const float w = birds_[static_cast<std::size_t>(j)].role == BirdRole::Leader ? 3.f : 1.f;
                    c.addScaledVector(prev_[static_cast<std::size_t>(j)].pos, w);
                    wsum += w;
                }
                c.multiplyScalar(1.f / wsum);
                c.sub(d.pos);
                // THE DEAD ZONE IS NOT OPTIONAL. Without it a bird sitting on
                // the centroid gets a force that flips sign every frame and
                // jitters at the integration rate, which at 144 Hz is a visible
                // shimmer through the whole flock.
                const float dist = c.length();
                if (dist > params_.cohesionDeadZone) {
                    c.multiplyScalar((dist - params_.cohesionDeadZone) / dist);
                    c.multiplyScalar(b.sociability * (b.role == BirdRole::Loner ? 0.25f : 1.f));
                    fCoh = c;
                }
            }
            a.addScaledVector(fCoh, params_.wCohesion * flockScale);

            // 4 — wander
            Vector3 fWan = b.wanderDir;
            fWan.multiplyScalar(b.role == BirdRole::Leader ? 2.5f : 1.f);
            a.addScaledVector(fWan, params_.wWander * flockScale);

            // 5 — altitude, a spring/damper rather than a target: a hard
            // altitude hold makes every bird sit on one plane, which is what a
            // flock never does.
            Vector3 fAlt{0, 0, 0};
            {
                const float yPref = altitudeRef(d.pos.x, d.pos.z) + params_.cruiseAltitude * b.yPrefFrac;
                const float omega = 0.8f;
                fAlt.y = omega * omega * (yPref - d.pos.y) - 2.f * 0.9f * omega * d.vel.y;
            }
            a.addScaledVector(fAlt, committing ? 0.f : params_.wAltitude);

            // 6 — bounds. QUADRATIC, so there is no wall to bounce off — only a
            // region that gets progressively less comfortable.
            //
            // Plus a RADIAL DAMPER, which is not decoration. A pure position
            // spring overshoots by v²/2a whatever its gain: at 9 m/s against the
            // 9 m/s² the anisotropic clamp allows, that is nine metres past
            // wherever the force finally saturates, and the flock's true extent
            // ends up a tuning accident rather than a property. Damping the
            // OUTWARD RATE — and only the outward one, so a bird flying home is
            // never slowed — bounds the excursion directly. It is still a region
            // that gets progressively less comfortable; the discomfort now
            // includes not being able to keep going.
            Vector3 fBnd{0, 0, 0};
            {
                Vector3 toHome = homeDrifted_;
                toHome.sub(d.pos);
                toHome.y = 0.f;// altitude is force 5's job
                const float dist = toHome.length();
                const float soft = 0.75f * params_.roamRadius;
                if (dist > soft) {
                    const float t = (dist - soft) / (0.25f * params_.roamRadius);
                    const float inv = 1.f / std::max(dist, 1e-3f);
                    fBnd = toHome;
                    fBnd.multiplyScalar(t * t * inv);

                    const float outward = -(toHome.x * d.vel.x + toHome.z * d.vel.z) * inv;
                    if (outward > 0.f) {
                        fBnd.addScaledVector(toHome, 1.4f * outward * std::min(t, 3.f) * inv);
                    }
                }
            }
            a.addScaledVector(fBnd, params_.wBounds * flockScale);

            // 7 — obstacle
            Vector3 fObs{0, 0, 0};
            if (hasField_) {
                Vector3 look = d.pos;
                look.addScaledVector(d.vel, params_.lookaheadTime);
                const float c = perch_.clearanceAt(look);
                if (c < params_.obstacleMargin) {
                    Vector3 grad;
                    perch_.clearanceGradient(look, grad);
                    const float t = 1.f - c / params_.obstacleMargin;
                    fObs = grad;
                    fObs.multiplyScalar(t * t);
                }
            }
            a.addScaledVector(fObs, params_.wObstacle);

            // 8 — ground. A COMMITTED BIRD STILL GETS A FLOOR, just a lower one.
            // Suppressing this outright during a landing is what lets a bird
            // that misses its perch keep descending: it is no longer flocking,
            // nothing else looks down, and it ends up ten metres under the
            // terrain still politely steering toward a branch above it.
            Vector3 fGrd{0, 0, 0};
            float belowFloor = 0.f;
            if (hasField_) {
                const float floorY = floorFor(b, d.pos.x, d.pos.z);
                const float clearance = committing ? 0.f : params_.minGroundClearance;
                const float h = d.pos.y - floorY;
                if (h < clearance || h < 0.f) {
                    const float span = std::max(clearance, 0.5f);
                    const float t = std::clamp(1.f - h / span, 0.f, 1.f);
                    fGrd.y = 4.f * t;
                }
                belowFloor = std::max(0.f, -h);
            }
            a.addScaledVector(fGrd, params_.wGround);

            // 9 — goal (Approach/Flare only)
            Vector3 fGoal{0, 0, 0};
            if (committing) {
                Vector3 target = goalTarget(b);
                target.sub(d.pos);
                const float dist = target.length();
                if (dist > 1e-3f) {
                    fGoal = target;
                    // Saturating at 8 m of error rather than 4: the capture, not
                    // the cruise leg, is what sets this number. At 4 m the goal
                    // force tops out at 4.8 m/s², which cannot bend a 6 m/s
                    // approach onto a point, and nine out of ten approaches
                    // become fly-bys.
                    fGoal.multiplyScalar(std::min(dist, 8.f) / dist);
                }
            }
            a.addScaledVector(fGoal, params_.wGoal);

            // 10 — observer. Birds are shy of the camera, very slightly. Any
            // more and the flock visibly parts around the viewer, which reads as
            // the scene knowing where you are.
            Vector3 fObserver{0, 0, 0};
            if (observer_) {
                Vector3 eye;
                eye.setFromMatrixPosition(*observer_->matrixWorld);
                Vector3 away = d.pos;
                away.sub(eye);
                const float dist = away.length();
                if (dist < 8.f && dist > 1e-3f) {
                    fObserver = away;
                    fObserver.multiplyScalar((1.f - dist / 8.f) / dist);
                }
            }
            a.addScaledVector(fObserver, params_.wObserver);

            // Evade overrides the lot with a hard turn away from the epicentre.
            if (b.state == BirdState::Evade) {

                a.multiplyScalar(0.25f);

                // …EXCEPT THE TERRITORY, WHICH IS RESTORED TO FULL STRENGTH. A
                // startled bird runs at 1.5× cruise for up to a second; with the
                // bounds force quartered along with everything else that is
                // twenty metres of unopposed flight outward, and a flock
                // startled repeatedly walks itself out of the scene one flush at
                // a time. A real bird flushes WITHIN its territory — it turns
                // away from the threat, it does not emigrate.
                a.addScaledVector(fBnd, params_.wBounds * flockScale * 0.75f);

                a.addScaledVector(b.evadeAway, params_.maxAccelLateral * (0.8f + 0.4f * b.startleStrength));
                a.y += 2.5f;
            }

            // ANISOTROPIC CLAMP. Birds turn hard and accelerate slowly, and the
            // asymmetry is what removes the boids sprint-stop yo-yo: an
            // isotropic clamp lets a bird go 4.5 → 17 m/s in half a second,
            // which reads as surging.
            const float along = std::clamp(a.dot(b.fwd), -params_.maxAccelAlong, params_.maxAccelAlong);
            Vector3 cross = a;
            cross.addScaledVector(b.fwd, -a.dot(b.fwd));
            cross.clampLength(0.f, params_.maxAccelLateral);
            a = cross;
            a.addScaledVector(b.fwd, along);

            // THE FLOOR IS A CONSTRAINT, NOT A PREFERENCE. Below the terrain the
            // ground force is re-added outside the anisotropic clamp, because
            // inside it a bird already descending at 7 m/s recovers over 2.7 m
            // and spends that whole time under the mesh. Above the floor this
            // term is exactly zero, so the ordinary steering is untouched.
            if (belowFloor > 0.f) {
                a.y += params_.wGround * 4.f * std::min(1.f + belowFloor, 6.f);
            }

            // A SOFT LAW, NEVER A CLAMP. Speed breathes ±8 % instead of pinning
            // every bird at one obviously-authored rate.
            const float speed = d.vel.length();
            float want = b.targetSpeed;
            if (b.state == BirdState::Evade) want *= 1.5f;

            if (committing) {
                // A CLOSING LAW, not a fraction of cruise speed. At 6.75 m/s and
                // the goal force's 4.8 m/s² the turn radius is 9.5 m, so a bird
                // aimed at a perch simply orbits it at nine metres and never gets
                // inside flareDistance — it flies a perfect approach for ever.
                // Scaling the demand with range shrinks the radius as it closes
                // (1.9 m at 3 m/s), which is the difference between a landing and
                // a holding pattern.
                const float dist = d.pos.distanceTo(goalTarget(b));
                want = std::clamp(0.9f * dist, 1.2f, b.targetSpeed);
            } else {
                // minSpeed is a floor on the DEMAND, not a clamp on the state: a
                // bird genuinely slowed below it keeps its real speed and the
                // drag pulls it back up, which is a stall recovery rather than a
                // teleport to the minimum. It does NOT apply to a landing, where
                // going slow is the entire objective.
                want = std::max(want, params_.minSpeed);
            }
            a.addScaledVector(b.fwd, params_.speedDrag * (want - speed));

            return a;
        }

        // Where the BODY sits when the feet are on `spot`: a leg-length above
        // it along the spot normal. Every landing target in the system is
        // expressed this way, so the bird arrives standing rather than arriving
        // with its belly in the branch.
        [[nodiscard]] Vector3 standPoint(const Bird& b) const {

            Vector3 p = b.spotPos;
            p.addScaledVector(b.spotNormal, standHeight(b));
            return p;
        }

        [[nodiscard]] float standHeight(const Bird& b) const {

            return (0.72f * params_.shape.bodyRadius + tmpl_.legLength) * b.size;
        }

        // THE GATE IS A WAYPOINT, AND SOMETHING HAS TO HAND OFF FROM IT. Steering
        // at the gate for the whole approach means the bird converges on a point
        // 2.2–3 m short of the perch and settles into an orbit around it: the
        // Approach → Flare test measures range to the SPOT, which never falls
        // below flareDistance, so the state machine runs a flawless approach
        // that can never complete. `gatePassed` latches so the target cannot
        // flicker back and forth across the handoff radius.
        [[nodiscard]] Vector3 goalTarget(const Bird& b) const {

            if (b.state == BirdState::Flare || b.gatePassed) return standPoint(b);
            return b.gate;
        }

        // ── Decisions (§5.1, §5.2) ───────────────────────────────────────
        void decide(int i, float dt) {

            Bird& b = birds_[static_cast<std::size_t>(i)];
            const Dyn& d = prev_[static_cast<std::size_t>(i)];

            b.stateTime += dt;
            if (b.perchSuppress > 0.f) b.perchSuppress = std::max(0.f, b.perchSuppress - dt);

            // The startle wave is the ONLY thing that bypasses the decision
            // cadence, and even it carries the per-bird latency drawn in
            // startle().
            if (b.reactAt >= 0.f && time_ >= b.reactAt) {
                b.reactAt = -1.f;
                triggerEvade(i, b, d, b.startleFrom);
                return;
            }

            // Disturbance source: the same test every field guide states as a
            // flight-initiation distance, scaled by the bird's own boldness so a
            // walker flushes the bold ones last.
            if (disturbance_) {
                Vector3 src;
                src.setFromMatrixPosition(*disturbance_->matrixWorld);
                const float fid = params_.flightInitiationDistance * b.boldness;
                if (d.pos.distanceToSquared(src) < fid * fid) {
                    triggerEvade(i, b, d, src);
                    return;
                }
            }

            if (time_ < b.nextDecision) return;

            // += rather than = time_ + period: the phase a bird was given at
            // construction is the whole point, and resetting from `now` would
            // let two birds converge onto the same tick and stay there.
            b.nextDecision += b.decisionPeriod * (isFar(i) ? 2.f : 1.f);
            if (b.nextDecision <= time_) b.nextDecision = time_ + b.decisionPeriod;

            switch (b.state) {
                case BirdState::Cruise: decideCruise(i, b, d); break;
                case BirdState::Approach: decideApproach(i, b, d); break;
                case BirdState::Perched: decidePerched(i, b, d); break;
                default: break;
            }
        }

        void decideCruise(int i, Bird& b, const Dyn& d) {

            // Wander turn: a per-bird unit vector rotating about a per-bird
            // axis. Drawn on the decision tick so it is a pure function of
            // (seed, bird, sequence), never of frame count.
            const float turn = (roll(b, i) * 2.f - 1.f) * 0.35f * b.decisionPeriod;
            b.wanderDir.applyAxisAngle(b.wanderAxis, turn);
            b.wanderDir = fauna::detail::safeNormalized(b.wanderDir, b.fwd);

            if (!params_.perching || b.perchSuppress > 0.f) return;
            if (b.perchUrge <= 1.f) return;
            if (perch_.spots().empty()) return;

            // The sky is never allowed to empty. maxPerchedFraction counts every
            // bird already committed to a landing, not just the ones on the
            // ground — otherwise N birds all commit in the same second and the
            // fraction only bites once they have all arrived.
            const float frac = static_cast<float>(committed_) / static_cast<float>(count_);
            if (frac >= params_.maxPerchedFraction) return;

            if (!claimSpot(i, b, d)) {
                b.perchUrge = 0.6f;
                return;
            }
            b.state = BirdState::Approach;
            b.abortRolled = false;
            b.gatePassed = false;
            b.stateTime = 0.f;
        }

        void decideApproach(int i, Bird& b, const Dyn& d) {

            if (b.claim < 0) {
                abortApproach(b);
                return;
            }

            // REAL BIRDS GO AROUND CONSTANTLY. An approach that always succeeds
            // is one of the loudest tells in the whole system, and this is one
            // line to buy the opposite.
            //
            // ROLLED ONCE PER APPROACH, ON THE FIRST TICK AFTER COMMITTING, NOT
            // ONCE PER TICK. An approach spans six to eight decision ticks, so a
            // per-tick roll turns abortChance = 0.12 into a SIXTY-FOUR PER CENT
            // abort rate and the flock spends the session circling a perch it
            // never reaches. The parameter has to mean what its name says.
            if (!b.abortRolled) {
                b.abortRolled = true;
                if (roll(b, i) < params_.abortChance) {
                    abortApproach(b);
                    return;
                }
            }

            // Bearing is judged against the CURRENT goal — the gate first, the
            // perch after the handoff. Judging it against the perch while the
            // bird is still flying the gate leg aborts the very approaches whose
            // whole point is to arrive from below and short.
            Vector3 to = goalTarget(b);
            to.sub(d.pos);
            const float dist = to.length();
            if (b.gatePassed && dist > 1e-3f) {
                to.multiplyScalar(1.f / dist);
                if (to.dot(b.fwd) < 0.f && dist > params_.flareDistance * 2.f) {
                    abortApproach(b);
                    return;
                }
            }

            // A hard ceiling on the attempt. Everything above is a soft test and
            // a soft test can, on some geometry, never fire — and an approach
            // that never resolves is a bird permanently removed from the flock
            // with no state anyone would think to look at.
            if (b.stateTime > 12.f) abortApproach(b);
        }

        void decidePerched(int i, Bird& b, const Dyn& d) {

            (void) d;

            // Launch contagion — a neighbour left, so this bird thinks about
            // leaving. Positive feedback, damped by restIntervalMin and
            // maxPerchedFraction; §7.3's soak test is what catches a limit cycle
            // in it.
            if (launchedRecently_ > 0.f && roll(b, i) < params_.launchContagion) {
                const std::size_t base = static_cast<std::size_t>(i) * kMaxNeighbours;
                const int n = nbrCount_[static_cast<std::size_t>(i)];
                for (int m = 0; m < n; ++m) {
                    const int j = nbrIdx_[base + static_cast<std::size_t>(m)];
                    const Bird& o = birds_[static_cast<std::size_t>(j)];
                    if (o.state != BirdState::Launch) continue;
                    if (prev_[static_cast<std::size_t>(i)].pos.distanceToSquared(prev_[static_cast<std::size_t>(j)].pos) <
                        params_.contagionRadius * params_.contagionRadius) {
                        beginLaunch(b);
                        return;
                    }
                }
            }

            if (b.restUrge > 1.f) {
                beginLaunch(b);
                return;
            }

            if (!b.walker) return;

            // Ground micro-goals with vigilance pauses between them. Ground-
            // feeding birds spend more time with the head up than moving, and
            // getting that ratio wrong is what makes a walking bird look like a
            // wind-up toy.
            if (time_ < b.pauseUntil) return;
            if (b.anchor.distanceToSquared(b.groundGoal) > 0.04f) return;

            const float ang = roll(b, i) * math::TWO_PI;
            const float rad = 0.3f + 1.7f * roll(b, i);
            const float pause = 0.5f + 2.5f * roll(b, i);

            Vector3 g = b.spotPos;
            g.x += std::cos(ang) * rad;
            g.z += std::sin(ang) * rad;
            // Never wander further than 0.8 m from the spot the bake vouched
            // for: outside it there is no evidence the surface even exists.
            Vector3 off = g;
            off.sub(b.spotPos);
            off.y = 0.f;
            const float dist = off.length();
            if (dist > 0.8f) off.multiplyScalar(0.8f / dist);
            b.groundGoal = b.spotPos;
            b.groundGoal.add(off);
            b.pauseUntil = time_ + pause;
        }

        void triggerEvade(int i, Bird& b, const Dyn& d, const Vector3& from) {

            releaseClaim(b);

            Vector3 away = d.pos;
            away.sub(from);
            away.y = 0.f;
            b.evadeAway = fauna::detail::safeNormalized(away, b.fwd);

            if (b.state == BirdState::Perched) {
                // A perched bird cannot simply turn: it has to get off the
                // branch first, so the startle routes through the same launch
                // the bird would have used voluntarily — with the crouch cut
                // short, which is what an alarmed take-off actually looks like.
                beginLaunch(b);
                b.stateTime = 0.10f;
            } else {
                b.state = BirdState::Evade;
                b.stateTime = 0.f;
                b.feetPlanted = false;
            }

            b.evadeUntil = time_ + 0.4f + 0.6f * roll(b, i);
            b.perchSuppress = std::max(b.perchSuppress,
                                       rollRange(b, i, params_.postFlushCalmMin, params_.postFlushCalmMax));
            b.perchUrge = 0.f;
        }

        void beginLaunch(Bird& b) {

            releaseClaim(b);
            b.state = BirdState::Launch;
            b.stateTime = 0.f;
            b.restUrge = 0.f;
            b.perchUrge = 0.f;
            b.perchSuppress = std::max(b.perchSuppress, 1.5f);
            launchedRecently_ = 0.6f;
        }

        void abortApproach(Bird& b) {

            releaseClaim(b);
            b.state = BirdState::Cruise;
            b.stateTime = 0.f;
            b.perchUrge = 0.6f;
        }

        // ── Perch claims and selection (§5.1) ────────────────────────────
        void syncClaims() {

            claimedBy_.assign(perch_.spots().size(), -1);
            for (auto& b : birds_) {
                if (b.claim < 0) continue;
                b.claim = -1;
                if (b.state == BirdState::Approach || b.state == BirdState::Flare) {
                    b.state = BirdState::Cruise;
                    b.stateTime = 0.f;
                } else if (b.state == BirdState::Perched) {
                    // A bird already standing on a spot that no longer exists
                    // keeps standing there until its rest urge fires. That is
                    // the documented worst case of a stale bake: a bird in
                    // mid-air, not a dereference of freed geometry.
                    b.restUrge = std::max(b.restUrge, 0.8f);
                }
            }
        }

        void releaseAllClaims() {

            for (auto& b : birds_) releaseClaim(b);
        }

        void releaseClaim(Bird& b) {

            if (b.claim >= 0 && static_cast<std::size_t>(b.claim) < claimedBy_.size()) {
                claimedBy_[static_cast<std::size_t>(b.claim)] = -1;
            }
            b.claim = -1;
        }

        bool claimSpot(int i, Bird& b, const Dyn& d) {

            spotScratch_.clear();
            perch_.querySpots(d.pos, params_.perchSearchRadius, spotScratch_);
            if (spotScratch_.empty()) return false;

            const auto& spots = perch_.spots();
            const bool wantGround = roll(b, i) < params_.groundBias;

            int best = -1;
            float bestScore = std::numeric_limits<float>::infinity();

            // The ≤ 8 nearest CLAIMABLE spots, scored. Bearing is gated at 75°
            // of the heading: a bird does not turn round to land, it picks
            // somewhere it was already going.
            int considered = 0;
            for (const int s : spotScratch_) {

                if (considered >= 8) break;
                const std::size_t si = static_cast<std::size_t>(s);
                if (si >= claimedBy_.size() || claimedBy_[si] >= 0) continue;

                // A BIRD DOES NOT LEAVE ITS TERRITORY TO LAND. perchSearchRadius
                // is measured from the BIRD, so without this a bird already out
                // near the soft boundary can commit to a spot another 32 m
                // beyond it — and while it is committed the bounds force is
                // attenuated, so nothing pulls it home until it has arrived.
                // Repeat that and an ambient flock walks out of the scene it was
                // placed in, one landing at a time.
                Vector3 fromHome = spots[si].position;
                fromHome.sub(homeDrifted_);
                fromHome.y = 0.f;
                if (fromHome.lengthSq() > params_.roamRadius * params_.roamRadius) continue;

                Vector3 to = spots[si].position;
                to.sub(d.pos);
                const float dist = to.length();
                if (dist < 1e-3f) continue;
                if (to.dot(b.fwd) / dist < 0.2588f) continue;// cos 75°
                ++considered;

                const float occ = occupancy(s);
                const float lonerBias = (b.role == BirdRole::Loner) ? 3.f : 1.f;
                const float jitter = 1.f + 0.4f * roll(b, i);
                float score = dist * dist * (1.f + 3.f * occ * lonerBias) * jitter;
                if (wantGround && !spots[si].ground) score *= 2.5f;
                // Contagion, the other way round: a spot near birds already down
                // is MORE attractive, up to perchContagion.
                score *= 1.f - params_.perchContagion * std::min(occ, 1.f) * 0.5f;

                if (score < bestScore) {
                    bestScore = score;
                    best = s;
                }
            }

            if (best < 0) return false;

            const std::size_t bi = static_cast<std::size_t>(best);
            claimedBy_[bi] = i;
            b.claim = best;
            b.spotNormal = spots[bi].normal;
            b.spotWalkable = spots[bi].walkable;

            // ±0.06 m so birds are not standing on exact bake centres. An
            // unjittered flock lands on a lattice, and a lattice is the one
            // thing no observer ever mistakes for wildlife.
            const float jx = (roll(b, i) * 2.f - 1.f) * 0.06f;
            const float jz = (roll(b, i) * 2.f - 1.f) * 0.06f;
            b.spotPos = spots[bi].position;
            b.spotPos.x += jx;
            b.spotPos.z += jz;

            b.walker = spots[bi].walkable && roll(b, i) < params_.groundBias;
            b.groundGoal = b.spotPos;
            b.gate = approachGate(b, d);
            return true;
        }

        [[nodiscard]] float occupancy(int spot) const {

            const auto& spots = perch_.spots();
            const std::size_t si = static_cast<std::size_t>(spot);
            if (si >= spots.size()) return 0.f;

            int taken = 0;
            int near = 0;
            const float r2 = params_.contagionRadius * params_.contagionRadius;
            for (std::size_t k = 0; k < spots.size(); ++k) {
                if (spots[k].position.distanceToSquared(spots[si].position) > r2) continue;
                ++near;
                if (k < claimedBy_.size() && claimedBy_[k] >= 0) ++taken;
            }
            return near > 0 ? static_cast<float>(taken) / static_cast<float>(near) : 0.f;
        }

        // THE GATE IS BELOW AND SHORT OF THE SPOT, NOT AT IT. Steering straight
        // at a perch produces a dive that has to be cancelled in the last metre;
        // aiming 0.45 m below and 2.2 m short converts the final approach into a
        // swoop UP, which is both what birds do and what makes the flare read as
        // deceleration rather than as a brake.
        [[nodiscard]] Vector3 approachGate(const Bird& b, const Dyn& d) const {

            Vector3 dir = b.spotPos;
            dir.sub(d.pos);
            dir.y = 0.f;
            dir = fauna::detail::safeNormalized(dir, b.fwd);

            Vector3 gate = standPoint(b);

            // A SPOT'S OWN COLUMN CANNOT ANSWER "IS THIS ELEVATED", AND ASKING
            // IT MADE THE SWOOP ABOVE DEAD CODE FOR EVERY PERCH IN THE SCENE.
            //
            // heightAt() reports the HIGHEST sampled surface in a column, and a
            // perch is by definition ON the highest surface of its own column.
            // So `spotPos.y - groundAt(spotPos)` is ~0 for a branch forty metres
            // up exactly as it is for a kerbstone, this test was never once
            // true, and every landing in the flock — rooftop, fencepost, bough —
            // flew the shallow ground glide. The symptom is subtle and damning:
            // birds descend onto branches like helicopters instead of arriving
            // from below, which is the single tell the gate exists to remove.
            //
            // Sample the ground the bird is ARRIVING OVER instead. Three metres
            // back along the approach, a post top stands four metres proud and a
            // ground spot is level with itself.
            const float backX = b.spotPos.x - dir.x * 3.0f;
            const float backZ = b.spotPos.z - dir.z * 3.0f;
            const bool elevated = hasField_ && (b.spotPos.y - groundAt(backX, backZ)) > 1.0f;

            if (elevated) {
                gate.addScaledVector(b.spotNormal, -0.45f);
                gate.addScaledVector(dir, -2.2f);
            } else {
                // Ground spot: 3 m out on a 6° descent.
                gate.addScaledVector(dir, -3.0f);
                gate.y += 3.0f * 0.105f;
            }
            return gate;
        }

        // ── Integration (§5.5 step 7) ────────────────────────────────────
        void integrate(int i, float dt) {

            Bird& b = birds_[static_cast<std::size_t>(i)];
            const Dyn& p = prev_[static_cast<std::size_t>(i)];
            Dyn& q = next_[static_cast<std::size_t>(i)];
            q = p;

            switch (b.state) {
                case BirdState::Perched: integratePerched(i, b, q, dt); break;
                case BirdState::Launch: integrateLaunch(i, b, p, q, dt); break;
                case BirdState::Flare: integrateFlare(i, b, p, q, dt); break;
                default: integrateFlight(i, b, p, q, dt); break;
            }

            advanceBeat(i, b, q, dt);
            advanceHead(i, b, dt);
            advanceSprings(i, b, dt);
            advanceState(i, b, q, dt);

            // ONE NaN VERTEX BLOWS THE BOUNDING SPHERE FOR THE REST OF THE
            // SESSION, and on Vulkan it can wreck a BLAS refit. Respawning is
            // ugly; a flock that silently disappears twenty minutes into a demo
            // is worse, and much harder to report.
            if (!fauna::detail::isFinite(q.pos) || !fauna::detail::isFinite(q.vel)) {
                q.pos = homeDrifted_;
                q.vel.set(0, 0, 0);
                b.state = BirdState::Cruise;
                b.stateTime = 0.f;
                b.feetPlanted = false;
                b.claim = -1;
                b.cyclePos = rnd01(params_.seed, static_cast<std::uint32_t>(i), 0xF10Cu);
            }
        }

        void integrateFlight(int i, Bird& b, const Dyn& p, Dyn& q, float dt) {

            Vector3 v = p.vel;
            v.addScaledVector(accel_[static_cast<std::size_t>(i)], dt);

            // Flap-bounding: lift withdrawn, gravity unopposed. Applied AFTER
            // the anisotropic clamp on purpose — it is not a steering choice,
            // it is what happens when the wings stop.
            if (b.bounding) v.y -= kGravity * dt * 0.55f;

            float speed = v.length();
            Vector3 dir = (speed > 1e-4f) ? Vector3{v.x / speed, v.y / speed, v.z / speed} : b.fwd;

            // HEADING SLEW CAP. Without it a large lateral acceleration can flip
            // the heading through 180° in one step and the bird visibly
            // teleports its own orientation.
            const float turnScale = (b.state == BirdState::Evade) ? 1.8f : 1.f;
            slewHeading(b, dir, params_.maxTurnRate * turnScale * dt);

            speed = std::clamp(speed, 0.f, params_.maxSpeed);
            // A stall is a real event, not an error: below minSpeed the bird
            // drops its nose, which the pitch term picks up for free.
            v = b.fwd;
            v.multiplyScalar(speed);

            q.vel = v;
            q.pos = p.pos;
            q.pos.addScaledVector(v, dt);
            clampToFloor(b, q);

            b.feetPlanted = false;
            b.legExtend = approachLegExtend(b, q);
            updateBankPitch(i, b, dt);
        }

        void integrateFlare(int i, Bird& b, const Dyn& p, Dyn& q, float dt) {

            // THE ONE PLACE A SPEED IS COMMANDED RATHER THAN STEERED. A soft
            // drag cannot guarantee an arrival speed, and a landing that
            // overshoots the branch by half a metre reads as broken software
            // rather than as a bad landing. kFlare is solved implicitly here:
            // the law itself carries v(flareDistance) → 0.4 m/s.
            Vector3 target = standPoint(b);
            Vector3 to = target;
            to.sub(p.pos);
            const float dist = to.length();

            const float want = 0.4f + (params_.cruiseSpeed * 0.55f - 0.4f) *
                                              std::clamp(dist / params_.flareDistance, 0.f, 1.f);

            Vector3 dir = fauna::detail::safeNormalized(to, b.fwd);
            slewHeading(b, dir, params_.maxTurnRate * 2.f * dt);

            float speed = p.vel.length();
            speed = math::damp(speed, want, 6.f, dt);

            Vector3 v = dir;
            v.multiplyScalar(speed);
            q.vel = v;
            q.pos = p.pos;
            q.pos.addScaledVector(v, dt);
            clampToFloor(b, q);

            b.feetPlanted = false;
            b.legExtend = 1.f;
            // Pitch bias +0.85: nose up, wings cupped, tail fanned into the
            // airflow. This is the pose everyone recognises and nobody animates.
            updateBankPitch(i, b, dt, 0.85f);
        }

        void integrateLaunch(int i, Bird& b, const Dyn& p, Dyn& q, float dt) {

            const float t = b.stateTime;

            if (t < 0.15f) {
                // CROUCH. Ease-in t², legs flexing to 0.55 of full extension,
                // body dropping a body-depth. A take-off that starts with the
                // wings is a helicopter; a take-off that starts with the legs is
                // a bird.
                const float e = (t / 0.15f) * (t / 0.15f);
                b.legExtend = math::lerp(1.f, 0.55f, e);
                b.bodyLift = -0.9f * params_.shape.bodyRadius * e;
                b.feetPlanted = true;
                q.pos = b.anchor;
                q.pos.addScaledVector(b.spotNormal, standHeight(b) * math::lerp(1.f, 0.62f, e));
                q.vel.set(0, 0, 0);
                plantStance(b);
                return;
            }

            if (!b.leaped) {
                b.leaped = true;
                Vector3 v = b.spotNormal;
                v.multiplyScalar(2.6f);
                v.addScaledVector(b.fwd, 1.5f);
                q.vel = v;
                // THE ONLY LEGITIMATE cyclePos RESET IN THE ENTIRE SYSTEM. Top
                // of stroke, so the first thing that happens is a downstroke.
                // Launches are staggered per bird, so it never re-synchronises
                // the flock — and resetting phase anywhere else is forbidden.
                b.cyclePos = 0.f;
                b.feetPlanted = false;
            }

            Vector3 v = q.vel;
            v.addScaledVector(accel_[static_cast<std::size_t>(i)], dt);
            v.y -= kGravity * dt * 0.35f;// the climb-out is not free

            const float speed = std::clamp(v.length(), 0.f, params_.maxSpeed);
            Vector3 dir = fauna::detail::safeNormalized(v, b.fwd);
            // Climb out at ~35°, which is what a startled bird does with the
            // whole of its energy budget.
            dir.y = std::max(dir.y, 0.45f);
            dir = fauna::detail::safeNormalized(dir, b.fwd);
            slewHeading(b, dir, params_.maxTurnRate * 1.5f * dt);

            v = b.fwd;
            v.multiplyScalar(speed);
            q.vel = v;
            q.pos = p.pos;
            q.pos.addScaledVector(v, dt);

            b.legExtend = std::max(0.f, 1.f - (t - 0.15f) / 0.35f);
            b.feetPlanted = false;
            updateBankPitch(i, b, dt, 0.3f);
        }

        void integratePerched(int i, Bird& b, Dyn& q, float dt) {

            // Velocity is NOT zeroed on contact. The residual bleeds into a
            // short forward stagger, which is the difference between a bird
            // landing and a bird being placed.
            b.stagger.addScaledVector(q.vel, dt);
            b.stagger.multiplyScalar(std::exp(-dt / 0.09f));
            b.stagger.clampLength(0.f, 0.09f);
            q.vel.multiplyScalar(std::exp(-dt / 0.09f));

            if (b.walker) groundGait(b, dt);
            else plantStance(b);

            q.pos = b.anchor;
            q.pos.addScaledVector(b.spotNormal, standHeight(b));
            q.pos.add(b.stagger);

            b.feetPlanted = true;
            b.legExtend = 1.f;

            // Perched birds face into the wind, with ±35° of scatter. Free
            // realism, and it explains the residual alignment: a row of birds
            // roughly facing one way reads as birds in wind, whereas exactly one
            // way reads as copy-paste and pure random reads as noise.
            Vector3 into{-params_.wind.x, 0.f, -params_.wind.y};
            if (into.lengthSq() < 1e-8f) into.set(0, 0, 1);
            into.normalize();
            const float scatter = (rnd01(params_.seed, static_cast<std::uint32_t>(i), 0x5EEDu) * 2.f - 1.f) * 0.61f;
            into.applyAxisAngle(Vector3{0, 1, 0}, scatter);
            slewHeading(b, into, 2.0f * dt);

            b.bank = math::damp(b.bank, 0.f, 6.f, dt);
            // Tracked hard rather than damped gently: the hop's nose-down /
            // nose-up swing is a kinematic fact of the cycle, and a slow damp
            // would smear it into a gentle nod. 40 ms against a 380 ms cycle.
            b.pitch = math::damp(b.pitch, 0.06f + b.gaitPitch, 25.f, dt);
        }

        // FEET HAVE WORLD PLANT POSITIONS AND DO NOT SLIDE. A step arcs the
        // swing foot to its new plant over kStepTime on a parabola; between
        // steps the foot is locked, and the arc ENDS EXACTLY ON THE PLANT, which
        // is what lets BirdPose::feetPlanted stay true the whole time. Writing
        // the swing foot here rather than at pose time is deliberate: a foot
        // interpolated in two places snaps back the frame the two disagree.
        void groundGait(Bird& b, float dt) {

            Vector3 to = b.groundGoal;
            to.sub(b.anchor);
            to.y = 0.f;
            const float dist = to.length();

            const bool moving = dist > 0.05f && time_ >= b.pauseUntil;
            if (!moving) {
                // Between micro-goals: head up, faster saccades. That is
                // vigilance behaviour, and it is what ground-feeding birds
                // actually spend most of their time doing.
                b.headLead = math::damp(b.headLead, 0.f, 12.f, dt);
                b.stepStart = -1.f;
                b.gaitLift = 0.f;
                b.gaitPitch = 0.f;
                plantStance(b);
                return;
            }

            to.multiplyScalar(1.f / dist);

            if (params_.gait == Gait::Hop) {
                hopCycle(b, to, dist);
                return;
            }
            b.gaitPitch = 0.f;
            b.gaitLift = 0.f;

            const float period = 1.f / kStepRate;
            const float advance = std::min(kGroundSpeed * dt, dist);
            b.anchor.addScaledVector(to, advance);

            if (b.stepStart < 0.f || time_ - b.stepStart >= period) {
                // Commit the finished step BEFORE choosing the next one, or the
                // foot that just landed is still remembered at its old plant.
                if (b.stepStart >= 0.f) b.footWorld[static_cast<std::size_t>(b.swingFoot)] = b.stepTo;
                b.stepStart = time_ - std::fmod(std::max(0.f, time_ - b.stepStart - period), period);
                b.swingFoot = 1 - b.swingFoot;
                b.stepFrom = b.footWorld[static_cast<std::size_t>(b.swingFoot)];
                b.stepTo = stanceFoot(b, b.swingFoot);
                b.stepTo.addScaledVector(to, kGroundSpeed * period * 0.6f);
            }

            const float tStep = std::max(0.f, time_ - b.stepStart);
            const float e = std::clamp(tStep / std::min(kStepTime, period), 0.f, 1.f);
            Vector3 f = b.stepFrom;
            f.lerp(b.stepTo, e);
            // Parabola of height 0.25 × step length. A foot that slides flat
            // between plants reads as a puppet on a rail.
            f.addScaledVector(b.spotNormal, b.stepFrom.distanceTo(b.stepTo) * e * (1.f - e));
            b.footWorld[static_cast<std::size_t>(b.swingFoot)] = f;
            b.footWorld[static_cast<std::size_t>(1 - b.swingFoot)] = stanceFoot(b, 1 - b.swingFoot);

            // THE HEAD BOB IS DERIVED FROM GAZE STABILISATION, NOT GUESSED AS A
            // SINUSOID. For the first 65 % of the step the head holds a fixed
            // WORLD position, so its body-space offset is exactly the distance
            // the body has travelled; over the remaining 35 % it eases back. At
            // 0.38 m/s and 3 Hz that is an 0.082 m excursion, which is the
            // measured pigeon figure — and the correct derivation is also the
            // cheaper code.
            const float hold = 0.65f * period;
            if (tStep < hold) {
                b.headLead = -kGroundSpeed * tStep;
            } else {
                const float ease = math::smoothstep(hold, period, tStep);
                b.headLead = math::lerp(-kGroundSpeed * hold, 0.f, ease);
            }
        }

        // BOTH FEET TOGETHER, and a real ballistic phase. Finches and sparrows
        // hop; starlings, crows and pigeons walk. Getting the pair the wrong way
        // round is the kind of error nobody can name and everybody notices.
        //
        // The feet are carried through the ballistic arc WITH the body rather
        // than left on the ground: at 0.25 × step height the leg would otherwise
        // stretch to twice its length for a fifth of a second, which reads as
        // the bird being lifted by a wire instead of pushing off.
        void hopCycle(Bird& b, const Vector3& dir, float dist) {

            static constexpr float kCrouchEnd = 0.10f / kHopCycle;// 0.10 s crouch
            static constexpr float kFlightEnd = 0.28f / kHopCycle;// 0.18 s ballistic

            const float hopLen = std::min(0.22f, dist);

            if (b.stepStart < 0.f) {
                b.stepStart = time_ - b.gaitPhase * kHopCycle;
                b.stepFrom = b.anchor;
            }

            float tau = (time_ - b.stepStart) / kHopCycle;
            if (tau >= 1.f) {
                b.stepFrom.addScaledVector(dir, hopLen);
                b.stepStart = time_;
                b.liftX = -0.10f * tmpl_.legLength;// the landing compression
                b.liftV = 0.f;
                tau = 0.f;
            }

            b.anchor = b.stepFrom;
            float arc = 0.f;

            if (tau < kCrouchEnd) {
                const float e = (tau / kCrouchEnd) * (tau / kCrouchEnd);
                b.gaitLift = -0.55f * params_.shape.bodyRadius * b.size * e;
            } else if (tau < kFlightEnd) {
                const float e = (tau - kCrouchEnd) / (kFlightEnd - kCrouchEnd);
                b.anchor.addScaledVector(dir, hopLen * e);
                arc = 0.25f * hopLen * 4.f * e * (1.f - e);
                b.gaitLift = arc;
            } else {
                b.anchor.addScaledVector(dir, hopLen);
                b.gaitLift = 0.f;
            }

            // Nose-down at push-off, up at landing — one cosine, and it is what
            // makes a hop read as effort rather than as a translation.
            b.gaitPitch = 0.26f * std::cos(math::TWO_PI * tau);
            b.headLead = 0.f;

            for (int g = 0; g < 2; ++g) {
                Vector3 f = stanceFoot(b, g);
                f.addScaledVector(b.spotNormal, arc);
                b.footWorld[static_cast<std::size_t>(g)] = f;
            }
        }

        // Feet directly under the hips, in the body's OWN frame, so a perched
        // bird's stance turns with it rather than staying pinned to world axes —
        // which is what makes a bird that shuffles round to face the wind look
        // like it turned rather than like it rotated.
        [[nodiscard]] Vector3 stanceFoot(const Bird& b, int g) const {

            Vector3 bx, by, bz;
            bodyBasis(b, bx, by, bz);
            Vector3 f = b.anchor;
            f.addScaledVector(bx, (g == 0 ? 1.f : -1.f) * 0.34f * params_.shape.bodyRadius * b.size);
            return f;
        }

        void plantStance(Bird& b) {

            for (int g = 0; g < 2; ++g) {
                b.footWorld[static_cast<std::size_t>(g)] = stanceFoot(b, g);
            }
        }

        void slewHeading(Bird& b, const Vector3& want, float maxAngle) {

            Vector3 dir = fauna::detail::safeNormalized(want, b.fwd);
            const float c = std::clamp(b.fwd.dot(dir), -1.f, 1.f);
            const float ang = std::acos(c);
            if (!(ang > 1e-5f)) {
                b.fwd = dir;
                return;
            }
            if (ang <= maxAngle) {
                b.fwd = dir;
                return;
            }

            Vector3 axis;
            axis.crossVectors(b.fwd, dir);
            // Exactly antiparallel: the cross product is degenerate and any
            // perpendicular axis is as good as another. Picking one keeps a bird
            // that has been reversed by a hard evade from freezing its heading.
            axis = fauna::detail::safeNormalized(axis, {0, 1, 0});
            b.fwd.applyAxisAngle(axis, maxAngle);
            b.fwd = fauna::detail::safeNormalized(b.fwd, dir);
        }

        void updateBankPitch(int i, Bird& b, float dt, float pitchBias = 0.f) {

            // BANK COMES FROM THE COMMANDED LATERAL ACCELERATION, never from the
            // measured yaw rate. The measured rate lags by a frame and is noisy
            // at low speed, which produces a wobble the eye reads as a damaged
            // bird; the command is exactly what the bird "intends" and arrives
            // one frame early, which is what a real animal's roll does.
            const Vector3& a = accel_[static_cast<std::size_t>(i)];

            Vector3 bx, by, bz;
            bodyBasisFlat(b, bx, by, bz);

            const float side = a.dot(bx);
            const float target = std::clamp(-params_.bankGain * side / kGravity,
                                            -params_.maxBank, params_.maxBank);
            b.bank = math::damp(b.bank, target + b.rollTrim, 1.f / params_.bankTau, dt);

            const Dyn& d = next_[static_cast<std::size_t>(i)];
            const float speed = d.vel.length();
            const float climb = speed > 1e-3f ? std::asin(std::clamp(d.vel.y / speed, -1.f, 1.f)) : 0.f;
            b.pitch = math::damp(b.pitch, climb * 0.6f + pitchBias, 1.f / params_.pitchTau, dt);
        }

        // The unbanked, unpitched frame: bz along the heading, bx = up × bz.
        // Matrix4::lookAt is FORBIDDEN here — its z column points away from the
        // target and its x column is up × z, which silently mirrors the bird and
        // produces a flock whose wings all beat the wrong way round.
        void bodyBasisFlat(const Bird& b, Vector3& bx, Vector3& by, Vector3& bz) const {

            bz = b.fwd;
            bx.crossVectors(Vector3{0, 1, 0}, bz);
            bx = fauna::detail::safeNormalized(bx, {1, 0, 0});
            by.crossVectors(bz, bx);
        }

        void bodyBasis(const Bird& b, Vector3& bx, Vector3& by, Vector3& bz) const {

            bodyBasisFlat(b, bx, by, bz);

            // Pitch about bx. Rotating bz about +bx by +θ tips the nose DOWN,
            // so the sign is inverted here to keep "positive pitch = nose up"
            // meaning what everyone assumes it means.
            if (b.pitch != 0.f) {
                by.applyAxisAngle(bx, -b.pitch);
                bz.applyAxisAngle(bx, -b.pitch);
            }
            if (b.bank != 0.f) {
                bx.applyAxisAngle(bz, b.bank);
                by.applyAxisAngle(bz, b.bank);
            }
        }

        // ── Wingbeat, head, springs, state exits ─────────────────────────
        void advanceBeat(int i, Bird& b, const Dyn& q, float dt) {

            const float speedNorm = std::clamp(q.vel.length() / std::max(params_.cruiseSpeed, 1e-3f), 0.f, 2.f);

            // Named beatScale, not scale: Object3D::scale is in this class's
            // scope and a bare `scale` here shadows it (/W4 C4458). Harmless
            // today, and a genuinely nasty bug the day someone edits this
            // function expecting to touch the node's transform.
            float beatScale = 1.f;
            switch (b.state) {
                case BirdState::Cruise: beatScale = 0.75f + 0.35f * speedNorm; break;
                case BirdState::Approach: beatScale = 0.90f; break;
                case BirdState::Flare: beatScale = 0.85f; break;
                case BirdState::Launch: beatScale = 1.35f; break;
                case BirdState::Evade: beatScale = 1.20f; break;
                case BirdState::Perched: beatScale = 0.75f; break;
            }

            float f = baseBeatHz() * b.beatRate * beatScale;

            // THE NYQUIST GUARD IS NOT OPTIONAL. At 30 fps an unguarded 8.5 Hz
            // beat is 3.5 samples per cycle, so the stroke extremes land on
            // random phases and the amplitude visibly flickers. Guarded, a 30 fps
            // host gets a smooth 5 Hz beat: a slightly slow bird reads as a
            // bigger bird, a strobing bird reads as broken software.
            if (params_.nyquistGuard && dtSmoothed_ > 1e-5f) {
                f = std::min(f, 1.f / (6.f * dtSmoothed_));
            }

            // PHASE IS ADVANCED, NEVER EVALUATED AS sin(2π·f(t)·t). The moment f
            // changes, the closed form teleports the wing; the accumulator
            // cannot. This one line is why "flap harder", "ease into a glide"
            // and "resume from a glide" are all free of pops.
            b.cyclePos = fauna::detail::frac01(b.cyclePos + f * dt);

            // Flap-bounding. Suppressed while climbing, banked, near an obstacle
            // or in any state but Cruise — real birds flap continuously when
            // they need control authority, and that suppression is what stops
            // the undulation itself from becoming metronomic.
            const bool suppress = b.state != BirdState::Cruise ||
                                  q.vel.y > 1.f ||
                                  std::abs(b.bank) > 0.44f ||
                                  (hasField_ && perch_.clearanceAt(q.pos) < params_.obstacleMargin);

            if (suppress) {
                b.bounding = false;
                b.boundT = 0.f;
            } else if (b.bounding) {
                b.boundT += dt;
                if (b.boundT > 0.30f) {
                    b.bounding = false;
                    // Bursts of 5 ± 1 beats. A fixed burst length turns the
                    // undulation itself into a metronome, which is the very
                    // thing flap-bounding was added to break.
                    b.beatsLeft = 4.f + 2.f * rnd01(params_.seed, static_cast<std::uint32_t>(i), b.seq++);
                }
            } else {
                b.beatsLeft -= f * dt;
                if (b.beatsLeft <= 0.f) {
                    b.bounding = true;
                    b.boundT = 0.f;
                }
            }

            float target = 1.f;
            switch (b.state) {
                case BirdState::Cruise: target = b.bounding ? 0.f : 1.f; break;
                case BirdState::Approach: {
                    // (1 − settle)²: amplitude falls to zero BEFORE the feet
                    // touch. At 0.6 m it is already 25 %, and a bird that aborts
                    // picks the envelope back up from wherever it fell to, so
                    // the transition is continuous in both directions.
                    const float dist = q.pos.distanceTo(standPoint(b));
                    const float settle = math::smoothstep(0.f, 1.f, 1.f - dist / std::max(params_.flareDistance * 4.f, 1e-3f));
                    target = (1.f - settle) * (1.f - settle);
                    break;
                }
                case BirdState::Flare: target = 0.f; break;
                case BirdState::Perched:
                    // THE BALANCE HOLD: wings pinned half-raised for 0.12 s after
                    // contact before folding. Every real bird does it and nobody
                    // animates it; without it the fold is a single-frame pop the
                    // eye catches at 40 m.
                    target = (b.stateTime < kBalanceHold) ? 0.30f : 0.f;
                    break;
                case BirdState::Launch: target = (b.stateTime < 0.15f) ? 0.f
                                                                      : 1.f + 0.25f * std::exp(-(b.stateTime - 0.15f) / 0.30f);
                    break;
                case BirdState::Evade: target = 1.f; break;
            }
            b.flapWeight = math::damp(b.flapWeight, target, 12.f, dt);

            // perchFold: 0 → 1 over the 0.23 s that follow the balance hold, and
            // back over 0.20 s at the leap. Nothing in the pose function
            // switches; every one of these blends is C¹ by construction.
            float foldTarget = 0.f;
            if (b.state == BirdState::Perched) {
                foldTarget = (b.stateTime < kBalanceHold) ? 0.f : 1.f;
                if (b.shuffleStart >= 0.f && time_ - b.shuffleStart < 0.35f) foldTarget = 0.75f;
            } else if (b.state == BirdState::Launch && b.stateTime < 0.15f) {
                foldTarget = 1.f;
            }
            // settleTime is contact → wings fully folded, and the balance hold
            // eats the first 0.12 s of it, so the fold itself gets what remains.
            const float foldSpan = std::max(params_.settleTime - kBalanceHold, 0.05f);
            const float foldRate = (b.state == BirdState::Launch) ? 1.f / 0.20f : 1.f / foldSpan;
            b.perchFold = math::damp(b.perchFold, foldTarget, 3.f * foldRate, dt);

            float spread = 0.10f;
            switch (b.state) {
                case BirdState::Approach: spread = 0.55f; break;
                case BirdState::Flare: spread = 1.00f; break;
                case BirdState::Perched: spread = 0.05f; break;
                case BirdState::Launch: spread = 0.60f; break;
                case BirdState::Evade: spread = 0.75f; break;
                default: break;
            }
            b.tailSpread = math::damp(b.tailSpread, spread, 8.f, dt);
        }

        void advanceHead(int i, Bird& b, float dt) {

            // BIRDS HAVE ALMOST NO EYE MOVEMENT IN THE SOCKET, so gaze is
            // stabilised by the neck: the head SNAPS and HOLDS. It never sweeps.
            // Lerping toward a moving target is forbidden — a 0.25 s eased head
            // turn reads as a lizard, and this is the motion that gets inspected
            // when someone finally looks straight at the flock.
            if (b.saccadeStart < 0.f && time_ >= b.saccadeNext) {

                b.headYawFrom = b.headYaw;
                b.headPitchFrom = b.headPitch;

                const float yawU = rnd01(params_.seed, static_cast<std::uint32_t>(i), b.seq++);
                const float pitchU = rnd01(params_.seed, static_cast<std::uint32_t>(i), b.seq++);
                const float holdU = rnd01(params_.seed, static_cast<std::uint32_t>(i), b.seq++);

                const bool preening = (b.preenUntil > time_);
                b.headYawTo = preening ? (yawU < 0.5f ? -1.15f : 1.15f) : (yawU * 2.f - 1.f) * 1.22f;
                b.headPitchTo = preening ? -0.6f : (pitchU * 2.f - 1.f) * 0.44f;

                const float hold = preening ? 1.f + 2.f * holdU
                                            : (b.state == BirdState::Perched && b.walker && time_ < b.pauseUntil
                                                       ? 0.35f + 0.6f * holdU
                                                       : 0.35f + 2.15f * holdU);
                b.saccadeStart = time_;
                b.saccadeNext = time_ + kSaccade + hold;
            }

            if (b.saccadeStart >= 0.f) {
                const float e = math::smoothstep(0.f, kSaccade, time_ - b.saccadeStart);
                b.headYaw = math::lerp(b.headYawFrom, b.headYawTo, e);
                b.headPitch = math::lerp(b.headPitchFrom, b.headPitchTo, e);
                if (e >= 1.f) b.saccadeStart = -1.f;
            }

            if (b.state != BirdState::Perched) {
                // In flight the head stays level while the body banks 50°, which
                // is the single most legible cue that the thing is alive.
                b.headYaw = math::damp(b.headYaw, 0.f, 6.f, dt);
                b.headPitch = math::damp(b.headPitch, 0.f, 6.f, dt);
            }
            b.headRoll = std::clamp(-0.7f * b.bank, -0.6f, 0.6f);
        }

        void advanceSprings(int i, Bird& b, float dt) {

            const auto slot = static_cast<std::uint32_t>(i);

            // Second-order landing spring, ω = 22 rad/s, ζ = 0.55 — one visible
            // rebound, settled in ~0.35 s. THE INITIAL DISPLACEMENT IS NEGATIVE:
            // a landing compresses first. A +cos impulse pops the body UP on the
            // exact frame the feet meet the branch and reads as the bird being
            // struck from below.
            const float acc = -kLiftOmega * kLiftOmega * b.liftX - 2.f * kLiftZeta * kLiftOmega * b.liftV;
            b.liftV += acc * dt;
            b.liftX += b.liftV * dt;

            if (b.state == BirdState::Perched) {

                // The gait's own vertical offset, the landing spring and the
                // breathing bob are three INDEPENDENT contributions summed here
                // rather than three functions each assigning bodyLift — which is
                // how a hop arc quietly stops existing the day someone adds a
                // fourth.
                b.bodyLift = b.liftX + b.gaitLift;
                // Continuous 0.35 Hz breathing, 4 mm. Below the threshold of
                // conscious notice and above the threshold of "that model is
                // frozen".
                b.bodyLift += 0.004f * std::sin(math::TWO_PI * (0.35f * time_ + b.idlePhase[0]));

                if (b.flickStart >= 0.f) {
                    const float e = time_ - b.flickStart;
                    b.tailPitch = (e < 0.18f) ? -0.25f * (1.f - math::smoothstep(0.f, 0.18f, e)) : 0.f;
                    if (e >= 0.18f) b.flickStart = -1.f;
                } else if (time_ >= b.flickAt) {
                    b.flickStart = time_;
                    b.flickAt = time_ + 2.f + 4.f * rnd01(params_.seed, slot, b.seq++);
                }

                if (b.shuffleStart >= 0.f && time_ - b.shuffleStart > 0.35f) b.shuffleStart = -1.f;
                if (b.shuffleStart < 0.f && time_ >= b.shuffleAt) {
                    b.shuffleStart = time_;
                    b.shuffleAt = time_ + 8.f + 12.f * rnd01(params_.seed, slot, b.seq++);
                }
                if (time_ >= b.preenAt) {
                    // Hoisted, never two draws in one expression: argument
                    // evaluation order is unspecified and this is exactly how a
                    // fixed seed comes to mean two different things.
                    const float hold = rnd01(params_.seed, slot, b.seq++);
                    const float gap = rnd01(params_.seed, slot, b.seq++);
                    b.preenUntil = time_ + 1.f + 2.f * hold;
                    b.preenAt = time_ + 15.f + 45.f * gap;
                }

            } else if (b.state == BirdState::Launch && b.stateTime < 0.15f) {
                // bodyLift already written by the crouch.
            } else {
                b.bodyLift = math::damp(b.bodyLift, 0.f, 10.f, dt);
                b.tailPitch = math::damp(b.tailPitch, b.state == BirdState::Flare ? -0.35f : 0.f, 8.f, dt);
            }

            b.tailRoll = math::damp(b.tailRoll, -0.35f * b.bank, 6.f, dt);
        }

        void advanceState(int i, Bird& b, Dyn& q, float dt) {

            switch (b.state) {

                case BirdState::Cruise:
                    b.perchUrge += dt / std::max(b.perchInterval, 0.5f);
                    break;

                case BirdState::Approach: {
                    if (!b.gatePassed) {
                        b.gate = approachGate(b, q);
                        // Latched at 1.5 m, or as soon as the gate is behind the
                        // wing line — a bird that overshoots the waypoint must
                        // not turn round for it.
                        Vector3 toGate = b.gate;
                        toGate.sub(q.pos);
                        if (toGate.lengthSq() < 2.25f || toGate.dot(b.fwd) < 0.f) b.gatePassed = true;
                    }
                    Vector3 to = standPoint(b);
                    to.sub(q.pos);
                    const float dist = to.length();
                    if (dist < params_.flareDistance && dist > 1e-4f &&
                        (to.x * b.fwd.x + to.y * b.fwd.y + to.z * b.fwd.z) / dist > 0.6f) {
                        b.state = BirdState::Flare;
                        b.stateTime = 0.f;
                    }
                    break;
                }

                case BirdState::Flare: {
                    const Vector3 target = standPoint(b);
                    const float footHeight = q.pos.y - standHeight(b) - b.spotPos.y;
                    const bool arrived = q.pos.distanceToSquared(target) < 0.0036f;
                    const bool bounced = q.vel.y > 0.f && b.stateTime > 0.15f;
                    if (footHeight < 0.02f || arrived || bounced || b.stateTime > 2.5f) land(i, b, q);
                    break;
                }

                case BirdState::Perched:
                    b.restUrge += dt / std::max(b.restInterval, 0.5f);
                    break;

                case BirdState::Launch:
                    if (b.stateTime >= 0.85f || q.vel.length() > 0.9f * params_.cruiseSpeed) {
                        b.state = BirdState::Cruise;
                        b.stateTime = 0.f;
                        b.leaped = false;
                        b.beatsLeft = 5.f;
                    }
                    break;

                case BirdState::Evade:
                    if (time_ >= b.evadeUntil) {
                        b.state = BirdState::Cruise;
                        b.stateTime = 0.f;
                    }
                    break;
            }
        }

        void land(int i, Bird& b, Dyn& q) {

            b.state = BirdState::Perched;
            b.stateTime = 0.f;
            b.restUrge = 0.f;
            b.perchUrge = 0.f;
            b.leaped = false;

            b.anchor = b.spotPos;
            b.groundGoal = b.spotPos;
            b.stagger = q.vel;
            b.stagger.multiplyScalar(0.04f);
            b.stagger.clampLength(0.f, 0.09f);

            // The compression spring, kicked downward. See advanceSprings().
            b.liftX = -0.18f * tmpl_.legLength;
            b.liftV = 0.f;

            b.stepStart = -1.f;
            b.pauseUntil = time_ + 0.4f;
            b.saccadeNext = time_ + 0.15f;

            q.pos = b.anchor;
            q.pos.addScaledVector(b.spotNormal, standHeight(b));
            plantStance(b);

            (void) i;
        }

        [[nodiscard]] float approachLegExtend(const Bird& b, const Dyn& q) const {

            if (b.state != BirdState::Approach) return 0.f;
            const float dist = q.pos.distanceTo(standPoint(b));
            // Legs down over the last two metres. Any earlier and the bird looks
            // like it is dangling; any later and the gear appears in one frame.
            return std::clamp(1.f - (dist - params_.flareDistance) / 2.f, 0.f, 1.f);
        }

        [[nodiscard]] bool isFar(int i) const {

            if (!observer_) return false;
            Vector3 eye;
            eye.setFromMatrixPosition(*observer_->matrixWorld);
            return prev_[static_cast<std::size_t>(i)].pos.distanceToSquared(eye) >
                   params_.lodFarDistance * params_.lodFarDistance;
        }

        // ── The vertex bake (§5.5 steps 9-11) ────────────────────────────
        void bakeVertices() {

            if (!posAttr_ || !nrmAttr_) return;

            auto& pos = posAttr_->array();
            auto& nrm = nrmAttr_->array();

            if (birds_.empty()) {
                geometry_->boundingSphere = Sphere(Vector3{}, 0.f);
                return;
            }

            Vector3 lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            Vector3 hi{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};

            Vector3 eye;
            const bool lod = observer_ != nullptr;
            if (lod) eye.setFromMatrixPosition(*observer_->matrixWorld);
            const float lodFar2 = params_.lodFarDistance * params_.lodFarDistance;

            fauna::BirdPose pose;

            for (int i = 0; i < count_; ++i) {

                Bird& b = birds_[static_cast<std::size_t>(i)];
                const Dyn& d = prev_[static_cast<std::size_t>(i)];

                // THE AABB COMES FROM SIMULATION POSITIONS, NEVER FROM THE
                // WRITTEN VERTICES, so the LOD's skipped bakes cannot make the
                // bounding sphere stale — which would frustum-cull exactly the
                // distant birds the LOD exists to serve.
                lo.x = std::min(lo.x, d.pos.x);
                lo.y = std::min(lo.y, d.pos.y);
                lo.z = std::min(lo.z, d.pos.z);
                hi.x = std::max(hi.x, d.pos.x);
                hi.y = std::max(hi.y, d.pos.y);
                hi.z = std::max(hi.z, d.pos.z);

                // THE ONE AND ONLY LOD. A far bird's position still integrates
                // every frame; only the vertex write is skipped. NEVER drop the
                // wingbeat with distance — frequency and amplitude are the
                // entire read at range, and a distant bird whose beat is
                // decimated stops being a bird and becomes a moving dot.
                if (lod && b.baked && d.pos.distanceToSquared(eye) > lodFar2 &&
                    ((frame_ + static_cast<std::uint64_t>(i)) % 2u) != 0u) {
                    continue;
                }

                fillPose(i, b, d, pose);
                toLocal(pose);
                fauna::poseBird(tmpl_, pose, kin_, pos, nrm, i * fauna::kVertsPerBird);
                b.baked = true;
            }

            Vector3 centre{(lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f};
            Vector3 half{(hi.x - lo.x) * 0.5f, (hi.y - lo.y) * 0.5f, (hi.z - lo.z) * 0.5f};

            // One bird radius of slack, scaled by the largest bird: the sphere
            // is built from body ORIGINS, and a wing tip reaches half a span
            // past one.
            const float pad = 0.6f * std::max(params_.shape.wingSpan, params_.shape.bodyLength) *
                              (1.f + params_.sizeVariation);
            float radius = half.length() + pad;

            if (!identityWorld_) {
                centre.applyMatrix4(invWorld_);
                radius *= invScale_;
            }
            if (!std::isfinite(radius) || radius < 0.f) radius = 0.f;
            if (!fauna::detail::isFinite(centre)) centre.set(0, 0, 0);

            // ASSIGNED EVERY FRAME. boundingSphere is a std::optional that
            // NOTHING recomputes once populated (BufferGeometry.hpp:38;
            // Frustum.cpp:68-72), so a stale one pops the whole flock out of
            // view the first time it travels. Because it is assigned,
            // frustumCulled stays TRUE — which is strictly better than the usual
            // workaround of switching culling off.
            geometry_->boundingSphere = Sphere(centre, radius);

            posAttr_->needsUpdate();
            nrmAttr_->needsUpdate();
        }

        void fillPose(int i, const Bird& b, const Dyn& d, fauna::BirdPose& pose) const {

            bodyBasis(b, pose.bx, pose.by, pose.bz);
            pose.pos = d.pos;
            pose.scale = b.size;

            pose.cyclePos = b.cyclePos;
            pose.flapWeight = std::clamp(b.flapWeight, 0.f, 1.f);
            pose.beatAmp = b.beatAmp;
            pose.wingAsym = b.wingAsym;
            pose.perchFold = std::clamp(b.perchFold, 0.f, 1.f);

            pose.headYaw = b.headYaw;
            pose.headPitch = b.headPitch;
            pose.headRoll = b.headRoll;
            pose.headLead = b.headLead;

            pose.tailSpread = std::clamp(b.tailSpread, 0.f, 1.f);
            pose.tailPitch = b.tailPitch;
            pose.tailRoll = b.tailRoll;

            pose.legExtend = std::clamp(b.legExtend, 0.f, 1.f);
            // THE ONE DISCONTINUOUS INPUT IN THE WHOLE POSE CONTRACT. Flipping
            // it teleports the foot from the hanging position to footWorld, so
            // it is only ever flipped on the frame those two coincide — at
            // touchdown, where legExtend is already 1 and the foot is already on
            // the surface. Everything else the caller drives is C¹.
            pose.feetPlanted = b.feetPlanted;
            // The swing foot is written once, in groundGait(), and simply copied
            // here. Interpolating it in two places is how a foot comes to snap
            // back a frame after it lands.
            pose.footWorld = b.footWorld;

            pose.bodyLift = b.bodyLift;
            (void) i;
        }

        // ── State ────────────────────────────────────────────────────────
        Params params_;
        fauna::BirdTemplate tmpl_;
        fauna::BirdKinematics kin_{};

        int count_ = 0;
        std::vector<Bird> birds_;
        std::vector<Dyn> prev_, next_;
        std::vector<Vector3> accel_;
        std::vector<int> nbrIdx_;
        std::vector<int> nbrCount_;
        std::vector<float> rngDraws_;

        fauna::PerchIndex perch_;
        std::vector<int> claimedBy_;
        std::vector<int> spotScratch_;
        std::function<bool(const Mesh&)> filter_;
        bool bakeRequested_ = false;
        bool hasField_ = false;

        FloatBufferAttribute* posAttr_ = nullptr;
        FloatBufferAttribute* nrmAttr_ = nullptr;

        Matrix4 invWorld_;
        float invScale_ = 1.f;
        bool identityWorld_ = true;

        const Object3D* disturbance_ = nullptr;
        const Camera* observer_ = nullptr;

        Vector3 centroid_{}, meanVel_{}, homeDrifted_{};
        std::array<Vector3, 3> driftAxis_{};
        std::array<float, 3> driftPhase_{};
        int leaderCount_ = 0, lonerCount_ = 0;
        int perchedNow_ = 0, committed_ = 0;
        float nextLeaderVote_ = 0.f;
        float launchedRecently_ = 0.f;

        float time_ = 0.f;
        float lastDt_ = 0.f;
        float dtSmoothed_ = 1.f / 60.f;
        std::uint64_t frame_ = 0;
        std::uint64_t updates_ = 0;
        std::uint64_t stalled_ = 0;

        Vector3 zero_{};
    };

}// namespace threepp

#endif// THREEPP_FLOCK_HPP
