// One-time scene bake for the ambient flock: where a bird may land, where it
// must not fly, and how high the ground is.
//
// THE OUTPUT IS A SNAPSHOT, AND THAT IS THE WHOLE POINT.
//
// The obvious design keeps a BVH per host geometry alive and probes it a few
// rays per frame. BVH caches a raw `const BufferGeometry*` with no dirty
// tracking (BVH.hpp:96), so the first time the host regenerates a terrain tile
// or frees a tree, the next probe dereferences freed memory — an intermittent
// crash in the render loop, in a subsystem nobody will suspect. Every BVH built
// here is destroyed when the bake completes; the products below are plain
// world-space values holding no pointer and no transform. The worst a scene
// edit can do afterwards is leave a bird perched in mid-air, which the host
// fixes by calling the bake again.
//
// The bake is amortised over frames by a WORK-UNIT COUNT, never a millisecond
// budget: a time budget would make the number of frames the bake takes — and
// therefore every bird's trajectory — depend on machine speed, silently
// breaking the deterministic-for-a-fixed-seed contract. bakeBlocking() and an
// amortised begin()/step() sequence produce BIT-IDENTICAL perch tables, because
// the budget decides only when step() returns, never what order anything is
// visited in.
//
// THE HOST DOES NOT HAVE TO CALL updateMatrixWorld() FIRST — begin() calls it.
// Both renderers refresh world matrices themselves, but only inside render(),
// which runs AFTER update() in the canonical animate lambda. A host that does
//     mesh->position.set(10, 0, 5); scene->add(mesh); index.bakeBlocking(*scene, …);
// before entering the loop would otherwise bake every perch as if the scene were
// collapsed at the origin, and a headless test hits that 100 % of the time.
// Worse for an amortised bake: rays cast on frame 0 would use identity matrices
// and rays on frame 3 real ones, stitching one perch table out of two different
// coordinate systems.
//
// ZERO PERCHES IS A NORMAL ANSWER, NOT AN ERROR. A sky-only scene, a scene of
// nothing but steep roofs, or a scene whose meshes the filter rejected all bake
// to an empty spot table. Nothing logs, nothing throws, every query returns its
// wide-open answer and the flock simply flies. `spots().empty()` is the answer
// to "my birds never land", and it is meant to be read, not prevented.
//
// Known limitations, stated rather than fixed:
//   · The obstacle field is 2 m cells by default. A bird may clip a bare twig, a
//     wire or a lamp post. The ground floor, which is what actually matters,
//     comes from the heightfield and is much finer.
//   · `walkable` is a pure slope test. The flat top of a 0.08 m rail is
//     therefore "walkable" even though nothing could walk on it; the flock is
//     expected to treat a spot's surroundings, not this flag, as the authority
//     on whether to take a step.
//   · `heightAt` is the HIGHEST sampled surface in its column, not the terrain
//     under an overpass. That is deliberate — it is what the flock uses as a
//     floor, and a floor that ignores roofs would fly birds through them.
//   · An InstancedMesh is a Mesh, so the traversal finds one — and bakes its
//     base geometry once, at the NODE's transform, ignoring every instance
//     matrix. Scattered vegetation authored that way therefore contributes one
//     copy of itself at the origin of the field. Reject it with the `filter`
//     predicate, or bake a proxy; there is no in-tree way to enumerate instance
//     transforms cheaply enough to be worth doing implicitly.
//
// Header-only, dependency-free beyond threepp core (+ threepp/utils/BVH.hpp,
// which is public and compiled into the library but is NOT pulled in by
// threepp.hpp).

#ifndef THREEPP_EXTRAS_FAUNA_PERCHINDEX_HPP
#define THREEPP_EXTRAS_FAUNA_PERCHINDEX_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Ray.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/BVH.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace threepp::fauna {

    namespace detail {

        // ── Deterministic bit tools ──────────────────────────────────────
        //
        // Both of these are pure functions of their arguments. Nothing here
        // decides WHETHER a spot is accepted — mix64 only chooses where a key
        // lands in the thinning table (linear probing then makes membership
        // order-independent), and morton3 only chooses the final sort order.
        // Keeping them pure is what lets the bake stay bit-reproducible without
        // a single ordered container in the hot path.

        inline std::uint64_t mix64(std::uint64_t x) {

            // math::Rng::mixBits IS this function (one splitmix64 step);
            // delegating keeps the bake bit-identical while retiring the
            // private copy.
            return math::Rng::mixBits(x);
        }

        // Spread the low 21 bits of `v` so that bit i lands at bit 3i.
        inline std::uint64_t spread21(std::uint64_t v) {

            v &= 0x1fffffULL;
            v = (v | (v << 32)) & 0x1f00000000ffffULL;
            v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
            v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
            v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
            v = (v | (v << 2)) & 0x1249249249249249ULL;
            return v;
        }

        inline std::uint64_t morton3(std::uint32_t x, std::uint32_t y, std::uint32_t z) {

            return spread21(x) | (spread21(y) << 1) | (spread21(z) << 2);
        }

        inline std::size_t nextPow2(std::size_t v) {

            std::size_t n = 1;
            while (n < v) n <<= 1;
            return n;
        }

        // Quantise a world position onto a separation lattice and pack the three
        // signed cell indices into one 64-bit key. 21 bits per axis at the 0.45 m
        // default reaches ±470 km from the origin, which is several orders of
        // magnitude past any scene this ships in.
        inline std::uint64_t latticeKey(const Vector3& p, float invSpacing) {

            const auto ix = static_cast<std::uint64_t>(
                    static_cast<std::int64_t>(std::floor(p.x * invSpacing)) & 0x1fffffLL);
            const auto iy = static_cast<std::uint64_t>(
                    static_cast<std::int64_t>(std::floor(p.y * invSpacing)) & 0x1fffffLL);
            const auto iz = static_cast<std::uint64_t>(
                    static_cast<std::int64_t>(std::floor(p.z * invSpacing)) & 0x1fffffLL);

            return (ix << 42) | (iy << 21) | iz;
        }

    }// namespace detail

    // A baked landing site. Pointer-free and transform-free by design.
    struct PerchSpot {
        Vector3 position{};     // world
        Vector3 normal{0, 1, 0};// world, unit, already flipped upward
        bool walkable = false;  // near-flat and with room to step
        bool ground = false;    // within groundEpsilon of the local ground height

        bool operator==(const PerchSpot&) const = default;
    };

    class PerchIndex {

    public:
        struct Params {
            // ── Sampling ─────────────────────────────────────────────────
            float probeSpacing = 1.0f;       // m, downward ray grid pitch
            int maxProbesPerAxis = 128;      // hard cap; a huge scene gets a coarser grid
            int maxPerches = 4096;           // hard cap on the spot table
            float perchMinSeparation = 0.45f;// m, thinning distance between accepted spots

            // ── Acceptance ───────────────────────────────────────────────
            float maxSlope = 0.61f;      // rad (35°); keep spots with normal.y > cos(this)
            float headroomLow = 0.30f;   // m above the spot that must be clear
            float headroomHigh = 0.80f;  // m above the spot that must be clear
            float walkableSlope = 0.44f; // rad (25°); flatter than this ⇒ walkable
            float groundEpsilon = 0.60f; // m; spot within this of the column height ⇒ ground

            // ── Obstacle field ───────────────────────────────────────────
            float cellSize = 2.0f;// m, obstacle grid cell
            int maxCellsX = 64;   // hard caps on the 3-D grid
            int maxCellsY = 32;
            int maxCellsZ = 64;
            int chamferPasses = 8;// saturating distance, in cells
            int heightGrid = 128; // XZ resolution of the ground heightfield

            // ── Cost control ─────────────────────────────────────────────
            int bakeWorkPerFrame = 30000;      // work units per step(); 0 ⇒ blocking
            int bvhMaxTriangles = 40000;       // meshes above this get no BVH (see banner)
            int maxSamplesPerTriangleAxis = 16;// barycentric sampling cap for big triangles

            bool operator==(const Params&) const = default;
        };

        PerchIndex() = default;

        // Begin (or restart) a bake over `root`. `exclude` is skipped by pointer
        // identity along with all of its descendants — pass the Flock itself.
        // `filter` may be empty; when set, a Mesh is considered only if it
        // returns true. Calls root.updateMatrixWorld() itself: nothing else
        // refreshes it before the first render, and a bake against identity
        // matrices puts every perch at the world origin.
        //
        // PHASE 0 (COLLECT) RUNS HERE, SYNCHRONOUSLY, AND IS DELIBERATELY NOT
        // AMORTISED. Spreading the traversal over frames would mean holding an
        // Object3D* across frames — the exact dangling-pointer hazard this whole
        // file exists to remove. Everything after the traversal works from
        // values that have already been copied out, so the rest of the bake can
        // be spread as thin as the host likes.
        void begin(Object3D& root, const Params& params,
                   const Object3D* exclude,
                   const std::function<bool(const Mesh&)>& filter) {

            clear();
            params_ = sanitise(params);

            root.updateMatrixWorld();

            // Gate on the Mesh cast FIRST. Frustum::intersectsObject and friends
            // dereference object.geometry() with no null check (Frustum.cpp:68-70)
            // and geometry() returns nullptr for Group and Light nodes; the same
            // trap sits waiting for anything that walks a scene by hand.
            root.traverseType<Mesh>([&](Mesh& mesh) {
                if (excluded(mesh, exclude)) return;
                if (filter && !filter(mesh)) return;

                const auto geometry = mesh.geometry();
                if (!geometry) return;

                const auto* index = geometry->getIndex();
                const auto* position = geometry->getAttribute<float>("position");
                if (!index || !position) return;

                const int triCount = index->count() / 3;
                const int vertCount = position->count();
                if (triCount <= 0 || vertCount <= 0) return;

                MeshEntry entry;
                entry.geometry = geometry;
                entry.worldMatrix.copy(*mesh.matrixWorld);
                entry.triCount = triCount;
                entry.vertCount = vertCount;

                // The world AABB is computed fresh from the attribute rather than
                // read from geometry->boundingBox: that optional is a cache the
                // host may never have filled, and if it did fill it before
                // deforming the mesh it is now a lie. We also do not write it
                // back — a bake has no business mutating the scene it reads.
                Box3 local;
                local.makeEmpty();
                Vector3 v;
                for (int i = 0; i < vertCount; ++i) {
                    v.set(position->getX(static_cast<std::size_t>(i)),
                          position->getY(static_cast<std::size_t>(i)),
                          position->getZ(static_cast<std::size_t>(i)));
                    if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) continue;
                    local.expandByPoint(v);
                }
                if (local.isEmpty()) return;

                entry.worldBox.copy(local).applyMatrix4(entry.worldMatrix);
                if (entry.worldBox.isEmpty()) return;

                bounds_.union_(entry.worldBox);
                sampleTriTotal_ += triCount;
                meshes_.push_back(std::move(entry));
            });

            phase_ = Phase::GridAlloc;
        }

        // Advance the bake by up to Params::bakeWorkPerFrame work units.
        // Returns true when the bake is complete. Safe to call after completion
        // (returns true, does nothing) and before begin() (returns false and
        // does nothing — there is no bake to advance).
        bool step() {

            if (complete_) return true;
            if (phase_ == Phase::Idle) return false;

            const bool blocking = params_.bakeWorkPerFrame <= 0;
            std::int64_t budget = blocking
                                          ? std::numeric_limits<std::int64_t>::max()
                                          : static_cast<std::int64_t>(params_.bakeWorkPerFrame);

            while (budget > 0 && !complete_) {

                std::int64_t spent = 0;
                switch (phase_) {
                    case Phase::GridAlloc: spent = allocateGrids(); break;
                    case Phase::Sample: spent = sampleTriangles(budget); break;
                    case Phase::Chamfer: spent = relaxChamfer(budget); break;
                    case Phase::RayGrid: spent = castProbeGrid(budget); break;
                    case Phase::Finalise: spent = finalise(); break;
                    default: spent = budget; break;
                }
                budget -= (spent > 0 ? spent : 1);
            }

            return complete_;
        }

        // Run the whole bake now. Equivalent to begin() then step() until done.
        void bakeBlocking(Object3D& root, const Params& params,
                          const Object3D* exclude,
                          const std::function<bool(const Mesh&)>& filter) {

            begin(root, params, exclude, filter);
            while (!step()) {}
        }

        // Discard everything. complete() becomes false, all queries return their
        // empty-scene answers.
        void clear() {

            params_ = Params{};
            phase_ = Phase::Idle;
            complete_ = false;

            spots_.clear();
            spots_.shrink_to_fit();
            bounds_.makeEmpty();

            dist_.clear();
            dist_.shrink_to_fit();
            chamferScratch_.clear();
            chamferScratch_.shrink_to_fit();
            nx_ = ny_ = nz_ = 0;
            cell_ = 0.f;
            invCell_ = 0.f;
            saturation_ = 0;
            gridMin_.set(0, 0, 0);

            height_.clear();
            height_.shrink_to_fit();
            heightN_ = 0;
            heightCellX_ = heightCellZ_ = 0.f;

            clearScratch();
            clearSpotGrid();
        }

        [[nodiscard]] bool complete() const {

            return complete_;
        }

        [[nodiscard]] float progress() const {

            if (complete_) return 1.f;

            switch (phase_) {
                case Phase::Idle: return 0.f;
                case Phase::GridAlloc: return kwCollect;
                case Phase::Sample:
                    return kwCollect + kwSample * ratio(sampleTriDone_, sampleTriTotal_);
                case Phase::Chamfer:
                    return kwCollect + kwSample + kwChamfer * ratio(chamferPass_, std::max(1, saturation_));
                case Phase::RayGrid:
                    return kwCollect + kwSample + kwChamfer +
                           kwRayGrid * ratio(static_cast<int>(rayMesh_), static_cast<int>(meshes_.size()));
                default: return 1.f - kwFinalise;
            }
        }

        [[nodiscard]] const std::vector<PerchSpot>& spots() const {

            return spots_;
        }

        [[nodiscard]] const Box3& sceneBounds() const {

            return bounds_;
        }

        // ── Runtime queries. O(1), no allocation, no scene access. ───────

        // Ground height under (x, z). Returns sceneBounds().min.y where nothing
        // was sampled, and 0 when the index is empty.
        [[nodiscard]] float heightAt(float x, float z) const {

            if (height_.empty()) return 0.f;

            const float floorY = bounds_.isEmpty() ? 0.f : bounds_.min().y;

            const int ix = static_cast<int>(std::floor((x - bounds_.min().x) / heightCellX_));
            const int iz = static_cast<int>(std::floor((z - bounds_.min().z) / heightCellZ_));
            if (ix < 0 || ix >= heightN_ || iz < 0 || iz >= heightN_) return floorY;

            const float h = height_[static_cast<std::size_t>(iz) * heightN_ + ix];
            return h > -kUnsampled ? h : floorY;
        }

        // Saturating distance to the nearest occupied cell, in METRES.
        // Returns a large value (chamferPasses · cellSize) outside the grid and
        // when the index is empty — i.e. "wide open", never "blocked".
        //
        // A 0-MEANS-UNKNOWN CONVENTION WOULD BE A DISASTER HERE: the flock's
        // obstacle force reads this directly, so an unbaked scene would behave
        // like solid rock and every bird would spend the run shoving itself away
        // from nothing.
        [[nodiscard]] float clearanceAt(const Vector3& worldPos) const {

            if (dist_.empty()) return emptyClearance();

            const float open = static_cast<float>(saturation_) * cell_;

            int ix, iy, iz;
            if (!cellOf(worldPos.x, worldPos.y, worldPos.z, ix, iy, iz)) return open;

            return static_cast<float>(dist_[cellIndex(ix, iy, iz)]) * cell_;
        }

        // Central-difference gradient of clearanceAt, normalised, pointing AWAY
        // from geometry. Writes (0,0,0) when the sample is already clear or the
        // index is empty.
        void clearanceGradient(const Vector3& worldPos, Vector3& out) const {

            out.set(0, 0, 0);
            if (dist_.empty()) return;

            int ix, iy, iz;
            if (!cellOf(worldPos.x, worldPos.y, worldPos.z, ix, iy, iz)) return;
            if (dist_[cellIndex(ix, iy, iz)] >= saturation_) return;

            const float gx = tap(ix + 1, iy, iz) - tap(ix - 1, iy, iz);
            const float gy = tap(ix, iy + 1, iz) - tap(ix, iy - 1, iz);
            const float gz = tap(ix, iy, iz + 1) - tap(ix, iy, iz - 1);

            // Vector3::normalize() divides by length with only a NaN guard, not a
            // zero guard (Vector3.cpp:403-409), so a flat plateau in the field
            // would hand the caller an infinity. Test the length first, always.
            const float len2 = gx * gx + gy * gy + gz * gz;
            if (len2 < 1e-12f) return;

            const float inv = 1.f / std::sqrt(len2);
            out.set(gx * inv, gy * inv, gz * inv);
        }

        // Indices into spots() whose position is within `radius` of `p`.
        // Appends; caller clears. Results are in ascending spot-index order.
        void querySpots(const Vector3& p, float radius, std::vector<int>& out) const {

            if (spots_.empty() || !(radius > 0.f)) return;

            const std::size_t first = out.size();
            const float r2 = radius * radius;

            if (qStart_.empty()) {
                // No acceleration grid (a degenerate spot set — every spot at one
                // point). Linear, and already in ascending index order.
                for (std::size_t i = 0; i < spots_.size(); ++i) {
                    if (spots_[i].position.distanceToSquared(p) <= r2) out.push_back(static_cast<int>(i));
                }
                return;
            }

            const int lo[3]{qClamp(p.x - radius, 0), qClamp(p.y - radius, 1), qClamp(p.z - radius, 2)};
            const int hi[3]{qClamp(p.x + radius, 0), qClamp(p.y + radius, 1), qClamp(p.z + radius, 2)};

            for (int iz = lo[2]; iz <= hi[2]; ++iz) {
                for (int iy = lo[1]; iy <= hi[1]; ++iy) {
                    for (int ix = lo[0]; ix <= hi[0]; ++ix) {
                        const auto c = static_cast<std::size_t>((iz * qn_ + iy) * qn_ + ix);
                        for (int k = qStart_[c]; k < qStart_[c + 1]; ++k) {
                            const int s = qIndex_[static_cast<std::size_t>(k)];
                            if (spots_[s].position.distanceToSquared(p) <= r2) out.push_back(s);
                        }
                    }
                }
            }

            // A spot lives in exactly one bucket, so there are no duplicates —
            // only the cell-major visit order to undo. std::sort over the range
            // this call appended is in place and allocation-free.
            std::sort(out.begin() + static_cast<std::ptrdiff_t>(first), out.end());
        }

        // Replace the perch table with an authored one. Marks the index
        // complete; leaves the obstacle field and heightfield untouched.
        //
        // Calling this while a bake is in flight CANCELS that bake — the scene
        // scratch is dropped and no further spot will be appended behind the
        // caller's back. The obstacle field keeps whatever it had reached, which
        // for a mid-bake call means a partial field; call clear() first if that
        // matters.
        void setSpots(std::vector<PerchSpot> spots) {

            clearScratch();
            spots_ = std::move(spots);
            buildSpotGrid();
            phase_ = Phase::Done;
            complete_ = true;
        }

        void addSpot(const PerchSpot& spot) {

            clearScratch();
            spots_.push_back(spot);
            buildSpotGrid();
            phase_ = Phase::Done;
            complete_ = true;
        }

    private:
        // ── Bake state machine ───────────────────────────────────────────
        enum class Phase : std::uint8_t {
            Idle = 0,     // nothing begun
            GridAlloc = 1,// size the obstacle grid and the heightfield
            Sample = 2,   // barycentric triangle sampling → occupancy + height
            Chamfer = 3,  // saturating distance transform
            RayGrid = 4,  // per-mesh BVH + downward probe grid
            Finalise = 5, // sort, index, drop every pointer
            Done = 6,
        };

        // Progress weights. Sampling and the ray grid are the two phases that
        // actually cost anything; the rest are rounding error on a real scene.
        static constexpr float kwCollect = 0.03f;
        static constexpr float kwSample = 0.42f;
        static constexpr float kwChamfer = 0.10f;
        static constexpr float kwRayGrid = 0.42f;
        static constexpr float kwFinalise = 0.03f;

        static constexpr float kUnsampled = 1e30f;// heightfield "never written" sentinel
        static constexpr int kMaxGridCells = 2000000;

        // Scene scratch. EVERY MEMBER IN THIS BLOCK IS DEAD BY THE TIME step()
        // RETURNS TRUE — that is the single invariant this file exists to
        // enforce. The geometry is held by shared_ptr rather than raw pointer for
        // the duration of the bake, so a host that frees a mesh half way through
        // an amortised bake gets a stale perch rather than a use-after-free.
        struct MeshEntry {
            std::shared_ptr<BufferGeometry> geometry;
            Matrix4 worldMatrix;
            Box3 worldBox;
            int triCount = 0;
            int vertCount = 0;
        };

        struct Candidate {
            Vector3 position;
            Vector3 normal;
        };

        Params params_{};
        Phase phase_ = Phase::Idle;
        bool complete_ = false;

        // ── Products (pointer-free, transform-free, survive the bake) ────
        std::vector<PerchSpot> spots_;
        Box3 bounds_;

        // Obstacle chamfer field. dist_[c] == 0 ⇔ the cell is occupied, and it
        // stays that way for ever: the relaxation writes 1 + min(neighbours),
        // which can never reach 0 for an unoccupied cell. That invariant is what
        // lets the headroom test be two array reads instead of a second raycast,
        // and it is why there is no separate occupancy array.
        std::vector<std::uint8_t> dist_;
        std::vector<std::uint8_t> chamferScratch_;
        int nx_ = 0, ny_ = 0, nz_ = 0;
        float cell_ = 0.f;
        float invCell_ = 0.f;
        int saturation_ = 0;
        Vector3 gridMin_{};

        // XZ ground heightfield over the same AABB.
        std::vector<float> height_;
        int heightN_ = 0;
        float heightCellX_ = 0.f;
        float heightCellZ_ = 0.f;

        // Spot lookup acceleration (rebuilt whenever the table changes).
        std::vector<int> qStart_;
        std::vector<int> qIndex_;
        int qn_ = 0;
        Vector3 qMin_{};
        Vector3 qCell_{1, 1, 1};

        // ── Bake scratch ─────────────────────────────────────────────────
        std::vector<MeshEntry> meshes_;
        std::vector<Candidate> pending_;
        std::vector<std::uint64_t> thinSlots_;
        std::size_t thinMask_ = 0;
        std::optional<BVH> bvh_;

        std::size_t meshCursor_ = 0;
        int triCursor_ = 0;
        int sampleTriDone_ = 0;
        int sampleTriTotal_ = 0;

        int chamferPass_ = 0;
        int chamferCell_ = 0;

        std::size_t rayMesh_ = 0;
        bool bvhReady_ = false;
        int probeCursor_ = 0;
        int probeNX_ = 0;
        int probeNZ_ = 0;
        float probeStepX_ = 0.f;
        float probeStepZ_ = 0.f;

        // ── Parameter hygiene ────────────────────────────────────────────
        //
        // Clamped on the way in rather than trusted, because two of these turn a
        // typo into a subsystem that quietly does the opposite of its job:
        // chamferPasses ≤ 0 makes the saturated clearance 0, i.e. the whole world
        // reads as solid, and an unbounded cell cap lets a 10 km scene ask for a
        // 4 GB grid.
        [[nodiscard]] static Params sanitise(const Params& in) {

            Params p = in;
            p.probeSpacing = std::max(p.probeSpacing, 1e-3f);
            p.maxProbesPerAxis = std::clamp(p.maxProbesPerAxis, 1, 512);
            p.maxPerches = std::clamp(p.maxPerches, 0, 1 << 20);
            p.perchMinSeparation = std::max(p.perchMinSeparation, 1e-3f);
            p.maxSlope = std::clamp(p.maxSlope, 0.f, 1.5707f);
            p.walkableSlope = std::clamp(p.walkableSlope, 0.f, 1.5707f);
            p.headroomLow = std::max(p.headroomLow, 0.f);
            p.headroomHigh = std::max(p.headroomHigh, p.headroomLow);
            p.cellSize = std::max(p.cellSize, 1e-3f);
            p.maxCellsX = std::clamp(p.maxCellsX, 4, 256);
            p.maxCellsY = std::clamp(p.maxCellsY, 4, 256);
            p.maxCellsZ = std::clamp(p.maxCellsZ, 4, 256);
            p.chamferPasses = std::clamp(p.chamferPasses, 1, 200);
            p.heightGrid = std::clamp(p.heightGrid, 1, 1024);
            p.bvhMaxTriangles = std::max(p.bvhMaxTriangles, 0);
            p.maxSamplesPerTriangleAxis = std::clamp(p.maxSamplesPerTriangleAxis, 1, 16);
            return p;
        }

        [[nodiscard]] static bool excluded(const Mesh& mesh, const Object3D* exclude) {

            if (!exclude) return false;

            // Pointer identity up the parent chain: the flock adds itself to the
            // scene like anything else, so add() order must not matter.
            for (const Object3D* p = &mesh; p; p = p->parent) {
                if (p == exclude) return true;
            }
            return false;
        }

        [[nodiscard]] static float ratio(int done, int total) {

            if (total <= 0) return 1.f;
            return std::clamp(static_cast<float>(done) / static_cast<float>(total), 0.f, 1.f);
        }

        [[nodiscard]] float emptyClearance() const {

            return static_cast<float>(std::max(1, params_.chamferPasses)) * std::max(params_.cellSize, 1e-3f);
        }

        // ── Grid addressing ──────────────────────────────────────────────
        [[nodiscard]] int cellIndex(int ix, int iy, int iz) const {

            return (iz * ny_ + iy) * nx_ + ix;
        }

        [[nodiscard]] bool cellOf(float x, float y, float z, int& ix, int& iy, int& iz) const {

            if (dist_.empty()) return false;

            ix = static_cast<int>(std::floor((x - gridMin_.x) * invCell_));
            iy = static_cast<int>(std::floor((y - gridMin_.y) * invCell_));
            iz = static_cast<int>(std::floor((z - gridMin_.z) * invCell_));

            return ix >= 0 && ix < nx_ && iy >= 0 && iy < ny_ && iz >= 0 && iz < nz_;
        }

        // Out-of-range taps read as the saturated maximum — the same convention
        // the relaxation itself uses, so the gradient at the edge of the grid
        // points outward instead of inventing a wall.
        [[nodiscard]] float tap(int ix, int iy, int iz) const {

            if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) {
                return static_cast<float>(saturation_);
            }
            return static_cast<float>(dist_[cellIndex(ix, iy, iz)]);
        }

        // ── Phase 1 — grid allocation ────────────────────────────────────
        std::int64_t allocateGrids() {

            if (bounds_.isEmpty() || meshes_.empty()) {
                // Box3::getCenter/getSize guard isEmpty() and return (0,0,0)
                // (Box3.cpp:137-153), so an empty scene produces no NaN — but a
                // 1-cell grid at the origin would still be a lie. Skip straight
                // to the finish and let every query answer "wide open".
                phase_ = Phase::Finalise;
                return 1;
            }

            const Vector3 size = bounds_.getSize();

            // Reserve two cells of padding per axis so a surface flush with the
            // scene AABB still has a neighbour cell above it for the headroom
            // test, then grow the cell until the caps AND the total-cell budget
            // are both satisfied. A cap that silently truncated the grid would
            // leave geometry outside it reading as open sky.
            float cell = std::max({params_.cellSize,
                                   size.x / static_cast<float>(params_.maxCellsX - 2),
                                   size.y / static_cast<float>(params_.maxCellsY - 2),
                                   size.z / static_cast<float>(params_.maxCellsZ - 2)});

            auto dim = [&](float extent, int cap) {
                return std::min(cap, static_cast<int>(std::floor(extent / cell)) + 3);
            };

            nx_ = dim(size.x, params_.maxCellsX);
            ny_ = dim(size.y, params_.maxCellsY);
            nz_ = dim(size.z, params_.maxCellsZ);

            while (static_cast<std::int64_t>(nx_) * ny_ * nz_ > kMaxGridCells) {
                cell *= 1.26f;
                nx_ = dim(size.x, params_.maxCellsX);
                ny_ = dim(size.y, params_.maxCellsY);
                nz_ = dim(size.z, params_.maxCellsZ);
            }

            cell_ = cell;
            invCell_ = 1.f / cell_;
            gridMin_.set(bounds_.min().x - cell_, bounds_.min().y - cell_, bounds_.min().z - cell_);
            saturation_ = params_.chamferPasses;

            dist_.assign(static_cast<std::size_t>(nx_) * ny_ * nz_,
                         static_cast<std::uint8_t>(saturation_));

            heightN_ = params_.heightGrid;
            heightCellX_ = std::max(size.x, 1e-3f) / static_cast<float>(heightN_);
            heightCellZ_ = std::max(size.z, 1e-3f) / static_cast<float>(heightN_);
            height_.assign(static_cast<std::size_t>(heightN_) * heightN_, -kUnsampled);

            // Thinning table: open-addressed, linearly probed, fixed capacity.
            // NEVER std::unordered_set — even though this set is only ever
            // membership-tested, the house rule keeps unordered containers out of
            // the deterministic path entirely, because the day someone iterates
            // one to "just check something" is the day the bake stops being
            // reproducible and nobody knows why.
            const std::size_t capacity = detail::nextPow2(
                    static_cast<std::size_t>(std::max(16, params_.maxPerches)) * 4);
            thinSlots_.assign(capacity, 0);
            thinMask_ = capacity - 1;

            phase_ = Phase::Sample;
            return 1;
        }

        // ── Phase 2 — triangle sampling ──────────────────────────────────
        //
        // EVERY mesh is sampled, including the ones far above bvhMaxTriangles
        // that phase 4 will refuse to build a BVH for. A skipped 2 M-triangle
        // terrain would leave the ground heightfield empty exactly where the
        // ground is, and birds would fly through the mountain. A big mesh
        // DEGRADES here — it loses ray-accurate perches and keeps its obstacle
        // cells, its ground height and a coarser set of sampled perches.
        //
        // THE HEIGHTFIELD IS RASTERISED, NOT POINT-SAMPLED, AND THE TWO ARE NOT
        // INTERCHANGEABLE. The barycentric lattice below is sized against the
        // OBSTACLE cell (2 m by default), while the heightfield is 128 columns
        // across the scene — several times finer on anything smaller than a
        // couple of hundred metres. Feeding the heightfield from the same points
        // leaves most of its columns unwritten, and an unwritten column answers
        // `sceneBounds().min.y`: the floor on top of a 6 m roof reads as ground
        // level, the flock's ground force never engages, and birds descend
        // straight through the building. Every column whose centre the triangle
        // actually covers gets the plane's exact height instead — one lattice
        // test per covered column, and the only number in this bake the flock
        // treats as an absolute rather than a hint.
        std::int64_t sampleTriangles(std::int64_t budget) {

            std::int64_t work = 0;
            Vector3 a, b, c, p, n;

            while (work < budget && meshCursor_ < meshes_.size()) {

                MeshEntry& entry = meshes_[meshCursor_];

                // Fetched once per mesh, not once per triangle: getAttribute()
                // hashes a std::string, and doing that two million times turns a
                // bake into a stall.
                auto* position = entry.geometry->getAttribute<float>("position");
                const auto* index = entry.geometry->getIndex();
                if (!position || !index) {
                    ++meshCursor_;
                    triCursor_ = 0;
                    ++work;
                    continue;
                }

                const auto& indices = index->array();
                const auto vertCount = static_cast<unsigned int>(entry.vertCount);
                const bool emitCandidates = entry.triCount > params_.bvhMaxTriangles;

                while (work < budget && triCursor_ < entry.triCount) {

                    const auto base = static_cast<std::size_t>(triCursor_) * 3;
                    ++triCursor_;
                    ++sampleTriDone_;

                    if (base + 2 >= indices.size()) {
                        ++work;
                        continue;
                    }
                    const unsigned int ia = indices[base];
                    const unsigned int ib = indices[base + 1];
                    const unsigned int ic = indices[base + 2];
                    if (ia >= vertCount || ib >= vertCount || ic >= vertCount) {
                        ++work;
                        continue;
                    }

                    fetch(*position, ia, a).applyMatrix4(entry.worldMatrix);
                    fetch(*position, ib, b).applyMatrix4(entry.worldMatrix);
                    fetch(*position, ic, c).applyMatrix4(entry.worldMatrix);

                    if (!isFinite(a) || !isFinite(b) || !isFinite(c)) {
                        ++work;
                        continue;
                    }

                    // Sample density follows the triangle's own cell footprint:
                    // a 40 m ground quad and a 3 cm leaf both end up marking the
                    // cells they actually cover, and neither burns more than
                    // (n+1)(n+2)/2 ≤ 153 samples doing it.
                    const float ex = std::max({a.x, b.x, c.x}) - std::min({a.x, b.x, c.x});
                    const float ey = std::max({a.y, b.y, c.y}) - std::min({a.y, b.y, c.y});
                    const float ez = std::max({a.z, b.z, c.z}) - std::min({a.z, b.z, c.z});
                    const int span = static_cast<int>(std::ceil(std::max({ex, ey, ez}) * invCell_));
                    const int nSteps = std::clamp(span, 1, params_.maxSamplesPerTriangleAxis);
                    const float inv = 1.f / static_cast<float>(nSteps);

                    work += rasteriseHeight(a, b, c);

                    bool slopeOk = false;
                    if (emitCandidates) {
                        // World-space cross product of already-transformed
                        // vertices, so it needs no normal matrix and no
                        // non-uniform-scale correction.
                        Vector3 e1, e2;
                        e1.subVectors(b, a);
                        e2.subVectors(c, a);
                        n.crossVectors(e1, e2);
                        const float len2 = n.lengthSq();
                        if (len2 > 1e-20f) {
                            n.multiplyScalar(1.f / std::sqrt(len2));
                            if (n.y < 0.f) n.negate();
                            slopeOk = n.y > std::cos(params_.maxSlope);
                        }
                    }

                    for (int i = 0; i <= nSteps; ++i) {
                        for (int j = 0; i + j <= nSteps; ++j) {

                            const float u = static_cast<float>(i) * inv;
                            const float v = static_cast<float>(j) * inv;
                            const float w = 1.f - u - v;

                            p.set(a.x * w + b.x * u + c.x * v,
                                  a.y * w + b.y * u + c.y * v,
                                  a.z * w + b.z * u + c.z * v);

                            markOccupied(p);
                            if (slopeOk) offerCandidate(p, n);

                            ++work;
                        }
                    }
                }

                if (triCursor_ >= entry.triCount) {
                    ++meshCursor_;
                    triCursor_ = 0;
                }
            }

            if (meshCursor_ >= meshes_.size()) {
                work += flushCandidates();
                phase_ = Phase::Chamfer;
            }

            return work;
        }

        static Vector3& fetch(const FloatBufferAttribute& attr, unsigned int i, Vector3& out) {

            const auto k = static_cast<std::size_t>(i);
            out.set(attr.getX(k), attr.getY(k), attr.getZ(k));
            return out;
        }

        // Named isFinite, not finite: <cmath> on glibc still declares a
        // ::finite(double), and an unqualified call inside a template-heavy
        // header is not the place to discover that.
        [[nodiscard]] static bool isFinite(const Vector3& v) {

            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        void markOccupied(const Vector3& p) {

            int ix, iy, iz;
            if (!cellOf(p.x, p.y, p.z, ix, iy, iz)) return;
            dist_[cellIndex(ix, iy, iz)] = 0;
        }

        [[nodiscard]] int columnX(float x) const {

            return static_cast<int>(std::floor((x - bounds_.min().x) / heightCellX_));
        }

        [[nodiscard]] int columnZ(float z) const {

            return static_cast<int>(std::floor((z - bounds_.min().z) / heightCellZ_));
        }

        void markHeightPoint(const Vector3& p) {

            if (height_.empty()) return;

            const int ix = columnX(p.x);
            const int iz = columnZ(p.z);
            if (ix < 0 || ix >= heightN_ || iz < 0 || iz >= heightN_) return;

            float& h = height_[static_cast<std::size_t>(iz) * heightN_ + ix];
            if (p.y > h) h = p.y;
        }

        // Conservative XZ rasterisation of one world-space triangle into the
        // heightfield, charging one work unit per column tested. The cost is
        // proportional to the footprint the triangle actually covers, so a
        // tiled terrain pays for its columns exactly once no matter how it is
        // tessellated, and a scene-spanning quad is bounded by heightGrid².
        std::int64_t rasteriseHeight(const Vector3& a, const Vector3& b, const Vector3& c) {

            if (height_.empty()) return 0;

            int ix0 = columnX(std::min({a.x, b.x, c.x}));
            int ix1 = columnX(std::max({a.x, b.x, c.x}));
            int iz0 = columnZ(std::min({a.z, b.z, c.z}));
            int iz1 = columnZ(std::max({a.z, b.z, c.z}));
            if (ix1 < 0 || iz1 < 0 || ix0 >= heightN_ || iz0 >= heightN_) return 1;

            ix0 = std::max(ix0, 0);
            iz0 = std::max(iz0, 0);
            ix1 = std::min(ix1, heightN_ - 1);
            iz1 = std::min(iz1, heightN_ - 1);

            const float det = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z);

            std::int64_t work = 0;
            int touched = 0;

            if (std::abs(det) > 1e-12f) {

                const float invDet = 1.f / det;

                for (int iz = iz0; iz <= iz1; ++iz) {
                    const float pz = bounds_.min().z + (static_cast<float>(iz) + 0.5f) * heightCellZ_;
                    for (int ix = ix0; ix <= ix1; ++ix) {
                        const float px = bounds_.min().x + (static_cast<float>(ix) + 0.5f) * heightCellX_;
                        ++work;

                        const float l1 = ((b.z - c.z) * (px - c.x) + (c.x - b.x) * (pz - c.z)) * invDet;
                        const float l2 = ((c.z - a.z) * (px - c.x) + (a.x - c.x) * (pz - c.z)) * invDet;
                        const float l3 = 1.f - l1 - l2;
                        if (l1 < -1e-5f || l2 < -1e-5f || l3 < -1e-5f) continue;

                        const float y = l1 * a.y + l2 * b.y + l3 * c.y;
                        float& h = height_[static_cast<std::size_t>(iz) * heightN_ + ix];
                        if (y > h) h = y;
                        ++touched;
                    }
                }
            }

            if (touched == 0) {
                // Edge-on in XZ, or smaller than a column: the triangle covers no
                // column centre at all. Record its vertices instead — a wall's top
                // edge is genuinely the highest thing in the column it stands in,
                // and dropping it punches a hole in the floor exactly where a
                // building is.
                markHeightPoint(a);
                markHeightPoint(b);
                markHeightPoint(c);
                work += 3;
            }

            return work;
        }

        // A sampled candidate from a mesh phase 4 will skip. Slope and thinning
        // are settled here, in traversal order; headroom is not, because the
        // occupancy field is only complete once every mesh has been sampled.
        // The deferred half runs in flushCandidates(), still in insertion order,
        // so the accepted set stays a pure function of the traversal.
        void offerCandidate(const Vector3& p, const Vector3& n) {

            if (acceptedCount() >= params_.maxPerches) return;
            if (!reserveThinCell(p)) return;
            pending_.push_back(Candidate{p, n});
        }

        std::int64_t flushCandidates() {

            const auto work = static_cast<std::int64_t>(pending_.size()) + 1;

            for (const auto& candidate : pending_) {
                if (!headroomClear(candidate.position)) continue;
                pushSpot(candidate.position, candidate.normal);
            }
            pending_.clear();
            pending_.shrink_to_fit();

            return work;
        }

        // ── Phase 3 — the chamfer field ──────────────────────────────────
        //
        // Saturating L1 distance transform by repeated 6-neighbour relaxation,
        // DOUBLE-BUFFERED so pass order cannot matter. In-place relaxation would
        // propagate a distance several cells in a single sweep along whichever
        // axis the loop happens to run, which makes the field depend on the
        // iteration order and, worse, on where a step() happened to stop.
        std::int64_t relaxChamfer(std::int64_t budget) {

            const auto total = static_cast<int>(dist_.size());
            if (total == 0) {
                phase_ = Phase::RayGrid;
                return 1;
            }

            if (chamferScratch_.size() != dist_.size()) {
                chamferScratch_.assign(dist_.size(), 0);
            }

            std::int64_t work = 0;
            const auto sat = static_cast<std::uint8_t>(saturation_);
            const int planeStride = nx_ * ny_;

            while (work < budget && chamferPass_ < saturation_) {

                const auto remaining = static_cast<int>(std::min<std::int64_t>(budget - work, total - chamferCell_));
                const int end = chamferCell_ + std::max(1, remaining);
                const int stop = std::min(end, total);

                for (int c = chamferCell_; c < stop; ++c) {

                    std::uint8_t best = dist_[c];
                    if (best != 0) {
                        const int ix = c % nx_;
                        const int t = c / nx_;
                        const int iy = t % ny_;
                        const int iz = t / ny_;

                        std::uint8_t m = sat;
                        if (ix > 0) m = std::min(m, dist_[c - 1]);
                        if (ix < nx_ - 1) m = std::min(m, dist_[c + 1]);
                        if (iy > 0) m = std::min(m, dist_[c - nx_]);
                        if (iy < ny_ - 1) m = std::min(m, dist_[c + nx_]);
                        if (iz > 0) m = std::min(m, dist_[c - planeStride]);
                        if (iz < nz_ - 1) m = std::min(m, dist_[c + planeStride]);

                        best = static_cast<std::uint8_t>(std::min<int>(best, static_cast<int>(m) + 1));
                    }
                    chamferScratch_[c] = best;
                }

                work += (stop - chamferCell_);
                chamferCell_ = stop;

                if (chamferCell_ >= total) {
                    dist_.swap(chamferScratch_);
                    chamferCell_ = 0;
                    ++chamferPass_;
                }
            }

            if (chamferPass_ >= saturation_) {
                chamferScratch_.clear();
                chamferScratch_.shrink_to_fit();
                phase_ = Phase::RayGrid;
            }

            return std::max<std::int64_t>(work, 1);
        }

        // ── Phase 4 — the BVH ray grid ───────────────────────────────────
        //
        // A BVH BUILD IS ATOMIC AND CANNOT BE SPLIT. BVH::buildNode sorts the
        // index array at every level with a comparator doing two random-access
        // getMidpoint calls into a 36-byte-per-triangle array; at 40 000
        // triangles that is roughly 0.1 s of one frozen frame. bvhMaxTriangles
        // therefore defaults to 40 000 and not to the quarter-million a
        // "just build everything" design would use — and a mesh above the limit
        // is not skipped, it is served by the phase 2 sampler instead.
        std::int64_t castProbeGrid(std::int64_t budget) {

            std::int64_t work = 0;

            while (work < budget && rayMesh_ < meshes_.size()) {

                MeshEntry& entry = meshes_[rayMesh_];

                if (!bvhReady_) {
                    if (entry.triCount > params_.bvhMaxTriangles || entry.worldBox.isEmpty()) {
                        ++rayMesh_;
                        ++work;
                        continue;
                    }

                    // Deeper than the default cap of 10: at 40 000 triangles a
                    // depth-10 tree ends with ~40-triangle leaves, and the probe
                    // grid pays for every one of them. This changes cost only —
                    // raycast() returns the same closest hit at any depth.
                    int subdivisions = 10;
                    while ((1 << subdivisions) * 8 < entry.triCount && subdivisions < 20) ++subdivisions;

                    bvh_.emplace(8, subdivisions);
                    bvh_->build(*entry.geometry);
                    work += entry.triCount;

                    setUpProbeGrid(entry);
                    bvhReady_ = true;
                    probeCursor_ = 0;
                    continue;
                }

                const int probeCount = probeNX_ * probeNZ_;
                while (work < budget && probeCursor_ < probeCount) {
                    castProbe(entry, probeCursor_);
                    ++probeCursor_;
                    work += 20;
                }

                if (probeCursor_ >= probeCount) {
                    // Destroyed the instant this mesh's rays are done, not at the
                    // end of the bake: nothing that outlives a step() may hold a
                    // BufferGeometry*, and BVH holds one (BVH.hpp:96).
                    bvh_.reset();
                    bvhReady_ = false;
                    ++rayMesh_;
                }
            }

            if (rayMesh_ >= meshes_.size()) {
                bvh_.reset();
                bvhReady_ = false;
                phase_ = Phase::Finalise;
            }

            return std::max<std::int64_t>(work, 1);
        }

        void setUpProbeGrid(const MeshEntry& entry) {

            const Vector3 size = entry.worldBox.getSize();

            probeNX_ = std::clamp(static_cast<int>(std::floor(size.x / params_.probeSpacing)) + 1,
                                  1, params_.maxProbesPerAxis);
            probeNZ_ = std::clamp(static_cast<int>(std::floor(size.z / params_.probeSpacing)) + 1,
                                  1, params_.maxProbesPerAxis);

            probeStepX_ = probeNX_ > 1 ? size.x / static_cast<float>(probeNX_ - 1) : 0.f;
            probeStepZ_ = probeNZ_ > 1 ? size.z / static_cast<float>(probeNZ_ - 1) : 0.f;
        }

        void castProbe(const MeshEntry& entry, int k) {

            const int ix = k % probeNX_;
            const int iz = k / probeNX_;

            const float x = probeNX_ > 1
                                    ? entry.worldBox.min().x + static_cast<float>(ix) * probeStepX_
                                    : entry.worldBox.min().x + entry.worldBox.getSize().x * 0.5f;
            const float z = probeNZ_ > 1
                                    ? entry.worldBox.min().z + static_cast<float>(iz) * probeStepZ_
                                    : entry.worldBox.min().z + entry.worldBox.getSize().z * 0.5f;

            const Vector3 origin{x, entry.worldBox.max().y + 1.f, z};
            const Vector3 down{0, -1, 0};
            const float maxDistance = (entry.worldBox.max().y - entry.worldBox.min().y) + 2.f;

            Matrix4 inverse;
            inverse.copy(entry.worldMatrix).invert();

            Vector3 localOrigin, localDir;
            localOrigin.copy(origin).applyMatrix4(inverse);

            // Directions carry no translation, so transform a point one unit along
            // the ray instead; its length is how many local units a world unit is.
            // (This is AcousticScene::closestHit, which is private —
            // src/threepp/audio/Acoustics.cpp:165-204 — reimplemented verbatim.)
            localDir.copy(origin).add(down).applyMatrix4(inverse).sub(localOrigin);
            const float scale = localDir.length();
            if (scale < 1e-9f) return;
            localDir.multiplyScalar(1.f / scale);

            // rayEps is 1e-4 and the effective range is (maxDistance - rayEps)
            // (BVH.cpp:14, 291, 338): a probe started exactly on a surface
            // legitimately misses. Offset the origin, as the acoustics probe does.
            localOrigin.addScaledVector(localDir, 1e-4f);

            const auto hit = bvh_->raycast(Ray(localOrigin, localDir), maxDistance * scale);
            if (!hit) return;

            Vector3 point = hit->point;
            point.applyMatrix4(entry.worldMatrix);

            Matrix3 normalMatrix;
            normalMatrix.getNormalMatrix(entry.worldMatrix);
            Vector3 normal = hit->normal;
            normal.applyNormalMatrix(normalMatrix);

            // BVH::RayHit::normal is ALREADY flipped toward the ray origin
            // (BVH.hpp:47), unlike Intersection::face->normal, which is neither
            // flipped nor in world space. Mixing the two conventions silently
            // inverts half the perch set; this bake uses only the BVH one. The
            // guard below exists solely for a mirrored (negative-determinant)
            // world matrix, which the normal matrix can flip back over.
            if (normal.y < 0.f) normal.negate();

            if (!isFinite(point) || !isFinite(normal)) return;

            offerHit(point, normal);
        }

        // ── Acceptance and thinning (§6.5) ───────────────────────────────
        void offerHit(const Vector3& p, const Vector3& n) {

            if (acceptedCount() >= params_.maxPerches) return;
            if (!(n.y > std::cos(params_.maxSlope))) return;
            if (!headroomClear(p)) return;
            if (!reserveThinCell(p)) return;

            pushSpot(p, n);
        }

        [[nodiscard]] int acceptedCount() const {

            return static_cast<int>(spots_.size() + pending_.size());
        }

        void pushSpot(const Vector3& p, const Vector3& n) {

            PerchSpot spot;
            spot.position = p;
            spot.normal = n;
            spot.walkable = n.y > std::cos(params_.walkableSlope);
            spot.ground = (p.y - heightAt(p.x, p.z)) < params_.groundEpsilon;
            spots_.push_back(spot);
        }

        // HEADROOM MUST EXEMPT THE SPOT'S OWN CELL, AND THIS IS THE ONE PLACE THE
        // OBVIOUS READING PRODUCES A BAKE WITH ZERO PERCHES IN IT.
        //
        // The cells are 2 m and the low headroom sample is 0.30 m up, so
        // p + (0, headroomLow, 0) almost always lands in the SAME cell as the
        // spot — the cell the surface itself occupies, and therefore always
        // marked. Tested naively, every perch on every flat roof in the scene is
        // rejected, the spot table comes back empty, nothing logs, and the bug
        // surfaces days later as "my birds never land". A sample that resolves to
        // the spot's own cell is treated as clear; the test then does what it was
        // meant to do, which is reject spots under an overhang or a canopy.
        [[nodiscard]] bool headroomClear(const Vector3& p) const {

            if (dist_.empty()) return true;

            int sx, sy, sz;
            const bool inside = cellOf(p.x, p.y, p.z, sx, sy, sz);

            auto clear = [&](float dy) {
                int ix, iy, iz;
                if (!cellOf(p.x, p.y + dy, p.z, ix, iy, iz)) return true;// outside the grid ⇒ open sky
                if (inside && ix == sx && iy == sy && iz == sz) return true;
                return dist_[cellIndex(ix, iy, iz)] != 0;
            };

            return clear(params_.headroomLow) && clear(params_.headroomHigh);
        }

        // Open-addressed, linearly probed, fixed capacity. Returns false when the
        // lattice cell is already taken — an UNTHINNED grid puts birds at exact
        // bake centres, evenly spaced and shoulder to shoulder on invisible
        // lines, which is the single loudest giveaway that a surface was sampled
        // by a program.
        bool reserveThinCell(const Vector3& p) {

            if (thinSlots_.empty()) return true;

            const std::uint64_t key = detail::latticeKey(p, 1.f / params_.perchMinSeparation);
            const std::uint64_t stored = key + 1;

            std::size_t slot = static_cast<std::size_t>(detail::mix64(key)) & thinMask_;
            for (std::size_t probe = 0; probe <= thinMask_; ++probe) {
                if (thinSlots_[slot] == 0) {
                    thinSlots_[slot] = stored;
                    return true;
                }
                if (thinSlots_[slot] == stored) return false;
                slot = (slot + 1) & thinMask_;
            }
            return false;// table full — the maxPerches cap makes this unreachable
        }

        // ── Phase 5 — finalise ───────────────────────────────────────────
        std::int64_t finalise() {

            sortSpotsByMorton();
            buildSpotGrid();
            clearScratch();

            phase_ = Phase::Done;
            complete_ = true;
            return 1;
        }

        // Morton order over the quantised position, ties by insertion index
        // (std::stable_sort gives the tie rule for free). Spots that are near
        // each other in space end up near each other in the table, which is what
        // makes the per-cell scan in querySpots cache-friendly — and it is a pure
        // function of the accepted set, so it costs the determinism contract
        // nothing.
        void sortSpotsByMorton() {

            if (spots_.size() < 2) return;

            Box3 spotBox;
            spotBox.makeEmpty();
            for (const auto& s : spots_) spotBox.expandByPoint(s.position);

            const Vector3 size = spotBox.getSize();
            const float sx = size.x > 1e-6f ? 2097151.f / size.x : 0.f;
            const float sy = size.y > 1e-6f ? 2097151.f / size.y : 0.f;
            const float sz = size.z > 1e-6f ? 2097151.f / size.z : 0.f;

            std::vector<std::uint64_t> keys(spots_.size());
            for (std::size_t i = 0; i < spots_.size(); ++i) {
                const Vector3& p = spots_[i].position;
                const auto qx = static_cast<std::uint32_t>(std::clamp((p.x - spotBox.min().x) * sx, 0.f, 2097151.f));
                const auto qy = static_cast<std::uint32_t>(std::clamp((p.y - spotBox.min().y) * sy, 0.f, 2097151.f));
                const auto qz = static_cast<std::uint32_t>(std::clamp((p.z - spotBox.min().z) * sz, 0.f, 2097151.f));
                keys[i] = detail::morton3(qx, qy, qz);
            }

            std::vector<int> order(spots_.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int lhs, int rhs) { return keys[lhs] < keys[rhs]; });

            std::vector<PerchSpot> sorted;
            sorted.reserve(spots_.size());
            for (const int i : order) sorted.push_back(spots_[static_cast<std::size_t>(i)]);
            spots_.swap(sorted);
        }

        // ── Spot lookup grid ─────────────────────────────────────────────
        //
        // A CSR bucket grid rebuilt whenever the table changes. querySpots() is
        // called a few times per bird per second, so it has to be allocation-free
        // — which rules out building this lazily inside a const query.
        void clearSpotGrid() {

            qStart_.clear();
            qStart_.shrink_to_fit();
            qIndex_.clear();
            qIndex_.shrink_to_fit();
            qn_ = 0;
            qMin_.set(0, 0, 0);
            qCell_.set(1, 1, 1);
        }

        void buildSpotGrid() {

            clearSpotGrid();
            if (spots_.empty()) return;

            Box3 box;
            box.makeEmpty();
            for (const auto& s : spots_) box.expandByPoint(s.position);
            if (box.isEmpty()) return;

            const Vector3 size = box.getSize();
            qn_ = std::clamp(static_cast<int>(std::cbrt(static_cast<double>(spots_.size()))) + 1, 1, 32);
            qMin_.copy(box.min());
            qCell_.set(std::max(size.x, 1e-3f) / static_cast<float>(qn_),
                       std::max(size.y, 1e-3f) / static_cast<float>(qn_),
                       std::max(size.z, 1e-3f) / static_cast<float>(qn_));

            const int cells = qn_ * qn_ * qn_;
            qStart_.assign(static_cast<std::size_t>(cells) + 1, 0);
            qIndex_.assign(spots_.size(), 0);

            std::vector<int> cellOfSpot(spots_.size(), 0);
            for (std::size_t i = 0; i < spots_.size(); ++i) {
                const Vector3& p = spots_[i].position;
                const int ix = qClamp(p.x, 0);
                const int iy = qClamp(p.y, 1);
                const int iz = qClamp(p.z, 2);
                const int c = (iz * qn_ + iy) * qn_ + ix;
                cellOfSpot[i] = c;
                ++qStart_[static_cast<std::size_t>(c) + 1];
            }
            for (int c = 0; c < cells; ++c) qStart_[static_cast<std::size_t>(c) + 1] += qStart_[static_cast<std::size_t>(c)];

            std::vector<int> cursor(qStart_.begin(), qStart_.end() - 1);
            for (std::size_t i = 0; i < spots_.size(); ++i) {
                qIndex_[static_cast<std::size_t>(cursor[static_cast<std::size_t>(cellOfSpot[i])]++)] = static_cast<int>(i);
            }
        }

        [[nodiscard]] int qClamp(float v, int axis) const {

            const float base = axis == 0 ? qMin_.x : (axis == 1 ? qMin_.y : qMin_.z);
            const float cell = axis == 0 ? qCell_.x : (axis == 1 ? qCell_.y : qCell_.z);
            return std::clamp(static_cast<int>(std::floor((v - base) / cell)), 0, qn_ - 1);
        }

        // ── Scratch teardown ─────────────────────────────────────────────
        //
        // Everything here holds, directly or indirectly, a pointer into the host
        // scene. After this runs the index is a set of plain world-space values,
        // and nothing the host does to its scene can make a query misbehave.
        void clearScratch() {

            bvh_.reset();
            bvhReady_ = false;

            meshes_.clear();
            meshes_.shrink_to_fit();
            pending_.clear();
            pending_.shrink_to_fit();
            thinSlots_.clear();
            thinSlots_.shrink_to_fit();
            thinMask_ = 0;

            meshCursor_ = 0;
            triCursor_ = 0;
            sampleTriDone_ = 0;
            sampleTriTotal_ = 0;
            chamferPass_ = 0;
            chamferCell_ = 0;
            rayMesh_ = 0;
            probeCursor_ = 0;
            probeNX_ = 0;
            probeNZ_ = 0;
            probeStepX_ = 0.f;
            probeStepZ_ = 0.f;
        }
    };

}// namespace threepp::fauna

#endif// THREEPP_EXTRAS_FAUNA_PERCHINDEX_HPP
