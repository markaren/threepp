// The pour: a rate-accumulating lattice emitter for PBD granular material.
//
// Promoted out of examples/projects/Physics/granular_conveyor.cpp, where it
// started life as that demo's `Chute` and still lives (converting the example
// to this is optional cleanup, not a requirement). One thing changed on the
// way here, and it is what makes the type authorable: the demo's mouth was a
// fixed world-space point, this one is handed the chute's WORLD TRANSFORM on
// every tick. The lattice is laid out in the chute's own frame and carried
// into the world by that matrix, so a node being moved by a gizmo, a script or
// an animation pours from where it is now — and rotating it aims the pour,
// because the caller turns the launch velocity by the same matrix.
//
// Emission has to be NON-OVERLAPPING: PBD depenetrates an overlap violently
// unless the velocity clamp does all the work, so grains are placed on a
// lattice inside a thin slab and jittered by a fraction of a cell, never
// sampled uniformly. The slab grows in layers only as far as the burst needs,
// and the slots of a burst are SHUFFLED so a partial layer arrives as a
// scatter rather than a solid front.
//
// PhysX-free by construction — it produces positions and nothing else. Handing
// them to PbdParticles::Group::emit is the caller's job, and so is the
// velocity: this only says where.

#ifndef THREEPP_PHYSX_CHUTEEMITTER_HPP
#define THREEPP_PHYSX_CHUTEEMITTER_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace threepp {

    class ChuteEmitter {

    public:
        struct Settings {
            // Half-size of the pour mouth in the chute's own XZ plane. The cell
            // count is derived from it, so a mouth narrower than one cell still
            // pours — as a single column.
            float halfExtentX = 0.2f;
            float halfExtentZ = 0.2f;
            // Grain diameter (PbdParticles::Settings::spacing). The lattice
            // cell is this with 5% of clearance on it, which is what keeps two
            // freshly emitted neighbours from starting in contact.
            float spacing = 0.06f;
            // Jitter as a fraction of `spacing`, applied per axis, so that a
            // pour does not read as a marching grid. 0.2 is the value the demo
            // was tuned at; 1 is deliberately reachable and deliberately
            // violent — at that point the jitter exceeds the cell clearance and
            // the solver is depenetrating every burst.
            float jitter = 0.2f;
            unsigned seed = 20260812u;
        };

        explicit ChuteEmitter(const Settings& settings)
            : settings_(settings), rng_(settings.seed) {

            cell_ = std::max(settings_.spacing, 1e-4f) * 1.05f;
            cellsX_ = std::max(1u, unsigned(2.f * std::max(settings_.halfExtentX, 0.f) / cell_));
            cellsZ_ = std::max(1u, unsigned(2.f * std::max(settings_.halfExtentZ, 0.f) / cell_));
        }

        // Accumulate `rate` grains per second across `dt` and lay the burst out
        // in the frame `chuteWorld` describes. The accumulator is fractional so
        // a 42.7-grains-per-frame pour does not quantise to 42.
        //
        // `budget` is the room left in the buffer the burst is destined for.
        // What it clips is DROPPED rather than carried: a full group has
        // stopped pouring, and a remainder banked against a capacity that never
        // frees would come back as one impossible burst.
        //
        // Layers stack along the chute's local +Y, i.e. back up the direction a
        // default (-Y) pour travels — the slab fills away from the mouth.
        const std::vector<Vector3>& tick(float dt, float rate, const Matrix4& chuteWorld,
                                         unsigned budget = std::numeric_limits<unsigned>::max()) {

            pending_ += dt * std::max(rate, 0.f);
            auto want = unsigned(pending_);
            pending_ -= float(want);

            out_.clear();
            want = std::min(want, budget);
            if (want == 0) return out_;

            const unsigned perLayer = cellsX_ * cellsZ_;
            const unsigned layers = (want + perLayer - 1) / perLayer;
            slots_.resize(std::size_t(perLayer) * layers);
            std::iota(slots_.begin(), slots_.end(), 0u);
            std::shuffle(slots_.begin(), slots_.end(), rng_);

            const float amplitude = std::max(settings_.jitter, 0.f) * settings_.spacing;
            std::uniform_real_distribution<float> jitter(-amplitude, amplitude);
            out_.reserve(want);
            for (unsigned i = 0; i < want; ++i) {
                const unsigned s = slots_[i];
                const unsigned ix = s % cellsX_;
                const unsigned iz = (s / cellsX_) % cellsZ_;
                const unsigned iy = s / perLayer;
                Vector3 point((float(ix) - float(cellsX_ - 1) * 0.5f) * cell_ + jitter(rng_),
                              float(iy) * cell_ + jitter(rng_),
                              (float(iz) - float(cellsZ_ - 1) * 0.5f) * cell_ + jitter(rng_));
                // The full matrix, scale included: a scaled chute has a scaled
                // mouth, the same way a scaled conveyor has scaled bends.
                out_.push_back(point.applyMatrix4(chuteWorld));
            }
            return out_;
        }

        // The lattice, for a caller drawing the mouth or sizing a slab.
        [[nodiscard]] unsigned cellsX() const { return cellsX_; }
        [[nodiscard]] unsigned cellsZ() const { return cellsZ_; }
        [[nodiscard]] float cellSize() const { return cell_; }

    private:
        Settings settings_;
        std::mt19937 rng_;
        float cell_ = 0.f;
        unsigned cellsX_ = 1, cellsZ_ = 1;
        float pending_ = 0.f;
        std::vector<unsigned> slots_;
        std::vector<Vector3> out_;
    };

}// namespace threepp

#endif// THREEPP_PHYSX_CHUTEEMITTER_HPP
