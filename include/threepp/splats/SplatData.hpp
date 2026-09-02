// In-memory model for a 3D Gaussian Splat cloud, plus a deterministic
// procedural generator for it.
//
// Provenance: clean-room. The semantics are the ones glTF KHR_gaussian_splatting
// and OpenUSD UsdVolParticleField3DGaussianSplat independently agree on; the
// math is from arXiv 2308.04079. No third-party splatting code was consulted.
//
// Canonical units — everything in here is already *activated*, so a renderer
// never has to know what a file happened to store:
//
//   mean      local-space centre of the Gaussian
//   scale     LINEAR and non-negative (importers apply exp() to log-scale)
//   rotation  unit quaternion, threepp order (x, y, z, w) — stored as
//             SplatQuat, four plain floats, for the reason given there
//   opacity   [0, 1] (importers apply sigmoid() to logits)
//   sh        radiance, float3 per coefficient, degrees 0-3
//
// SH storage is COEFFICIENT-MAJOR: sh[(splat * coeffCount + c) * 3 + channel].
// The INRIA PLY stores the higher-order coefficients channel-major instead;
// SplatLoader reorders on import, and SplatLoader_test pins that it does.

#ifndef THREEPP_SPLATDATA_HPP
#define THREEPP_SPLATDATA_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/splats/SplatSH.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace threepp {

    namespace splats {

        // The two activations every 3DGS file needs on import, and their
        // inverses (which the test-only PLY writer needs on export).
        [[nodiscard]] inline float sigmoid(float x) {

            return 1.f / (1.f + std::exp(-x));
        }

        [[nodiscard]] inline float logit(float p) {

            return std::log(p / (1.f - p));
        }

        // Median distance from a point to its nearest neighbour, over a
        // fixed-stride sample of at most `sampleCount` points tested against
        // a hash grid of the whole set. Exact for a sample point whose
        // neighbour lies within one grid cell (twice the mean spacing the
        // bounds imply); farther than that the search widens three cells and
        // then reports the search radius. 0 for fewer than two finite points.
        //
        // Sizes the Gaussians a colour-only point cloud is imported as
        // (SplatLoader::loadPointCloudPly); nothing on the render path calls
        // it. O(n log n) in the point count.
        [[nodiscard]] float medianNeighbourSpacing(const std::vector<Vector3>& points,
                                                   size_t sampleCount = 20000);

    }// namespace splats

    // A splat rotation: four plain floats in threepp order (x, y, z, w),
    // default identity.
    //
    // NOT threepp::Quaternion, and the reason is arithmetic. Quaternion's four
    // components are float_view — a float plus a pointer to the owner's change
    // callback, 16 bytes each — and it carries a std::function<void()> member
    // besides, so it spends 128 bytes to hold 16 bytes of rotation. In a
    // per-splat array that is 112 bytes of pure overhead: two thirds of a
    // degree-0 splat, and 672 MB on a six-million-splat scan, paid on both
    // backends and before any renderer is involved. Nothing in the splat path
    // subscribes to the notification it buys — rotations are written once by a
    // loader or the generator and read by computeCovariance and the GPU
    // uploads. The onChange plumbing exists for Object3D transform
    // propagation, which is a different problem.
    //
    // Converts implicitly FROM Quaternion, so a caller holding one can still
    // just assign it. The other direction is named, because materialising a
    // Quaternion is exactly the cost this type exists to avoid.
    struct SplatQuat {

        float x{0.f};
        float y{0.f};
        float z{0.f};
        float w{1.f};

        SplatQuat() = default;

        SplatQuat(float x, float y, float z, float w)
            : x(x), y(y), z(z), w(w) {}

        SplatQuat(const Quaternion& q)
            : x(q.x), y(q.y), z(q.z), w(q.w) {}

        [[nodiscard]] Quaternion toQuaternion() const {

            return Quaternion(x, y, z, w);
        }

        SplatQuat& set(float x, float y, float z, float w) {

            this->x = x;
            this->y = y;
            this->z = z;
            this->w = w;

            return *this;
        }

        [[nodiscard]] float lengthSq() const {

            return x * x + y * y + z * z + w * w;
        }

        [[nodiscard]] float length() const {

            return std::sqrt(lengthSq());
        }

        // Reciprocal-multiply and the zero-length identity fallback, both
        // exactly as Quaternion::normalize does them. Mirroring that float
        // sequence, rather than writing the obvious `x /= length()`, is what
        // makes changing the storage type a no-op on the VALUES — the generator
        // feeds SplatData_test's determinism pin, the loader round trips, and
        // VulkanSplat_test's golden image, and a last-ulp drift here would
        // disturb all three to no purpose.
        SplatQuat& normalize() {

            auto l = length();

            if (l == 0) {

                x = 0.f;
                y = 0.f;
                z = 0.f;
                w = 1.f;

            } else {

                l = 1.f / l;

                x *= l;
                y *= l;
                z *= l;
                w *= l;
            }

            return *this;
        }

        bool operator==(const SplatQuat& o) const {

            return x == o.x && y == o.y && z == o.z && w == o.w;
        }

        bool operator!=(const SplatQuat& o) const {

            return !(*this == o);
        }
    };

    // The entire point of the type. A regression here is 112 bytes a splat.
    static_assert(sizeof(SplatQuat) == 4 * sizeof(float),
                  "SplatQuat must stay four plain floats");

    // Struct-of-arrays. All the per-splat vectors are the same length except
    // `sh`, which is count() * coeffCount() * 3.
    struct SplatData {

        std::vector<Vector3> means;
        std::vector<Vector3> scales;      // linear, non-negative
        std::vector<SplatQuat> rotations; // unit
        std::vector<float> opacities;     // [0, 1]
        std::vector<float> sh;            // coefficient-major, see header comment

        int shDegree = 0;

        // The escape hatch. Every PLY property the loader did not consume,
        // kept as a name -> per-splat float array — robotics feature fields
        // (semantic labels, per-splat confidences, learned descriptors) ride
        // in exactly this way, and dropping them would make the loader lossy.
        // Ordered so that a re-export is deterministic.
        std::map<std::string, std::vector<float>> extras;

        [[nodiscard]] size_t count() const {

            return means.size();
        }

        [[nodiscard]] int coeffCount() const {

            return splats::shCoeffCount(shDegree);
        }

        // Bytes the arrays hold. Reported from capacity() rather than size():
        // a loader that reserved for the file's splat count and then dropped
        // outliers is still holding the memory it reserved, and this number
        // exists to be budgeted against, not to look tidy.
        //
        // `sh` dominates and by a lot — at degree 3 it is 192 of the 236 bytes a
        // splat costs, which is why the loaders offer a degree cap at all. The
        // other 44 are mean, scale, rotation and opacity, and they come to 44
        // only because the rotation is four plain floats: see SplatQuat, which
        // is what this comment used to describe and the struct did not do.
        // Measured, degrees 0-3: 56, 92, 152, 236 bytes a splat.
        [[nodiscard]] std::size_t byteSize() const {

            std::size_t bytes = means.capacity() * sizeof(Vector3) +
                                scales.capacity() * sizeof(Vector3) +
                                rotations.capacity() * sizeof(SplatQuat) +
                                opacities.capacity() * sizeof(float) +
                                sh.capacity() * sizeof(float);

            for (const auto& [name, values] : extras) {
                bytes += name.capacity() + values.capacity() * sizeof(float);
            }
            return bytes;
        }

        // Sizes every array for `n` splats at `degree`, zero-filled. Rotations
        // become identity, so a partially-filled cloud is still renderable.
        void resize(size_t n, int degree);

        // Import-time hygiene: the file's quaternions are arbitrary length.
        void normalizeRotations();

        [[nodiscard]] float* shAt(size_t splat) {

            return sh.data() + splat * static_cast<size_t>(coeffCount()) * 3;
        }

        [[nodiscard]] const float* shAt(size_t splat) const {

            return sh.data() + splat * static_cast<size_t>(coeffCount()) * 3;
        }

        // Colour this splat shows when viewed from `viewDir` (a unit vector
        // pointing FROM the camera TO the splat, which is the direction the
        // 3DGS basis is parameterised by).
        [[nodiscard]] Vector3 colorAt(size_t splat, const Vector3& viewDir) const {

            return splats::evalSh(shAt(splat), shDegree, viewDir);
        }

        // Sets the DC coefficient so the splat renders as `rgb` from every
        // direction (higher bands untouched — zero them first for a flat look).
        void setDcColor(size_t splat, const Vector3& rgb);

        // Sigma = R * S * S^T * R^T for splat `i`, written to `out` as the six
        // distinct entries in the order (xx, xy, xz, yy, yz, zz). This is what
        // the renderer uploads: doing it once at load beats redoing the
        // quaternion-to-matrix conversion per frame per splat in the shader.
        void computeCovariance(size_t i, float* out) const;

        // Axis-aligned bounds of the means, dilated by `sigma` standard
        // deviations of each splat's own largest axis. Empty for an empty cloud.
        [[nodiscard]] Box3 computeBounds(float sigma = 3.f) const;

        // Thresholds for removeOutliers(). Every one is a ratio against the
        // cloud's own distribution, so nothing in here carries a unit and the
        // same numbers behave the same way on a scan measured in metres, in
        // centimetres, or in whatever arbitrary scale a COLMAP reconstruction
        // happened to land in.
        struct OutlierPolicy {

            // The cloud's robust radius: this percentile of the distance from
            // each mean to the component-wise median centre. Both rules below
            // are measured against it.
            float radiusPercentile = 0.99f;

            // "Smear". Both conditions have to hold:
            //   max(scale) > sizeVsRadius * robustRadius       scene-scale
            //   max(scale) > sizeVsPeers  * P_size(max(scale)) peer-relative
            // The first says the splat is as big as the whole reconstruction;
            // the second says it is nothing like its neighbours. Requiring
            // both is what keeps a cloud of a few large blobs — a test
            // fixture, a coarse proxy — from being mistaken for a scan.
            float sizePercentile = 0.99f;
            float sizeVsPeers = 8.f;
            float sizeVsRadius = 1.f;

            // "Stray": a point this far outside the reconstruction is a
            // reconstruction artefact, not part of the subject.
            float distanceVsRadius = 8.f;
        };

        // Drops what photogrammetry leaves behind: the handful of enormous
        // near-opaque smears a 3DGS optimiser parks across the sky to explain
        // the background, and the stray points scattered hundreds of units
        // outside the reconstruction. Returns how many were removed.
        //
        // NOT called by the loader. A raw cloud is what the file says, and a
        // caller may legitimately want it; this is an explicit, opt-in edit.
        //
        // THE RULE, with r = P_radiusPercentile(|mean - medianCentre|) — a
        // splat goes if EITHER
        //
        //   max(scale) > sizeVsRadius * r  AND  max(scale) > sizeVsPeers * P_size
        //   |mean - medianCentre| > distanceVsRadius * r
        //
        // Every threshold is a ratio of two lengths measured from the cloud
        // itself, which makes the rule scale-free, and all of them are
        // one-sided by construction: on a cloud with no tail the high
        // percentile is already close to the maximum, the factor puts the
        // threshold above it, and NOTHING is removed. A guard that fires on
        // clean input is a bug, and SplatData_test pins that it does not.
        //
        // Deterministic: percentiles are exact order statistics of the whole
        // cloud (std::nth_element on a copy), no sampling and no RNG.
        // Survivors keep their relative order, and `extras` and `sh` are
        // compacted alongside, so the cloud stays valid().
        //
        // Conservative by design, and measured that way. On the 5.0M-splat
        // Sanctuaire Sainte-Anne-de-Beaupré scan the defaults remove ~0.02%,
        // the sky stops being washed over by a dark smear, and the town below
        // the horizon is left alone. Loosening sizeVsRadius towards 0.06 —
        // eight times the 99th percentile of splat size, which sounds
        // reasonable and is not — starts deleting the far shore across the
        // river, because on a scan the distant background genuinely IS a
        // handful of enormous splats. Scene-relative size is the signal that
        // separates the two; peer-relative size on its own is not.
        // Two overloads rather than a defaulted argument: `= {}` on a nested
        // type needs that type's default member initializers complete while
        // the enclosing class still isn't, which GCC rejects.
        size_t removeOutliers();

        size_t removeOutliers(const OutlierPolicy& policy);

        // Reorders storage into Morton (Z-order) sequence, so that splats which
        // are neighbours in space become neighbours in memory. Returns the
        // permutation that was applied, as NEW INDEX -> OLD INDEX: after the
        // call, splat i is the splat that used to be at perm[i], which is the
        // direction a caller needs to remap an external per-splat array of its
        // own (remapped[i] = mine[perm[i]]). The identity permutation is
        // returned when nothing moved.
        //
        // WHY, since the render order is a per-frame depth sort and not this:
        // the shader fetches per-splat data by SORTED index while storage is
        // file order, so a depth slab through the scene walks the data textures
        // in a random permutation and the texture cache dies. Morton order does
        // not change what is drawn or in what order — it changes where the data
        // for consecutive draws LIVES, and a slab through a real surface then
        // hits contiguous runs instead of scattered texels.
        //
        // NOT called by the loader, for the same reason removeOutliers is not:
        // a raw cloud is what the file says. The one visible side effect is
        // that the depth sort's tie-break becomes Morton order instead of file
        // order, which is strictly the better tie-break.
        //
        // THE KEY. Each axis is quantised to 10 bits (1024 cells) over a
        // ROBUST interval — [P(1-p), P(p)] of that axis's own coordinates, not
        // its min/max. This is the outlier lesson again: one stray 1000 units
        // out would otherwise stretch the grid until the whole subject shares a
        // handful of cells and the reorder does nothing. Coordinates outside
        // the interval clamp into the edge cells, the same doctrine as the
        // depth sort's clamp — an outlier gets a worse ordering, never an
        // undefined one — and clamping also keeps the strays grouped together
        // at the ends, which is what you want anyway. A non-finite coordinate
        // lands in cell 0 without ever reaching a float-to-integer conversion.
        //
        // The three cell indices interleave x -> y -> z, x MOST significant
        // within each 3-bit group: key bit 3i+2 is x bit i, 3i+1 is y bit i,
        // 3i is z bit i, for a 30-bit key in a uint32. Read from the top, the
        // key is x, y, z, x, y, z, ...; at one bit per axis it is exactly the
        // lexicographic order of (x, y, z), which is what SplatData_test hand
        // computes over the corners of a cube.
        //
        // Deterministic and repeat-callable: exact order statistics for the
        // bounds (no sampling, no RNG) and a STABLE sort by key, so splats
        // sharing a cell keep their relative order. A second call is therefore
        // a no-op that returns the identity — SplatData_test pins that.
        //
        // Every array moves as one tuple: means, scales, rotations, opacities,
        // the SH block (coeffCount() * 3 floats) and every `extras` array. The
        // permutation is applied in place by cycle-following rather than by
        // gathering into fresh vectors: at five million splats the degree-3 SH
        // block alone is near a gigabyte, and a gather would want a second copy
        // of it. An array whose length does not match count() is left alone
        // instead of being walked off the end; on a validate()-clean cloud
        // there are none.
        //
        // One-time cost at load. std::stable_sort at five million splats is
        // acceptable there; it is deliberately not parallelised, because
        // Parallel.hpp's par policy is MSVC-only and silently serial elsewhere,
        // which would make the cost a platform surprise rather than a constant.
        std::vector<std::uint32_t> reorderMorton(float boundsPercentile = 0.999f);

        // True when every array length agrees and the degree is in range.
        // `why` (optional) receives a description of the first problem found.
        [[nodiscard]] bool validate(std::string* why = nullptr) const;
    };

    // Deterministic, seeded, file-free splat clouds. This is the walking
    // skeleton the renderer was built against and a permanent test fixture:
    // it can produce the awkward cases on demand (near-zero scale, near-zero
    // opacity, extreme anisotropy, unnormalised quaternions) which is what
    // makes it useful beyond "some blobs appear".
    //
    // Determinism is not std::mt19937 + std::uniform_real_distribution — the
    // distribution is free to differ between standard libraries. Draws come
    // from math::Rng (explicit fixed-point conversion, multi-draws sequenced
    // through named locals), so the same seed gives bit-identical output on
    // every platform.
    class SplatGenerator {

    public:
        struct Options {

            size_t count = 512;
            int shDegree = 0;
            unsigned int seed = 1337u;

            // Means are drawn uniformly from the box [-extent/2, +extent/2].
            Vector3 extent{2.f, 2.f, 2.f};

            // Linear scale range for the *smallest* axis of a splat; the other
            // two axes are stretched by up to `anisotropy`.
            float minScale = 0.02f;
            float maxScale = 0.10f;
            float anisotropy = 4.f;

            float minOpacity = 0.30f;
            float maxOpacity = 1.00f;

            // Amplitude of the randomised higher-order SH bands, relative to
            // the DC term. 0 gives view-independent colour at any degree.
            float higherOrderAmplitude = 0.35f;

            // Emit deliberately unnormalised quaternions (length 0.3 - 3),
            // the way an INRIA PLY stores them. Whoever consumes the data has
            // to normalise; SplatCloud and SplatLoader both do.
            bool unnormalizedRotations = false;

            // Salt the cloud with the cases that break naive shaders:
            // every 37th splat gets zero scale, every 53rd near-zero opacity.
            bool includeDegenerates = false;
        };

        // Two overloads for the same reason as SplatData::removeOutliers.
        [[nodiscard]] static SplatData generate();

        [[nodiscard]] static SplatData generate(const Options& options);
    };

}// namespace threepp

#endif//THREEPP_SPLATDATA_HPP
