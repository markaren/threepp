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
//   rotation  unit quaternion, threepp order (x, y, z, w)
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

    }// namespace splats

    // Struct-of-arrays. All the per-splat vectors are the same length except
    // `sh`, which is count() * coeffCount() * 3.
    struct SplatData {

        std::vector<Vector3> means;
        std::vector<Vector3> scales;       // linear, non-negative
        std::vector<Quaternion> rotations; // unit
        std::vector<float> opacities;      // [0, 1]
        std::vector<float> sh;             // coefficient-major, see header comment

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
    // distribution is free to differ between standard libraries. It is a
    // fixed xorshift32 with an explicit fixed-point conversion, so the same
    // seed gives bit-identical output on every platform.
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

        [[nodiscard]] static SplatData generate(const Options& options = {});
    };

}// namespace threepp

#endif//THREEPP_SPLATDATA_HPP
