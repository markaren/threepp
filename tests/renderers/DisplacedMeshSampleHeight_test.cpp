// DisplacedMesh::sampleHeight against an analytic Gerstner wave written into
// the CPU cascade fields. The rendered surface point with rest position q sits
// at q + D(q), so the height over a world XZ is H(q) for the q whose displaced
// image is that XZ. sampleHeight used to return H(worldXZ), off by the local
// chop displacement. CPU only: no renderer or GPU involved.

#include <catch2/catch_test_macros.hpp>

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"

#include <algorithm>
#include <cmath>
#include <numbers>

using namespace threepp;

namespace {

    constexpr uint32_t kDim = 64;
    constexpr float kTile = 64.f;          // 1 m texels
    constexpr double kWavelength = 32.0;   // 32 texels per wavelength
    constexpr double kK = 2.0 * std::numbers::pi / kWavelength;

    // Response of the uniform cubic B-spline (cascades 0 and 1 are B-spline
    // filtered, not interpolated) to a sampled sinusoid of this frequency.
    // The alias terms are below a micrometre at 32 texels per wavelength.
    double bsplineGain() {
        const double x = std::numbers::pi / kWavelength;
        return std::pow(std::sin(x) / x, 4.0);
    }

    // Gerstner wave along the cascade's domain x: height A cos(kx),
    // horizontal displacement -A sin(kx). Stored unnormalized (x tileSize),
    // as the IFFT writes it.
    void fillGerstner(DisplacedMesh::CascadeField& cf, double amplitude) {
        cf.dim = kDim;
        cf.tileSize = kTile;
        cf.data.assign(size_t(kDim) * kDim * 2, 0.f);
        cf.disp.assign(size_t(kDim) * kDim * 2, 0.f);
        for (uint32_t z = 0; z < kDim; ++z) {
            for (uint32_t x = 0; x < kDim; ++x) {
                const double px = (double(x) + 0.5) / kDim * kTile;
                const size_t idx = (size_t(z) * kDim + x) * 2;
                cf.data[idx] = float(amplitude * std::cos(kK * px) * kTile);
                cf.disp[idx] = float(-amplitude * std::sin(kK * px) * kTile);
            }
        }
    }

    // Exact height over domain coordinate x: solve q - c a sin(kq) = x by
    // bisection (monotone while c a k < 1), then a cos(kq).
    double gerstnerHeightAt(double x, double a, double choppiness) {
        const double ca = choppiness * a;
        double lo = x - ca - 1.0;
        double hi = x + ca + 1.0;
        for (int i = 0; i < 200; ++i) {
            const double mid = 0.5 * (lo + hi);
            if (mid - ca * std::sin(kK * mid) < x) lo = mid;
            else hi = mid;
        }
        return a * std::cos(kK * 0.5 * (lo + hi));
    }

    std::shared_ptr<DisplacedMesh> makeMesh() {
        return DisplacedMesh::create(PlaneGeometry::create(1, 1),
                                     MeshBasicMaterial::create());
    }

}// namespace

TEST_CASE("sampleHeight without choppiness is the height field at the query") {
    auto mesh = makeMesh();
    mesh->params.choppiness = 0.f;
    mesh->params.waveScale = 1.f;
    const double a = 3.0;
    fillGerstner(mesh->heightFields[0], a);
    const double gain = bsplineGain();

    double maxErr = 0.0;
    for (double x = -50.0; x <= 50.0; x += 0.37) {
        const double expected = a * gain * std::cos(kK * x);
        maxErr = std::max(maxErr, std::abs(mesh->sampleHeight(float(x), 7.f) - expected));
    }
    CHECK(maxErr < 1e-3);
}

TEST_CASE("sampleHeight inverts the horizontal displacement") {
    auto mesh = makeMesh();
    const double choppiness = 1.0;
    const double waveScale = 1.5;
    mesh->params.choppiness = float(choppiness);
    mesh->params.waveScale = float(waveScale);
    const double a = 3.0;// c a k = 0.59: steep, but no fold
    fillGerstner(mesh->heightFields[0], a);
    const double aEff = a * bsplineGain();

    double maxErr = 0.0;
    double maxUninverted = 0.0;
    for (double x = -50.0; x <= 50.0; x += 0.37) {
        const double expected = waveScale * gerstnerHeightAt(x, aEff, choppiness);
        maxErr = std::max(maxErr, std::abs(mesh->sampleHeight(float(x), -3.f) - expected));
        maxUninverted = std::max(maxUninverted,
                                 std::abs(waveScale * aEff * std::cos(kK * x) - expected));
    }
    CHECK(maxErr < 5e-3);
    // The height at the query's own rest position is far off on this sea, so
    // the check above cannot pass without the inversion.
    CHECK(maxUninverted > 1.0);
}

TEST_CASE("sampleHeight inverts cascade 1 in its rotated domain") {
    auto mesh = makeMesh();
    const double choppiness = 0.8;
    mesh->params.choppiness = float(choppiness);
    mesh->params.waveScale = 1.f;
    const double a = 2.5;
    fillGerstner(mesh->heightFields[1], a);
    const double aEff = a * bsplineGain();

    // World -> cascade-1 domain, as oceanC1Domain(). In the domain the wave
    // is the plain Gerstner wave, so the exact height depends only on the
    // domain x of the query.
    const double c = kOceanCascade1RotCos;
    const double s = kOceanCascade1RotSin;

    double maxErr = 0.0;
    for (double wx = -40.0; wx <= 40.0; wx += 1.3) {
        for (double wz = -40.0; wz <= 40.0; wz += 1.7) {
            const double domainX = c * wx - s * wz;
            const double expected = gerstnerHeightAt(domainX, aEff, choppiness);
            maxErr = std::max(maxErr, std::abs(mesh->sampleHeight(float(wx), float(wz)) - expected));
            // Only cascade 1 holds data, so selecting it alone is the same query.
            CHECK(mesh->sampleHeight(float(wx), float(wz), 0b010u) ==
                  mesh->sampleHeight(float(wx), float(wz)));
        }
    }
    CHECK(maxErr < 5e-3);
    // Masking cascade 1 out leaves no field at all.
    CHECK(mesh->sampleHeight(3.f, 4.f, 0b101u) == 0.f);
}
