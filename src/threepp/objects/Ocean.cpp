#include "threepp/objects/Ocean.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>

namespace threepp {

    namespace {

        // The water material recipe — a thin-shell transmissive surface that the
        // path tracer / deferred renderer refract through with Beer-Lambert
        // absorption. Lifted from examples/vulkan/vulkan_ocean.cpp so the tuned
        // values now live in one first-party place.
        std::shared_ptr<MeshPhysicalMaterial> makeOceanMaterial(bool pond, bool fjord) {
            auto mat = MeshPhysicalMaterial::create();
            // Pure water has no diffuse pigment — the blue comes from
            // Beer-Lambert absorption through the medium, not albedo.
            mat->color = Color::white;
            // Small roughness stands in for the sub-pixel chop the FFT can't
            // resolve; broadens each highlight over a few pixels so it converges
            // fast under TAA instead of sparkling.
            mat->roughness = 0.04f;
            mat->metalness = 0.0f;
            mat->setIor(1.33f);
            mat->transmission = 1.0f;
            // doubleSided + thinWalled opt this single FFT-displaced plane into
            // the renderer's thin-shell transmission BSDF: every crossing applies
            // Beer-Lambert for `thickness` metres of in-medium depth, instead of
            // using the (much longer) actual ray distance through the water column
            // which would saturate to near-black.
            mat->side = Side::Double;
            mat->thickness = 2.0f;
            mat->thinWalled = true;
            mat->attenuationColor = Color(0.10f, 0.45f, 0.55f);
            mat->attenuationDistance = 3.0f;
            mat->clearcoat = 0.1f;
            // Pond/lake regime: the tropical-ocean murk above (3 m visibility,
            // red almost fully absorbed) renders a metre-deep pond as opaque
            // teal — the shallow-water transmission never gets to show the
            // bottom. Ponds get NATURAL POND water instead: green-brown,
            // ~2.5 m visibility — the bottom reads through a clearly-present
            // water medium rather than window glass (a first cut at 6 m
            // visibility looked like a swimming pool). Override on the
            // material afterwards for a clear alpine tarn or a black bog.
            if (pond) {
                mat->attenuationColor = Color(0.22f, 0.45f, 0.38f);
                mat->attenuationDistance = 2.5f;
                // Body-veil proxy depth. The deep-water veil brightness goes as
                // attenuationColor^(2·thickness/attenuationDistance): with the
                // ocean's 2 m proxy a murky (small-attDist) pond drives the
                // veil to BLACK instead of green. Sub-metre proxy keeps the
                // veil ≈ one attenuation length of colour — green murk, not tar.
                mat->thickness = 0.8f;
            }
            // Glacial fjord: the turquoise is SCATTERED light, so this recipe
            // brings absorption back toward physical (red gone in ~1 m, blue-
            // green carried) and lets the in-scatter make it bright. Getting
            // the same colour out of absorption alone needed attenuationDistance
            // stretched to 4 m — clear-open-ocean clarity for water you cannot
            // see a metre into. scatterDistance 5 m is a mean free path in the
            // range measured for turbid glacial melt; scatterColor is the
            // albedo weight, blue-green because rock flour scatters those
            // wavelengths back before the water absorbs them.
            if (fjord) {
                mat->attenuationColor = Color(0.10f, 0.50f, 0.48f);
                mat->attenuationDistance = 1.5f;
                mat->scatterColor = Color(0.30f, 0.62f, 0.60f);
                mat->scatterDistance = 5.0f;
            }
            return mat;
        }

    }// namespace

    Ocean::Ocean(const std::shared_ptr<BufferGeometry>& geometry,
                 const std::shared_ptr<Material>& material)
        : DisplacedMesh(geometry, material) {}

    std::string Ocean::type() const {
        return "Ocean";
    }

    std::shared_ptr<Ocean> Ocean::create() {
        return create(Options{});
    }

    std::shared_ptr<Ocean> Ocean::create(const Options& options) {
        // Rectangle support: sizeZ 0 = square. Everything SCALE-derived (the
        // cascade-0 tile, the auto tile trio, foam amount, the freshwater
        // material gate) keys on the LARGER extent — a 900 × 3200 fjord
        // channel is a 3200 m-class water body that happens to waste no
        // vertices on land.
        const float sizeX = options.size;
        const float sizeZ = options.sizeZ > 0.f ? options.sizeZ : options.size;
        const float scaleRef = std::max(sizeX, sizeZ);
        const bool pond = options.look == Look::Auto ? scaleRef < 100.f
                                                     : options.look == Look::Pond;
        const bool fjord = options.look == Look::Fjord;

        const unsigned int resX = std::max<unsigned int>(2u, options.resolution);
        // Default row count keeps grid cells square-ish across the rectangle.
        const unsigned int resZ = options.resolutionZ > 0
                ? std::max<unsigned int>(2u, options.resolutionZ)
                : std::max<unsigned int>(2u, static_cast<unsigned int>(
                          std::lround(double(resX) * double(sizeZ) / double(sizeX))));

        // Mesh density is decoupled from the FFT field: the renderer samples the
        // height texture via normalised UVs, so the plane can be any tessellation.
        auto geo = PlaneGeometry::create(sizeX, sizeZ, resX - 1u, resZ - 1u);
        geo->rotateX(-math::PI / 2.0f);

        auto ocean = std::make_shared<Ocean>(geo, makeOceanMaterial(pond, fjord));
        ocean->halfExtent_ = scaleRef * 0.5f;
        // Non-square grids can't be derived from sqrt(vertexCount) — hand the
        // renderer the actual topology.
        ocean->gridWidth = resX;
        ocean->gridDepth = resZ;

        // Resolve the auto (−1) tile sentinels: proportional to the water
        // body's scale, capped at the 1000 m reference trio — see
        // Options::tileSize1 for the why.
        const float tile1 = options.tileSize1 < 0.f
                ? std::min(scaleRef * (127.0f / 1000.f), 127.0f)
                : options.tileSize1;
        const float tile2 = options.tileSize2 < 0.f
                ? std::min(scaleRef * (9.3f / 1000.f), 9.3f)
                : options.tileSize2;

        auto& p = ocean->params;
        p.tileSize0 = scaleRef;
        p.tileSize1 = tile1;
        p.tileSize2 = tile2;
        p.windTheta = options.windTheta;
        p.windSpeed = options.windSpeed;
        p.fetch = options.fetch;
        p.waveScale = options.waveScale;
        p.choppiness = options.choppiness;
        // Natural whitecaps fade out with the water body's scale — see
        // Params::foamAmount. Overridable any time via params.
        p.foamAmount = std::min(scaleRef / 300.0f, 1.0f);

        // Per-cascade FFT resolution, derived from the band each cascade
        // actually carries. The renderer band-passes cascade 0 to
        // λ ∈ [tileSize1, size] and cascade 1 to λ ∈ [tileSize2, tileSize1],
        // so the resolution only has to sample the SHORTEST wavelength in the
        // band well (~10 texels per wavelength keeps the bicubic B-spline
        // reconstruction within ~2% amplitude at the band edge). Running them
        // at fftSize like before computed 1024² FFTs whose spectra held ~10
        // active modes per axis — ~50× wasted texel work per frame.
        // The last enabled cascade has an OPEN band (no kMax) and keeps the
        // resolution-driven detail cap, so it stays at fftSize / 2.
        const uint32_t half = std::max<uint32_t>(1u, options.fftSize / 2u);
        auto bandRes = [&](float tile, float lambdaMin, uint32_t cap) {
            uint32_t n = 64u;
            const float want = 10.f * tile / std::max(lambdaMin, 1e-3f);
            while (float(n) < want && n < cap) n *= 2u;
            return n;
        };
        if (tile1 > 0.f) {
            p.textureSize0 = bandRes(scaleRef, tile1, options.fftSize);
            if (tile2 > 0.f) {
                p.textureSize1 = bandRes(tile1, tile2, half);
                p.textureSize2 = half;
            } else {
                p.textureSize1 = half;// open band — full detail resolution
                p.textureSize2 = half;
            }
        } else {
            // Single-cascade setup: cascade 0 is the open band.
            p.textureSize0 = options.fftSize;
            p.textureSize1 = half;
            p.textureSize2 = half;
        }

        return ocean;
    }

    void Ocean::warpToward(float worldX, float worldZ, float coefA) {
        warp.centerX = worldX;
        warp.centerZ = worldZ;
        warp.halfRange = halfExtent_;
        warp.coefA = coefA;
    }

    void Ocean::setWind(float speed, float theta) {
        params.windSpeed = speed;
        params.windTheta = theta;
    }

}// namespace threepp
