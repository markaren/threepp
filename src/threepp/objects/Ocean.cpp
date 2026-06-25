#include "threepp/objects/Ocean.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>

namespace threepp {

    namespace {

        // The water material recipe — a thin-shell transmissive surface that the
        // path tracer / deferred renderer refract through with Beer-Lambert
        // absorption. Lifted from examples/vulkan/vulkan_ocean.cpp so the tuned
        // values now live in one first-party place.
        std::shared_ptr<MeshPhysicalMaterial> makeOceanMaterial() {
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
            return mat;
        }

    }// namespace

    Ocean::Ocean(const std::shared_ptr<BufferGeometry>& geometry,
                 const std::shared_ptr<Material>& material)
        : DisplacedMesh(geometry, material) {}

    std::string Ocean::type() const {
        return "Ocean";
    }

    std::shared_ptr<Ocean> Ocean::create(const Options& options) {
        const unsigned int res = std::max<unsigned int>(2u, options.resolution);
        // Mesh density is decoupled from the FFT field: the renderer samples the
        // height texture via normalised UVs, so the plane can be any tessellation.
        auto geo = PlaneGeometry::create(options.size, options.size,
                                         res - 1u, res - 1u);
        geo->rotateX(-math::PI / 2.0f);

        auto ocean = std::make_shared<Ocean>(geo, makeOceanMaterial());
        ocean->halfExtent_ = options.size * 0.5f;

        auto& p = ocean->params;
        p.tileSize0 = options.size;
        p.tileSize1 = options.tileSize1;
        p.tileSize2 = options.tileSize2;
        p.windTheta = options.windTheta;
        p.windSpeed = options.windSpeed;
        p.waveScale = options.waveScale;
        p.choppiness = options.choppiness;
        // Cascade-0 keeps the full FFT for macro-shape fidelity; the
        // shorter-wavelength cascades saturate well below it, so halve them.
        p.textureSize0 = options.fftSize;
        p.textureSize1 = std::max<uint32_t>(1u, options.fftSize / 2u);
        p.textureSize2 = std::max<uint32_t>(1u, options.fftSize / 2u);

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
