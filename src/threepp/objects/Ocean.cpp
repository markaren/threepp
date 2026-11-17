#include "threepp/objects/Ocean.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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
            // The renderers know the ocean by its mesh (the DisplacedMesh and its
            // foam attribute), not by this material: the Vulkan shade's water
            // branch and the path tracer's water interface both key on that.
            // `thickness` is the water body's proxy depth: a crossing absorbs over
            // `thickness` metres of water, not over the (much longer) ray distance
            // through the column, which would saturate to near-black.
            mat->side = Side::Double;
            mat->thickness = 2.0f;
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
                mat->scatterColor = Color(0.18f, 0.42f, 0.40f);// tuned by eye vs (0.30,0.62,0.60): that read milky/opaque
                mat->scatterDistance = 8.0f;    // 5 m over-saturated the body; 8 m keeps the wall reflection legible
            }
            return mat;
        }

    }// namespace

    Ocean::Ocean(const std::shared_ptr<BufferGeometry>& geometry,
                 const std::shared_ptr<Material>& material)
        : DisplacedMesh(geometry, material) {}

    const std::string& Ocean::type() const {
        static const std::string typeName = "Ocean";
        return typeName;
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

        // Resolve the auto (−1) tile sentinels: proportional to the water
        // body's scale, capped at the 1000 m reference trio — see
        // Options::tileSize1 for the why.
        const float tile1 = options.tileSize1 < 0.f
                ? std::min(scaleRef * (127.0f / 1000.f), 127.0f)
                : options.tileSize1;
        const float tile2 = options.tileSize2 < 0.f
                ? std::min(scaleRef * (9.3f / 1000.f), 9.3f)
                : options.tileSize2;

        // The patch mesh (DisplacedMesh::Lod). The renderer lays the patches
        // each frame and writes every vertex; what is set here is the topology
        // (each patch's 16 x 16 cells, wound as PlaneGeometry's after its
        // rotateX: row j runs toward -Z), and positions spread over the water's
        // extent so the bounds are the water's. A bounded water needs no more
        // slots than its finest patches (lodFinest: a small water's are finer)
        // can tile: over an extent of n patches the quadtree keeps at most
        // floor(n) + 2 a side, a patch that only touches the edge included
        // (a 7 m pond on 0.5 m patches: 16).
        const uint32_t cells = kLodCells, side = cells + 1u, perPatch = side * side;
        const float finestPatch = float(cells) * lodFinest(tile1, tile2);
        const uint32_t slots = options.horizon
                ? kLodSlots
                : uint32_t(std::min<double>(kLodSlots, (std::floor(sizeX / finestPatch) + 2.0) *
                                                               (std::floor(sizeZ / finestPatch) + 2.0)));
        const size_t vertexCount = size_t(slots) * perPatch;
        std::vector<float> positions(vertexCount * 3u, 0.f), normals(vertexCount * 3u, 0.f), uvs(vertexCount * 2u, 0.f);
        const uint32_t across = uint32_t(std::ceil(std::sqrt(double(slots))));
        for (uint32_t k = 0; k < slots; ++k) {
            const float px = -0.5f * sizeX + sizeX * (float(k % across) + 0.5f) / float(across);
            const float pz = -0.5f * sizeZ + sizeZ * (float(k / across) + 0.5f) / float(across);
            for (uint32_t v = 0; v < perPatch; ++v) {
                const size_t i = size_t(k) * perPatch + v;
                positions[3u * i + 0u] = std::clamp(px + (float(v % side) - 8.f) * sizeX / float(across * cells), -0.5f * sizeX, 0.5f * sizeX);
                positions[3u * i + 2u] = std::clamp(pz - (float(v / side) - 8.f) * sizeZ / float(across * cells), -0.5f * sizeZ, 0.5f * sizeZ);
                normals[3u * i + 1u] = 1.f;
            }
        }
        std::vector<unsigned int> indices;
        indices.reserve(size_t(slots) * cells * cells * 6u);
        for (uint32_t k = 0; k < slots; ++k) {
            const uint32_t base = k * perPatch;
            for (uint32_t j = 0; j < cells; ++j) {
                for (uint32_t i = 0; i < cells; ++i) {
                    const uint32_t a = base + j * side + i, b = base + (j + 1u) * side + i;
                    const uint32_t c = b + 1u, d = a + 1u;
                    indices.insert(indices.end(), {a, b, d, b, c, d});
                }
            }
        }
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        geo->setIndex(indices);

        auto ocean = std::make_shared<Ocean>(geo, makeOceanMaterial(pond, fjord));
        ocean->halfExtent_ = scaleRef * 0.5f;
        ocean->lod.slots = slots;
        ocean->lod.halfX = options.horizon ? 0.f : 0.5f * sizeX;
        ocean->lod.halfZ = options.horizon ? 0.f : 0.5f * sizeZ;
        // The patches follow the camera, wherever the bounds say they are.
        ocean->frustumCulled = false;

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

    void Ocean::setWeather(const Weather& weather) {
        params.windSpeed = weather.windSpeed;
        params.windTheta = weather.windTheta;
        params.fetch = weather.fetch;
        params.swellHeight = weather.swellHeight;
        params.swellPeriod = weather.swellPeriod;
        params.swellTheta = weather.swellTheta;
    }

}// namespace threepp
