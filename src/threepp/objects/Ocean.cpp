#include "threepp/objects/Ocean.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"

#include "threepp/objects/OceanLod.hpp"
#include "threepp/renderers/vulkan/EnvKnob.hpp"
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

        const unsigned int resX = std::max<unsigned int>(2u, options.resolution);
        // Default row count keeps grid cells square-ish across the rectangle.
        const unsigned int resZ = options.resolutionZ > 0
                ? std::max<unsigned int>(2u, options.resolutionZ)
                : std::max<unsigned int>(2u, static_cast<unsigned int>(
                          std::lround(double(resX) * double(sizeZ) / double(sizeX))));

        // Mesh density is decoupled from the FFT field: the renderer samples the
        // height texture via normalised UVs, so the plane can be any tessellation.
        // A skirt adds rings on every side; the plane keeps the inner grid's cell size
        // (the shader's normal stencil derives from it), the renderer places the rings.
        // the extended ocean's skirt and CDLOD mesh: neither with its master switch off (upstream's plane)
        const bool oceanExt = vulkan::env::flag("THREEPP_OCEAN_EXTENDED", true);
        const bool useCdlod = oceanExt && options.cdlod;
        const unsigned int ring = oceanExt ? options.skirtRings : 0u;
        std::shared_ptr<BufferGeometry> geo;
        if (!useCdlod) {
            geo = PlaneGeometry::create(sizeX * float(resX - 1u + 2u * ring) / float(resX - 1u),
                                         sizeZ * float(resZ - 1u + 2u * ring) / float(resZ - 1u),
                                         resX - 1u + 2u * ring, resZ - 1u + 2u * ring);
            geo->rotateX(-math::PI / 2.0f);
        }

        std::shared_ptr<BufferGeometry> mesh = geo;
        if (useCdlod) {
            using namespace oceanlod;
            mesh = BufferGeometry::create();
            std::vector<float> pos(vertices*3), norm(vertices*3), uv(vertices*2);
            std::vector<unsigned int> indices;
            indices.reserve(slots*cells*cells*6);
            for (unsigned p=0; p<slots; ++p) {
                for (unsigned z=0; z<side; ++z) for (unsigned x=0; x<side; ++x) {
                    unsigned i=p*side*side+z*side+x;
                    pos[3*i]=float(x); pos[3*i+2]=float(z); norm[3*i+1]=1.f;
                    if (x<cells && z<cells) {
                        unsigned a=i,b=i+side,c=b+1,d=i+1;
                        indices.insert(indices.end(),{a,b,d,b,c,d});
                    }
                }
            }
            mesh->setAttribute("position",FloatBufferAttribute::create(pos,3));
            mesh->setAttribute("normal",FloatBufferAttribute::create(norm,3));
            mesh->setAttribute("uv",FloatBufferAttribute::create(uv,2));
            mesh->setIndex(indices);
            // The patch template's own box is a few metres at the origin, but the surface it draws covers the warped
            // grid's extent and more. Hand bounds-driven consumers (the probe GI grid's fit to the scene) the box the
            // warped plane has, so the same scene gets the same probe grid under either mesh: with the template's box
            // the grid shrank onto the structures, and every non-water surface came out 2.5-10 luma darker by day.
            const float bx = 0.5f * sizeX * float(resX - 1u + 2u * ring) / float(resX - 1u);
            const float bz = 0.5f * sizeZ * float(resZ - 1u + 2u * ring) / float(resZ - 1u);
            mesh->boundingBox = Box3(Vector3(-bx, 0.f, -bz), Vector3(bx, 0.f, bz));
        }
        auto ocean = std::make_shared<Ocean>(mesh, makeOceanMaterial(pond, fjord));
        ocean->cdlod = useCdlod;
        ocean->halfExtent_ = scaleRef * 0.5f;
        // Non-square grids can't be derived from sqrt(vertexCount) — hand the
        // renderer the actual topology.
        ocean->gridWidth = resX + 2u * ring;
        ocean->gridDepth = resZ + 2u * ring;
        ocean->skirt.rings = ring;
        ocean->skirt.reach = options.skirtReach;
        if (useCdlod) {
            ocean->gridWidth=oceanlod::side;
            ocean->gridDepth=oceanlod::side*oceanlod::slots;
            ocean->skirt.rings=0;
            ocean->rtProxyCells=0;
            ocean->frustumCulled=false;
        }

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
        if (cdlod) return; // CDLOD owns allocation and preserves per-vertex motion history.
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
