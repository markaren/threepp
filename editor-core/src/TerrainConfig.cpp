
#include "threepp/extras/editor/TerrainConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;
using threepp::terrain::ErosionType;
using threepp::terrain::Falloff;
using threepp::terrain::NoiseType;
using threepp::terrain::TerrainParams;

namespace {

    // Every knob, once. The three lists below are walked by both encode() and
    // decode(), so a parameter cannot be saved and not loaded (or the reverse) —
    // the failure mode of forty hand-written if-chains.
#define THREEPP_TERRAIN_FLOATS(X)     \
    X(worldSize)                      \
    X(featureScale)                   \
    X(lacunarity)                     \
    X(gain)                           \
    X(amplitude)                      \
    X(warp)                           \
    X(ridgeSharpness)                 \
    X(heightExponent)                 \
    X(falloffStart)                   \
    X(inertia)                        \
    X(sedimentCapacity)               \
    X(minSlope)                       \
    X(erodeSpeed)                     \
    X(depositSpeed)                   \
    X(evaporation)                    \
    X(gravity)                        \
    X(talusAngle)                     \
    X(thermalRate)                    \
    X(snowLine)                       \
    X(snowNoiseAmp)                   \
    X(snowSlopeMax)                   \
    X(slopeGrassMax)                  \
    X(slopeRockMin)                   \
    X(bandEdge)                       \
    X(aoStrength)                     \
    X(aoMax)                          \
    X(aoCurvScale)                    \
    X(aoLo)                           \
    X(aoHi)

#define THREEPP_TERRAIN_INTS(X)       \
    X(resolution)                     \
    X(octaves)                        \
    X(terraces)                       \
    X(droplets)                       \
    X(dropletLifetime)                \
    X(erosionRadius)                  \
    X(thermalIterations)

#define THREEPP_TERRAIN_COLORS(X)     \
    X(rockColor)                      \
    X(grassColor)                     \
    X(screeColor)                     \
    X(snowColor)

    constexpr float kEditorWorldSize = 160.f;

    // Seeds come off std::random_device, so more than half of them are above
    // INT_MAX and codec::toInt (std::stoi) throws out_of_range on those — which
    // silently fell back to the default seed and made the stored config
    // describe a DIFFERENT landscape from the one in the file. Delta recovery
    // subtracts base(config): get the seed wrong and the whole mesh reads as
    // sculpt.
    unsigned int toUInt(std::string_view text, unsigned int fallback) {

        try {
            return static_cast<unsigned int>(std::stoul(std::string(text)));
        } catch (...) {
            return fallback;
        }
    }

}// namespace


const char* TerrainConfig::presetLabel(int preset) {

    switch (preset) {
        case 0: return "Alpine";
        case 1: return "Rolling Hills";
        case 2: return "Desert Mesa";
        case 3: return "Volcanic";
        default: return "Alpine";
    }
}

void TerrainConfig::applyPreset(int preset) {

    // The preset owns the CHARACTER (noise stack, erosion, band colours); this
    // node owns its SCALE. terrain::applyPreset writes hero dimensions —
    // 1200–1600 m across, 300–600 m tall — and dropping those on an editor
    // scene replaces it with a mountain range. Keep worldSize, resolution and
    // seed, and carry featureScale/amplitude across by the same ratio so the
    // landscape looks like the preset, just smaller.
    const float keepWorld = params.worldSize;
    const int keepResolution = params.resolution;
    const unsigned int keepSeed = params.seed;

    terrain::applyPreset(preset, params);

    const float ratio = keepWorld / std::max(params.worldSize, 1e-3f);
    params.worldSize = keepWorld;
    params.featureScale *= ratio;
    params.amplitude *= ratio;
    params.resolution = keepResolution;
    params.seed = keepSeed;

    // A preset's erosion choice is a request, not a bake: the pass runs behind
    // the Generate button, never on the click that picked the preset.
    eroded = false;
}

TerrainConfig TerrainConfig::makeDefault() {

    TerrainConfig config;
    auto& p = config.params;

    // A CANVAS, not a hero mountain. Add Terrain gives you ground to put a
    // scene on and sculpt up from — a 160 m field with a couple of metres of
    // gentle undulation, on which a 1 m box, a robot or a vehicle sits the way
    // it would on a floor. The dramatic landscapes are one click away in the
    // preset combo; a ridged massif as the DEFAULT fits nothing.
    p.worldSize = kEditorWorldSize;
    // 0.6 m cells: fine enough that a brush stroke has something to shape,
    // coarse enough to re-bake on a slider release without a stall.
    p.resolution = 256;
    p.noiseType = NoiseType::fBm;
    p.featureScale = 55.f;
    p.octaves = 4;
    p.amplitude = 2.6f;
    p.warp = 0.15f;
    p.heightExponent = 1.f;
    // No radial falloff: a canvas runs flat to its edges. Radial is for the
    // island-and-plain look the presets ask for.
    p.falloff = Falloff::None;

    // Erosion is configured but NOT baked — Add Terrain has to be instant, and
    // there is nothing on a two-metre swell for a droplet to carve anyway.
    p.erosion = ErosionType::None;

    // Above 1.0 the snow band can never open: at this relief a snowline would
    // be costume. Grass to a generous slope, so the whole canvas reads as one
    // continuous ground rather than a patchwork of bands.
    p.snowLine = 1.2f;
    p.slopeGrassMax = 0.45f;
    p.slopeRockMin = 0.65f;

    config.eroded = false;
    return config;
}


std::string TerrainConfig::encode() const {

    std::string out;
    const auto add = [&out](const char* key, const std::string& value) {
        if (!out.empty()) out += ';';
        out += key;
        out += '=';
        out += value;
    };

    add("seed", std::to_string(params.seed));
#define X(name) add(#name, number(params.name));
    THREEPP_TERRAIN_FLOATS(X)
#undef X
#define X(name) add(#name, std::to_string(params.name));
    THREEPP_TERRAIN_INTS(X)
#undef X
#define X(name)                                     \
    add(#name "R", number(params.name[0]));         \
    add(#name "G", number(params.name[1]));         \
    add(#name "B", number(params.name[2]));
    THREEPP_TERRAIN_COLORS(X)
#undef X
    add("noiseType", std::to_string(static_cast<int>(params.noiseType)));
    add("falloff", std::to_string(static_cast<int>(params.falloff)));
    add("erosion", std::to_string(static_cast<int>(params.erosion)));
    add("eroded", eroded ? "1" : "0");
    return out;
}

std::optional<TerrainConfig> TerrainConfig::decode(const std::string& text) {

    // Missing keys keep the EDITOR defaults, not the generator's hero ones: a
    // document this editor wrote is the only thing that lands here.
    TerrainConfig config = makeDefault();

    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "seed") {
            config.params.seed = toUInt(value, config.params.seed);
            return;
        }
#define X(name)                                                     \
    if (key == #name) {                                             \
        config.params.name = toFloat(value, config.params.name);    \
        return;                                                     \
    }
        THREEPP_TERRAIN_FLOATS(X)
#undef X
#define X(name)                                                     \
    if (key == #name) {                                             \
        config.params.name = toInt(value, config.params.name);      \
        return;                                                     \
    }
        THREEPP_TERRAIN_INTS(X)
#undef X
#define X(name)                                                              \
    if (key == #name "R") {                                                  \
        config.params.name[0] = toFloat(value, config.params.name[0]);       \
        return;                                                              \
    }                                                                        \
    if (key == #name "G") {                                                  \
        config.params.name[1] = toFloat(value, config.params.name[1]);       \
        return;                                                              \
    }                                                                        \
    if (key == #name "B") {                                                  \
        config.params.name[2] = toFloat(value, config.params.name[2]);       \
        return;                                                              \
    }
        THREEPP_TERRAIN_COLORS(X)
#undef X
        if (key == "noiseType") {
            config.params.noiseType = static_cast<NoiseType>(
                    std::clamp(toInt(value, static_cast<int>(config.params.noiseType)), 0, 2));
        } else if (key == "falloff") {
            config.params.falloff = static_cast<Falloff>(
                    std::clamp(toInt(value, static_cast<int>(config.params.falloff)), 0, 1));
        } else if (key == "erosion") {
            config.params.erosion = static_cast<ErosionType>(
                    std::clamp(toInt(value, static_cast<int>(config.params.erosion)), 0, 3));
        } else if (key == "eroded") {
            config.eroded = toBool(value, config.eroded);
        }
        // Unknown keys ignored on purpose: a document written by a newer editor
        // still loads here.
    });

    return config;
}

std::optional<TerrainConfig> TerrainConfig::read(const Object3D& object) {

    return readEntry<TerrainConfig>(object, userDataKey);
}

void TerrainConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

bool TerrainConfig::isTerrain(const Object3D& object) {

    return hasEntry(object, userDataKey);
}


int TerrainConfig::dim() const {

    return std::max(params.resolution, 1) + 1;
}

TerrainConfig::Bake TerrainConfig::bake() const {

    Bake out;
    terrain::TerrainGenerator generator(params.seed);
    generator.buildField(params);
    if (eroded) generator.erode(params);

    out.dim = generator.dim();
    out.field = generator.getField();
    // clone() to a PLAIN BufferGeometry, deliberately. makeGeometry hands back a
    // PlaneGeometry, and the exporter serializes those parametrically — width,
    // height, segments — which would write a displaced heightfield out as a flat
    // sheet and lose every sculpt on save. The mesh is the truth here, so it has
    // to be a mesh the document stores as buffers.
    const auto plane = generator.makeGeometry(params);
    out.geometry = plane->clone();
    plane->dispose();
    out.heights = heightsOf(*out.geometry);
    out.albedo = generator.bakeSplatColors(params);
    return out;
}

std::vector<unsigned char> TerrainConfig::bakeAlbedo(const std::vector<float>& field) const {

    terrain::TerrainGenerator generator(params.seed);
    generator.setField(field);
    return generator.bakeSplatColors(params);
}


std::vector<float> TerrainConfig::heightsOf(const BufferGeometry& geometry) {

    std::vector<float> out;
    const auto* position = geometry.getAttribute<float>("position");
    if (!position) return out;
    const auto& array = position->array();
    const int count = position->count();
    out.resize(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) out[static_cast<size_t>(i)] = array[i * 3 + 1];
    return out;
}

void TerrainConfig::setHeights(BufferGeometry& geometry, const std::vector<float>& heights) {

    auto* position = geometry.getAttribute<float>("position");
    if (!position) return;
    auto& array = position->array();
    const auto count = std::min(static_cast<size_t>(position->count()), heights.size());
    for (size_t i = 0; i < count; ++i) array[i * 3 + 1] = heights[i];
    position->needsUpdate();
}

std::vector<float> TerrainConfig::fieldOf(const BufferGeometry& geometry,
                                          const std::vector<float>& heights) const {

    // Inverts displaceTo: Y = field·amplitude − baseSink(x,z).
    std::vector<float> out(heights.size(), 0.f);
    const auto* position = geometry.getAttribute<float>("position");
    if (!position) return out;
    const auto& array = position->array();
    const auto count = std::min(static_cast<size_t>(position->count()), heights.size());
    const float amplitude = std::max(params.amplitude, 1e-3f);
    for (size_t i = 0; i < count; ++i) {
        const float x = array[i * 3 + 0];
        const float z = array[i * 3 + 2];
        out[i] = (heights[i] + terrain::TerrainGenerator::baseSink(x, z, params)) / amplitude;
    }
    return out;
}

std::vector<float> TerrainConfig::resample(const std::vector<float>& src, int srcDim, int dstDim) {

    if (srcDim < 2 || dstDim < 2) return {};
    if (static_cast<size_t>(srcDim) * srcDim != src.size()) return {};
    if (srcDim == dstDim) return src;

    std::vector<float> out(static_cast<size_t>(dstDim) * dstDim, 0.f);
    const float scale = static_cast<float>(srcDim - 1) / static_cast<float>(dstDim - 1);
    for (int z = 0; z < dstDim; ++z) {
        const float sz = static_cast<float>(z) * scale;
        const int z0 = std::min(static_cast<int>(sz), srcDim - 1);
        const int z1 = std::min(z0 + 1, srcDim - 1);
        const float fz = sz - static_cast<float>(z0);
        for (int x = 0; x < dstDim; ++x) {
            const float sx = static_cast<float>(x) * scale;
            const int x0 = std::min(static_cast<int>(sx), srcDim - 1);
            const int x1 = std::min(x0 + 1, srcDim - 1);
            const float fx = sx - static_cast<float>(x0);
            const float a = src[static_cast<size_t>(z0) * srcDim + x0];
            const float b = src[static_cast<size_t>(z0) * srcDim + x1];
            const float c = src[static_cast<size_t>(z1) * srcDim + x0];
            const float d = src[static_cast<size_t>(z1) * srcDim + x1];
            out[static_cast<size_t>(z) * dstDim + x] =
                    (a + (b - a) * fx) * (1.f - fz) + (c + (d - c) * fx) * fz;
        }
    }
    return out;
}


std::shared_ptr<Texture> TerrainConfig::albedoTexture(const Object3D& object) {

    const auto* mesh = object.as<Mesh>();
    if (!mesh || !mesh->material()) return nullptr;
    auto* standard = mesh->material()->as<MeshStandardMaterial>();
    if (!standard) return nullptr;
    // Deliberately NOT narrowed to DataTexture: the exporter writes the baked
    // splat out as an embedded image and the loader brings it back as a plain
    // Texture, so a reloaded terrain's albedo is not the class the bake made.
    // What matters is that it is an 8-bit map of the lattice's dimensions.
    return standard->map;
}

void TerrainConfig::applyAlbedo(Object3D& object, const std::vector<unsigned char>& albedo, int dim) {

    auto* mesh = object.as<Mesh>();
    if (!mesh || dim < 2 || albedo.size() != static_cast<size_t>(dim) * dim * 4) return;
    auto* standard = mesh->material() ? mesh->material()->as<MeshStandardMaterial>() : nullptr;
    if (!standard) return;

    const auto& existing = standard->map;
    const bool sameSize = existing && existing->image().width() == static_cast<unsigned int>(dim) &&
                          existing->image().height() == static_cast<unsigned int>(dim) &&
                          !existing->image().isFloat() && !existing->image().isHalfFloat();
    if (sameSize) {
        // In place: the texture is the same GPU object, so no material rebuild
        // and no descriptor churn on the Vulkan side.
        existing->image().setData(ImageData{albedo});
        existing->needsUpdate();
        return;
    }

    auto texture = DataTexture::create(ImageData{albedo},
                                       static_cast<unsigned int>(dim),
                                       static_cast<unsigned int>(dim));
    texture->colorSpace = ColorSpace::sRGB;
    texture->magFilter = Filter::Linear;
    texture->minFilter = Filter::Linear;
    standard->map = texture;
    // The albedo IS the map now; a tint on top would double-darken the bands.
    standard->color = Color::white;
    standard->needsUpdate();
}


std::shared_ptr<BufferGeometry> TerrainConfig::rebuild(Object3D& object,
                                                       const TerrainConfig& before,
                                                       const TerrainConfig& after) {

    auto* mesh = object.as<Mesh>();
    if (!mesh) return nullptr;

    // 1. Recover the sculpt layer against the config that PRODUCED the mesh in
    //    front of us. Using `after` here would fold the parameter change itself
    //    into the "sculpt" and the edit would do nothing.
    std::vector<float> delta;
    int deltaDim = 0;
    if (const auto geometry = mesh->geometry()) {
        const auto current = heightsOf(*geometry);
        const auto base = before.bake();
        if (!current.empty() && current.size() == base.heights.size()) {
            delta.resize(current.size());
            for (size_t i = 0; i < current.size(); ++i) delta[i] = current[i] - base.heights[i];
            deltaDim = base.dim;
        }
    }

    // 2. Re-bake from the new config.
    auto bakeAfter = after.bake();

    // 3. Put the sculpt back on top, resampling it if the lattice moved.
    if (deltaDim >= 2 && !delta.empty()) {
        const auto fitted = (deltaDim == bakeAfter.dim) ? delta
                                                        : resample(delta, deltaDim, bakeAfter.dim);
        if (fitted.size() == bakeAfter.heights.size()) {
            const float amplitude = std::max(after.params.amplitude, 1e-3f);
            for (size_t i = 0; i < fitted.size(); ++i) {
                bakeAfter.heights[i] += fitted[i];
                // The splat reads the field, so the sculpt has to reach it too.
                bakeAfter.field[i] += fitted[i] / amplitude;
            }
            setHeights(*bakeAfter.geometry, bakeAfter.heights);
            bakeAfter.geometry->computeVertexNormals();
            if (auto* normal = bakeAfter.geometry->getAttribute<float>("normal")) normal->needsUpdate();
            bakeAfter.geometry->computeBoundingBox();
            bakeAfter.geometry->computeBoundingSphere();
            bakeAfter.albedo = after.bakeAlbedo(bakeAfter.field);
        }
    }

    // 4. Swap on the SAME node — uuid, material, transform and userData all
    //    stay, so selection, physics and the command stack keep their handles.
    const auto old = mesh->geometry();
    mesh->setGeometry(bakeAfter.geometry);
    if (old && old != bakeAfter.geometry) old->dispose();
    applyAlbedo(object, bakeAfter.albedo, bakeAfter.dim);
    after.write(object);

    return bakeAfter.geometry;
}
