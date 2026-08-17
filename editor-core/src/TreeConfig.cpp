
#include "threepp/extras/editor/TreeConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <any>
#include <array>
#include <map>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;
using namespace threepp::vegetation;

namespace {

    // The atlases are drawn at this resolution and tiled; bigger buys nothing
    // at the distance a tree is read from, and every distinct one is a base64
    // PNG in the saved document.
    constexpr unsigned int kTextureSize = 256;

    // Bark tiles several times around the trunk and stretches up it — the
    // generator's trunk UVs are unit-scaled, so the repeat is where the ridge
    // spacing actually gets set. Matches examples/extras/vegetation/tree_demo.
    constexpr float kBarkRepeatU = 3.f;
    constexpr float kBarkRepeatV = 0.5f;

    // ── enum tokens ──────────────────────────────────────────────────────
    // Tokens rather than ordinals, so a saved document reads as English and
    // reordering an enum cannot silently reinterpret one.

    struct Token {
        const char* text;
        int value;
    };

    template<class E, std::size_t N>
    E tokenTo(std::string_view text, const Token (&table)[N], E fallback) {

        for (const auto& entry : table) {
            if (text == entry.text) return static_cast<E>(entry.value);
        }
        return fallback;
    }

    template<class E, std::size_t N>
    const char* tokenOf(E value, const Token (&table)[N]) {

        for (const auto& entry : table) {
            if (static_cast<int>(value) == entry.value) return entry.text;
        }
        return table[0].text;
    }

    constexpr Token kCrownShapes[]{
            {"sphere", static_cast<int>(CrownShape::Sphere)},
            {"ellipsoid", static_cast<int>(CrownShape::Ellipsoid)},
            {"cone", static_cast<int>(CrownShape::Cone)},
            {"hemisphere", static_cast<int>(CrownShape::Hemisphere)},
            {"cylinder", static_cast<int>(CrownShape::Cylinder)}};

    constexpr Token kBranchingModes[]{
            {"colonise", static_cast<int>(BranchingMode::Colonise)},
            {"whorl", static_cast<int>(BranchingMode::Whorl)}};

    constexpr Token kBarkStyles[]{
            {"furrowed", static_cast<int>(BarkStyle::Furrowed)},
            {"plated", static_cast<int>(BarkStyle::Plated)},
            {"papery", static_cast<int>(BarkStyle::Papery)}};

    constexpr Token kLeafStyles[]{
            {"quad", static_cast<int>(LeafStyle::Quad)},
            {"cluster", static_cast<int>(LeafStyle::Cluster)},
            {"crossQuad", static_cast<int>(LeafStyle::CrossQuad)},
            {"blob", static_cast<int>(LeafStyle::Blob)},
            {"frond", static_cast<int>(LeafStyle::Frond)}};

    constexpr Token kLeafShapes[]{
            {"ovate", static_cast<int>(LeafShape::Ovate)},
            {"lobed", static_cast<int>(LeafShape::Lobed)},
            {"serrate", static_cast<int>(LeafShape::Serrate)},
            {"lanceolate", static_cast<int>(LeafShape::Lanceolate)}};

    // ── colour triples ───────────────────────────────────────────────────
    // One key per colour ("0.3,0.22,0.15") rather than three: the codec splits
    // on ';' and '=' only, so a comma rides through untouched, and the saved
    // line stays readable as a colour.

    std::string encodeColor(const std::array<float, 3>& color) {

        return number(color[0]) + "," + number(color[1]) + "," + number(color[2]);
    }

    void decodeColor(std::string_view text, std::array<float, 3>& out) {

        std::array<float, 3> parsed = out;
        std::size_t start = 0;
        for (int i = 0; i < 3; ++i) {
            if (start > text.size()) return;// short triple: keep the rest of the default
            const auto end = text.find(',', start);
            const auto token = text.substr(start, (end == std::string_view::npos ? text.size() : end) - start);
            parsed[i] = toFloat(token, parsed[i]);
            if (end == std::string_view::npos) {
                // Fewer than three components is a malformed value, not a
                // partial one — a half-applied colour is worse than the default.
                if (i != 2) return;
                break;
            }
            start = end + 1;
        }
        out = parsed;
    }

    // ── texture cache ────────────────────────────────────────────────────
    // Editor-thread state. Weak, so the atlases die with the last material
    // holding them rather than pinning every species a session ever previewed.
    std::map<std::string, std::weak_ptr<DataTexture>>& textureCache() {

        static std::map<std::string, std::weak_ptr<DataTexture>> cache;
        return cache;
    }

    template<class Make>
    std::shared_ptr<DataTexture> cached(const std::string& key, Make&& make) {

        auto& cache = textureCache();
        if (const auto it = cache.find(key); it != cache.end()) {
            if (auto alive = it->second.lock()) return alive;
        }
        auto texture = make();
        cache[key] = texture;
        return texture;
    }

}// namespace


const char* TreeConfig::presetLabel(int preset) {

    switch (preset) {
        case 0: return "Oak";
        case 1: return "Pine";
        case 2: return "Birch";
        case 3: return "Willow";
        default: break;
    }
    return "Custom";
}

const char* TreeConfig::partToken(Part part) {

    return part == Part::Trunk ? "trunk" : "leaves";
}

const char* TreeConfig::label(Part part) {

    return part == Part::Trunk ? "Trunk" : "Leaves";
}


std::string TreeConfig::encode() const {

    const auto& p = params;

    std::string out;
    const auto add = [&out](const char* key, const std::string& value) {
        if (!out.empty()) out += ';';
        out += key;
        out += '=';
        out += value;
    };
    const auto addF = [&add](const char* key, float value) { add(key, number(value)); };
    const auto addI = [&add](const char* key, int value) { add(key, std::to_string(value)); };

    addI("seed", static_cast<int>(p.seed));

    addF("trunkHeight", p.trunkHeight);
    addF("trunkRadius", p.trunkRadius);

    add("crownShape", tokenOf(p.crownShape, kCrownShapes));
    addF("crownRadiusX", p.crownRadiusX);
    addF("crownRadiusZ", p.crownRadiusZ);
    addF("crownHeight", p.crownHeight);

    addI("attractorCount", p.attractorCount);
    addF("influenceDistance", p.influenceDistance);
    addF("killDistance", p.killDistance);
    addF("segmentLength", p.segmentLength);
    addI("maxIterations", p.maxIterations);
    addF("randomness", p.randomness);
    addF("tropism", p.tropism);

    addF("radiusExponent", p.radiusExponent);
    addF("minBranchRadius", p.minBranchRadius);
    addI("radialSegments", p.radialSegments);

    add("branchingMode", tokenOf(p.branchingMode, kBranchingModes));
    addF("whorlSpacing", p.whorlSpacing);
    addI("branchesPerWhorl", p.branchesPerWhorl);
    addF("whorlJitter", p.whorlJitter);
    addF("branchDroop", p.branchDroop);
    addF("branchTipUpturn", p.branchTipUpturn);
    addF("crownProfileExponent", p.crownProfileExponent);
    addF("sideTwigDensity", p.sideTwigDensity);
    addF("branchLength", p.branchLength);

    addF("trunkLean", p.trunkLean);
    addF("trunkBend", p.trunkBend);
    addF("trunkTwist", p.trunkTwist);

    addF("barkBumpAmp", p.barkBumpAmp);
    addI("barkBumpLobes", p.barkBumpLobes);
    addF("barkBumpAmp2", p.barkBumpAmp2);
    addI("barkBumpLobes2", p.barkBumpLobes2);
    addF("rootFlareAsym", p.rootFlareAsym);
    add("barkStyle", tokenOf(p.barkStyle, kBarkStyles));

    add("leafStyle", tokenOf(p.leafStyle, kLeafStyles));
    add("leafShape", tokenOf(p.leafShape, kLeafShapes));
    addF("leafSize", p.leafSize);
    addF("leafDensity", p.leafDensity);
    addI("leavesPerCluster", p.leavesPerCluster);
    addF("leafSpread", p.leafSpread);
    addF("leafClumping", p.leafClumping);
    addF("foliageOcclusion", p.foliageOcclusion);

    add("barkColor", encodeColor(p.barkColor));
    add("leafColor", encodeColor(p.leafColor));

    return out;
}

std::optional<TreeConfig> TreeConfig::decode(const std::string& text) {

    TreeConfig config;
    auto& p = config.params;

    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "seed") {
            p.seed = static_cast<unsigned int>(toInt(value, static_cast<int>(p.seed)));
        } else if (key == "trunkHeight") {
            p.trunkHeight = toFloat(value, p.trunkHeight);
        } else if (key == "trunkRadius") {
            p.trunkRadius = toFloat(value, p.trunkRadius);
        } else if (key == "crownShape") {
            p.crownShape = tokenTo(value, kCrownShapes, p.crownShape);
        } else if (key == "crownRadiusX") {
            p.crownRadiusX = toFloat(value, p.crownRadiusX);
        } else if (key == "crownRadiusZ") {
            p.crownRadiusZ = toFloat(value, p.crownRadiusZ);
        } else if (key == "crownHeight") {
            p.crownHeight = toFloat(value, p.crownHeight);
        } else if (key == "attractorCount") {
            p.attractorCount = toInt(value, p.attractorCount);
        } else if (key == "influenceDistance") {
            p.influenceDistance = toFloat(value, p.influenceDistance);
        } else if (key == "killDistance") {
            p.killDistance = toFloat(value, p.killDistance);
        } else if (key == "segmentLength") {
            p.segmentLength = toFloat(value, p.segmentLength);
        } else if (key == "maxIterations") {
            p.maxIterations = toInt(value, p.maxIterations);
        } else if (key == "randomness") {
            p.randomness = toFloat(value, p.randomness);
        } else if (key == "tropism") {
            p.tropism = toFloat(value, p.tropism);
        } else if (key == "radiusExponent") {
            p.radiusExponent = toFloat(value, p.radiusExponent);
        } else if (key == "minBranchRadius") {
            p.minBranchRadius = toFloat(value, p.minBranchRadius);
        } else if (key == "radialSegments") {
            p.radialSegments = toInt(value, p.radialSegments);
        } else if (key == "branchingMode") {
            p.branchingMode = tokenTo(value, kBranchingModes, p.branchingMode);
        } else if (key == "whorlSpacing") {
            p.whorlSpacing = toFloat(value, p.whorlSpacing);
        } else if (key == "branchesPerWhorl") {
            p.branchesPerWhorl = toInt(value, p.branchesPerWhorl);
        } else if (key == "whorlJitter") {
            p.whorlJitter = toFloat(value, p.whorlJitter);
        } else if (key == "branchDroop") {
            p.branchDroop = toFloat(value, p.branchDroop);
        } else if (key == "branchTipUpturn") {
            p.branchTipUpturn = toFloat(value, p.branchTipUpturn);
        } else if (key == "crownProfileExponent") {
            p.crownProfileExponent = toFloat(value, p.crownProfileExponent);
        } else if (key == "sideTwigDensity") {
            p.sideTwigDensity = toFloat(value, p.sideTwigDensity);
        } else if (key == "branchLength") {
            p.branchLength = toFloat(value, p.branchLength);
        } else if (key == "trunkLean") {
            p.trunkLean = toFloat(value, p.trunkLean);
        } else if (key == "trunkBend") {
            p.trunkBend = toFloat(value, p.trunkBend);
        } else if (key == "trunkTwist") {
            p.trunkTwist = toFloat(value, p.trunkTwist);
        } else if (key == "barkBumpAmp") {
            p.barkBumpAmp = toFloat(value, p.barkBumpAmp);
        } else if (key == "barkBumpLobes") {
            p.barkBumpLobes = toInt(value, p.barkBumpLobes);
        } else if (key == "barkBumpAmp2") {
            p.barkBumpAmp2 = toFloat(value, p.barkBumpAmp2);
        } else if (key == "barkBumpLobes2") {
            p.barkBumpLobes2 = toInt(value, p.barkBumpLobes2);
        } else if (key == "rootFlareAsym") {
            p.rootFlareAsym = toFloat(value, p.rootFlareAsym);
        } else if (key == "barkStyle") {
            p.barkStyle = tokenTo(value, kBarkStyles, p.barkStyle);
        } else if (key == "leafStyle") {
            p.leafStyle = tokenTo(value, kLeafStyles, p.leafStyle);
        } else if (key == "leafShape") {
            p.leafShape = tokenTo(value, kLeafShapes, p.leafShape);
        } else if (key == "leafSize") {
            p.leafSize = toFloat(value, p.leafSize);
        } else if (key == "leafDensity") {
            p.leafDensity = toFloat(value, p.leafDensity);
        } else if (key == "leavesPerCluster") {
            p.leavesPerCluster = toInt(value, p.leavesPerCluster);
        } else if (key == "leafSpread") {
            p.leafSpread = toFloat(value, p.leafSpread);
        } else if (key == "leafClumping") {
            p.leafClumping = toFloat(value, p.leafClumping);
        } else if (key == "foliageOcclusion") {
            p.foliageOcclusion = toFloat(value, p.foliageOcclusion);
        } else if (key == "barkColor") {
            decodeColor(value, p.barkColor);
        } else if (key == "leafColor") {
            decodeColor(value, p.leafColor);
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    return config;
}

std::optional<TreeConfig> TreeConfig::read(const Object3D& object) {

    return readEntry<TreeConfig>(object, userDataKey);
}

void TreeConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void TreeConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

bool TreeConfig::isTree(const Object3D& object) {

    return hasEntry(object, userDataKey);
}


bool TreeConfig::isDerived(const Object3D& object) {

    return !readString(object, derivedKey).empty();
}

void TreeConfig::markDerived(Object3D& object, Part part) {

    object.userData[derivedKey] = std::string(partToken(part));
}

Object3D* TreeConfig::derivedPart(const Object3D& tree, Part part) {

    const std::string_view wanted = partToken(part);
    for (auto* child : tree.children) {
        if (readString(*child, derivedKey) == wanted) return child;
    }
    return nullptr;
}


TreeConfig::Geometries TreeConfig::build() const {

    // One skeleton, both skins — see the header note.
    TreeGenerator generator(params.seed);
    generator.buildSkeleton(params);

    return Geometries{generator.makeTrunkGeometry(params),
                      generator.makeLeafGeometry(params)};
}


std::string TreeConfig::textureKey() const {

    const auto& p = params;

    // Only what the atlases are actually drawn from. The leaf card is a
    // needle frond in Frond style and a broadleaf sprig otherwise, so the
    // style belongs here even though the SHAPE is ignored in the frond case
    // — the key may be finer than necessary, never coarser.
    std::string out = std::to_string(p.seed);
    out += '|';
    out += tokenOf(p.barkStyle, kBarkStyles);
    out += '|';
    out += encodeColor(p.barkColor);
    out += '|';
    out += tokenOf(p.leafStyle, kLeafStyles);
    out += '|';
    out += tokenOf(p.leafShape, kLeafShapes);
    out += '|';
    out += encodeColor(p.leafColor);
    return out;
}

TreeConfig::Textures TreeConfig::textures() const {

    const auto key = textureKey();
    const auto& p = params;

    Textures out;

    // The pair is drawn in one call, so both halves are claimed under the
    // albedo's key and the normal is looked up under its own.
    out.barkAlbedo = cached(key + "|barkAlbedo", [&] {
        auto pair = makeBarkTextures(kTextureSize, p.seed, p.barkColor, p.barkStyle);
        pair.first->repeat.set(kBarkRepeatU, kBarkRepeatV);
        return pair.first;
    });
    out.barkNormal = cached(key + "|barkNormal", [&] {
        auto pair = makeBarkTextures(kTextureSize, p.seed, p.barkColor, p.barkStyle);
        pair.second->repeat.set(kBarkRepeatU, kBarkRepeatV);
        return pair.second;
    });
    out.leaf = cached(key + "|leaf", [&] {
        // Conifer fronds want the elongated needle cutout; every other style
        // the round leaf-cluster atlas.
        return p.leafStyle == LeafStyle::Frond
                       ? makeNeedleFrondTexture(kTextureSize, p.seed, p.leafColor)
                       : makeLeafClusterTexture(kTextureSize, p.seed, p.leafColor, p.leafShape);
    });

    return out;
}

std::shared_ptr<MeshStandardMaterial> TreeConfig::makeBarkMaterial() const {

    auto material = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}
                    .color(Color::white)
                    .roughness(0.92f)
                    .metalness(0.f));

    const auto maps = textures();
    material->map = maps.barkAlbedo;
    material->normalMap = maps.barkNormal;
    return material;
}

std::shared_ptr<MeshStandardMaterial> TreeConfig::makeLeafMaterial() const {

    auto material = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}
                    .color(Color::white)
                    .roughness(0.85f)
                    .metalness(0.f));

    material->map = textures().leaf;
    // Below the antialiased margin of the thin leaflets the atlases are drawn
    // with — at 0.5 a mipped distant card loses whole leaves.
    material->alphaTest = 0.4f;
    material->side = Side::Double;
    // makeLeafGeometry bakes the canopy occlusion into the vertex colours;
    // without this the foliage is flat and has no interior depth.
    material->vertexColors = true;
    material->translucency = 0.45f;
    material->translucencyColor = Color(0.55f, 0.85f, 0.30f);
    return material;
}

void TreeConfig::applyTextures(Material* bark, Material* leaves, const TreeConfig& config) {

    const auto maps = config.textures();

    if (auto* standard = bark ? bark->as<MeshStandardMaterial>() : nullptr) {
        if (standard->map != maps.barkAlbedo || standard->normalMap != maps.barkNormal) {
            standard->map = maps.barkAlbedo;
            standard->normalMap = maps.barkNormal;
            standard->needsUpdate();
        }
    }
    if (auto* standard = leaves ? leaves->as<MeshStandardMaterial>() : nullptr) {
        if (standard->map != maps.leaf) {
            standard->map = maps.leaf;
            standard->needsUpdate();
        }
    }
}
