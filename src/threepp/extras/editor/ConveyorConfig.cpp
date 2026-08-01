
#include "threepp/extras/editor/ConveyorConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Fixed 6 decimals with trailing zeros trimmed — same rationale as
    // SplineConfig: stable across platforms and byte-identical for an unchanged
    // value, which keeps saved documents diff-clean.
    std::string number(float value) {

        char buffer[32];
        const int n = std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
        std::string out(buffer, buffer + (n > 0 ? n : 0));
        if (out.find('.') == std::string::npos) return out;
        while (!out.empty() && out.back() == '0') out.pop_back();
        if (!out.empty() && out.back() == '.') out.pop_back();
        return out.empty() ? "0" : out;
    }

    float toFloat(std::string_view text, float fallback) {

        try {
            return std::stof(std::string(text));
        } catch (...) {
            return fallback;
        }
    }

    int toInt(std::string_view text, int fallback) {

        try {
            return std::stoi(std::string(text));
        } catch (...) {
            return fallback;
        }
    }

    conveyor::SegKind segFrom(std::string_view text, conveyor::SegKind fallback) {

        if (text == "flat") return conveyor::SegKind::Flat;
        if (text == "rollers") return conveyor::SegKind::Rollers;
        if (text == "cleats") return conveyor::SegKind::Cleats;
        return fallback;
    }

    const char* segToken(conveyor::SegKind kind) {

        switch (kind) {
            case conveyor::SegKind::Flat: return "flat";
            case conveyor::SegKind::Rollers: return "rollers";
            case conveyor::SegKind::Cleats: return "cleats";
        }
        return "flat";
    }

    // Walk a flat `key=value;…` string, calling `apply(key, value)` per pair.
    template<class F>
    void parsePairs(const std::string& text, F&& apply) {

        std::size_t start = 0;
        while (start <= text.size()) {
            const auto end = text.find(';', start);
            const auto token = std::string_view(text).substr(
                    start, (end == std::string::npos ? text.size() : end) - start);
            const auto eq = token.find('=');
            if (eq != std::string_view::npos) {
                apply(token.substr(0, eq), token.substr(eq + 1));
            }
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }

}// namespace


// --- ConveyorWaypointConfig -------------------------------------------------

std::string ConveyorWaypointConfig::encode() const {

    std::string out;
    out += "radius=";
    out += number(cornerRadius);
    out += ";seg=";
    out += segToken(segKind);
    return out;
}

ConveyorWaypointConfig ConveyorWaypointConfig::decode(const std::string& text) {

    ConveyorWaypointConfig config;
    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "radius") {
            config.cornerRadius = toFloat(value, config.cornerRadius);
        } else if (key == "seg") {
            config.segKind = segFrom(value, config.segKind);
        }
        // Unknown keys ignored on purpose — which also retires the earlier
        // `arc` centre flag: a document authored under that model degrades to
        // sharp corners rather than failing to load.
    });
    return config;
}

ConveyorWaypointConfig ConveyorWaypointConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return {};
    if (it->second.type() != typeid(std::string)) return {};
    return decode(std::any_cast<const std::string&>(it->second));
}

void ConveyorWaypointConfig::write(Object3D& object) const {

    // The default config and no entry mean the same thing; writing the entry
    // only when it says something keeps ordinary waypoints clean.
    if (*this == ConveyorWaypointConfig{}) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encode();
}

void ConveyorWaypointConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

const char* ConveyorWaypointConfig::label(conveyor::SegKind kind) {

    switch (kind) {
        case conveyor::SegKind::Flat: return "Flat";
        case conveyor::SegKind::Rollers: return "Rollers";
        case conveyor::SegKind::Cleats: return "Cleats";
    }
    return "Flat";
}


// --- ConveyorConfig ---------------------------------------------------------

std::string ConveyorConfig::encode() const {

    std::string out;
    out += "width=";
    out += number(width);
    out += ";speed=";
    out += number(speed);
    out += ";reverse=";
    out += reverse ? "1" : "0";
    out += ";smooth=";
    out += smooth ? "1" : "0";
    out += ";separator=";
    out += separator ? "1" : "0";
    out += ";wallHeight=";
    out += number(wallHeight);
    out += ";rollerRadius=";
    out += number(rollerRadius);
    out += ";cleatHeight=";
    out += number(cleatHeight);
    out += ";cleatSpacing=";
    out += number(cleatSpacing);
    out += ";samples=";
    out += std::to_string(samples);
    out += ";frame=";
    out += frame ? "1" : "0";
    return out;
}

std::optional<ConveyorConfig> ConveyorConfig::decode(const std::string& text) {

    ConveyorConfig config;
    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "width") {
            config.width = toFloat(value, config.width);
        } else if (key == "speed") {
            config.speed = toFloat(value, config.speed);
        } else if (key == "reverse") {
            config.reverse = value == "1" || value == "true";
        } else if (key == "smooth") {
            config.smooth = value == "1" || value == "true";
        } else if (key == "separator") {
            config.separator = value == "1" || value == "true";
        } else if (key == "wallHeight") {
            config.wallHeight = toFloat(value, config.wallHeight);
        } else if (key == "rollerRadius") {
            config.rollerRadius = toFloat(value, config.rollerRadius);
        } else if (key == "cleatHeight") {
            config.cleatHeight = toFloat(value, config.cleatHeight);
        } else if (key == "cleatSpacing") {
            config.cleatSpacing = toFloat(value, config.cleatSpacing);
        } else if (key == "samples") {
            config.samples = toInt(value, config.samples);
        } else if (key == "frame") {
            config.frame = value == "1" || value == "true";
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });
    return config;
}

std::optional<ConveyorConfig> ConveyorConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;
    return decode(std::any_cast<const std::string&>(it->second));
}

void ConveyorConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void ConveyorConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

bool ConveyorConfig::isConveyor(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    return it != object.userData.end() && it->second.type() == typeid(std::string);
}

Object3D* ConveyorConfig::conveyorOf(const Object3D& object) {

    auto* parent = object.parent;
    if (!parent || !isConveyor(*parent)) return nullptr;
    return isDerived(object) ? nullptr : parent;
}

bool ConveyorConfig::isDerived(const Object3D& object) {

    const auto it = object.userData.find(derivedKey);
    return it != object.userData.end();
}

void ConveyorConfig::markDerived(Object3D& object) {

    object.userData[derivedKey] = std::string("1");
}

Object3D* ConveyorConfig::derivedGroup(const Object3D& conveyor) {

    for (auto* child : conveyor.children) {
        if (isDerived(*child)) return child;
    }
    return nullptr;
}

std::string ConveyorConfig::roleOf(const Object3D& object) {

    const auto it = object.userData.find(roleKey);
    if (it == object.userData.end()) return {};
    if (it->second.type() != typeid(std::string)) return {};
    return std::any_cast<const std::string&>(it->second);
}

std::vector<Object3D*> ConveyorConfig::waypointNodes(const Object3D& conveyor) {

    std::vector<Object3D*> nodes;
    nodes.reserve(conveyor.children.size());
    for (auto* child : conveyor.children) {
        if (!isDerived(*child)) nodes.push_back(child);
    }
    return nodes;
}

std::size_t ConveyorConfig::pointIndexOf(const Object3D& conveyor, const Object3D& child) {

    std::size_t index = 0;
    for (const auto* candidate : conveyor.children) {
        if (isDerived(*candidate)) continue;
        if (candidate == &child) return index;
        ++index;
    }
    return index;// == the waypoint count, i.e. "not a waypoint"
}

std::size_t ConveyorConfig::childSlotForPointIndex(const Object3D& conveyor, std::size_t pointIndex) {

    std::size_t points = 0;
    std::size_t afterLastPoint = 0;
    for (std::size_t i = 0; i < conveyor.children.size(); ++i) {
        if (isDerived(*conveyor.children[i])) continue;
        if (points == pointIndex) return i;
        ++points;
        afterLastPoint = i + 1;
    }
    return afterLastPoint;
}

conveyor::ConveyorSpec ConveyorConfig::spec(const Object3D& conveyor) const {

    threepp::conveyor::ConveyorSpec s;
    s.width = std::max(width, 0.05f);
    s.speed = speed;
    s.reverse = reverse;
    s.smooth = smooth;
    s.separator = separator;
    s.wallHeight = std::max(wallHeight, 0.05f);
    s.rollerRadius = std::max(rollerRadius, 0.01f);
    s.cleatHeight = std::max(cleatHeight, 0.01f);
    s.cleatSpacing = std::max(cleatSpacing, 0.05f);
    s.samples = std::clamp(samples, 2, maxSamples);
    s.frame = frame;

    for (const auto* node : waypointNodes(conveyor)) {
        const auto wc = ConveyorWaypointConfig::read(*node);
        s.waypoints.push_back({node->position, std::max(wc.cornerRadius, 0.f), wc.segKind});
    }
    return s;
}

namespace {

    // Wholesale content replacement: every child of the derived group goes,
    // geometries disposed (the renderer keys GPU buffers on geometry identity;
    // an undisposed orphan both leaks them and can re-arm recycled-pointer
    // staleness). Materials are left alone — they may be shared.
    void clearGenerated(Object3D& group) {

        const auto children = group.children;// copy: removeFromParent rewrites it
        for (auto* child : children) {
            const auto geometry = child->geometry();
            child->removeFromParent();
            if (geometry) geometry->dispose();
        }
    }

    void tag(Object3D& object, const char* role) {

        object.userData[ConveyorConfig::roleKey] = std::string(role);
    }

    std::shared_ptr<Mesh> part(const std::shared_ptr<BufferGeometry>& geometry,
                               const std::shared_ptr<Material>& material,
                               const char* role, const std::string& name) {

        auto mesh = Mesh::create(geometry, material);
        mesh->name = name;
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        tag(*mesh, role);
        return mesh;
    }

}// namespace

void ConveyorConfig::syncDerived(Object3D& conveyor) const {

    namespace cv = threepp::conveyor;

    // Exactly one tagged group, ever. Extras (a duplicate, a hand-edited
    // document) are removed rather than tolerated.
    std::vector<Object3D*> tagged;
    for (auto* child : conveyor.children) {
        if (isDerived(*child)) tagged.push_back(child);
    }
    Object3D* group = tagged.empty() ? nullptr : tagged.front();
    for (auto* child : tagged) {
        if (child != group) child->removeFromParent();
    }

    const auto resolved = spec(conveyor);

    if (!group) {
        auto created = Group::create();
        created->name = "Conveyor Parts";
        markDerived(*created);
        conveyor.add(created);
        group = created.get();
    }

    clearGenerated(*group);

    if (resolved.waypoints.size() < 2) return;// legal while authoring — nothing to build yet

    const auto sampled = cv::resamplePath(resolved.waypoints, resolved.smooth, resolved.samples);
    if (sampled.size() < 2) return;

    if (resolved.separator) {
        // A separator is a guide rail / lane divider: one translucent wall,
        // no belt and no frame.
        auto wallMat = MeshStandardMaterial::create();
        wallMat->color = Color(0xbfc6cc);
        wallMat->roughness = 0.7f;
        wallMat->metalness = 0.f;
        wallMat->transparent = true;
        wallMat->opacity = 0.55f;
        wallMat->side = Side::Double;
        group->add(part(cv::wallGeometry(sampled, resolved.wallHeight), wallMat, "wall", "Wall"));
        return;
    }

    // One belt material per conveyor: every flat/cleats run shares it, so the
    // play session scrolls the whole conveyor by bumping one texture offset.
    auto beltTex = cv::beltTexture();
    beltTex->repeat.set(std::max(1.f, std::round(resolved.width / cv::kBeltTileLength)),
                        1.f / cv::kBeltTileLength);
    auto beltMat = MeshStandardMaterial::create();
    beltMat->color = Color(0xffffff);
    beltMat->roughness = 0.85f;
    beltMat->metalness = 0.f;
    beltMat->side = Side::Double;
    beltMat->map = beltTex;
    // The belt pattern scrolls via the texture offset with no geometric
    // motion — temporal passes (Vulkan TAA) hold a short history here or the
    // moving pattern smears.
    beltMat->textureAnimatedHint = true;

    auto rollerMat = MeshStandardMaterial::create();
    rollerMat->color = Color(0x8a9097);
    rollerMat->roughness = 0.4f;
    rollerMat->metalness = 0.7f;
    rollerMat->flatShading = true;

    auto cleatMat = MeshStandardMaterial::create();
    cleatMat->color = Color(0x202428);
    cleatMat->roughness = 0.6f;
    cleatMat->metalness = 0.3f;

    auto frameMat = MeshStandardMaterial::create();
    frameMat->color = Color(0x9098a0);
    frameMat->roughness = 0.5f;
    frameMat->metalness = 0.85f;

    int beltCount = 0, rollerCount = 0, cleatCount = 0;

    for (const auto& run : cv::resamplePathByKind(resolved.waypoints, resolved.smooth,
                                                  resolved.samples)) {
        if (run.pts.size() < 2) continue;

        if (run.kind == cv::SegKind::Rollers) {
            // A row of cylinders across the belt; the play session spins them.
            // One shared geometry per conveyor.
            static constexpr int radial = 12;
            auto cyl = CylinderGeometry::create(resolved.rollerRadius, resolved.rollerRadius,
                                                resolved.width, radial);
            for (const auto& r : cv::rollerTransforms(run.pts, resolved.rollerRadius,
                                                      cv::rollerSpacing(resolved.rollerRadius))) {
                auto roller = part(cyl, rollerMat, "roller", "Roller " + std::to_string(++rollerCount));
                roller->position.copy(r.center);
                roller->quaternion.copy(r.orientation);
                group->add(roller);
            }
            continue;
        }

        // Flat or cleats: the scrolling belt ribbon.
        group->add(part(cv::ribbonGeometry(run.pts, resolved.width), beltMat, "belt",
                        ++beltCount == 1 ? "Belt" : "Belt " + std::to_string(beltCount)));

        if (run.kind == cv::SegKind::Cleats) {
            // Static full-height preview bars; during play these are hidden and
            // the session drives its own travelling ones (real colliders).
            auto box = BoxGeometry::create(cv::kCleatThickness, resolved.cleatHeight,
                                           resolved.width);
            for (const auto& c : cv::cleatTransforms(run.pts, resolved.cleatHeight,
                                                     resolved.cleatSpacing)) {
                auto bar = part(box, cleatMat, "cleat", "Cleat " + std::to_string(++cleatCount));
                bar->position.copy(c.center);
                bar->quaternion.copy(c.orientation);
                group->add(bar);
            }
        }
    }

    if (resolved.frame) {
        // The support frame — the first-party procedural stand-in for a
        // conveyor model: side rails along the whole path, legs down to the
        // conveyor's local ground plane (y = 0), and an end drum (pulley) at
        // each open end. Purely visual.
        const auto profile = cv::FrameProfile::forWidth(resolved.width);

        group->add(part(cv::railGeometry(sampled, resolved.width, -1, profile), frameMat,
                        "frame", "Rail L"));
        group->add(part(cv::railGeometry(sampled, resolved.width, +1, profile), frameMat,
                        "frame", "Rail R"));

        auto legBox = BoxGeometry::create(profile.legThickness, 1.f, profile.legThickness);
        int legCount = 0;
        for (const auto& leg : cv::legTransforms(sampled, resolved.width, 0.f, profile)) {
            auto mesh = part(legBox, frameMat, "frame", "Leg " + std::to_string(++legCount));
            mesh->position.copy(leg.center);
            mesh->quaternion.copy(leg.orientation);
            mesh->scale.y = leg.length;
            group->add(mesh);
        }

        auto drumGeom = CylinderGeometry::create(profile.drumRadius, profile.drumRadius,
                                                 resolved.width + 2.f * profile.railThickness, 16);
        int drumCount = 0;
        for (const auto& drum : cv::endDrumTransforms(sampled, profile)) {
            auto mesh = part(drumGeom, frameMat, "drum", "Drum " + std::to_string(++drumCount));
            mesh->position.copy(drum.center);
            mesh->quaternion.copy(drum.orientation);
            group->add(mesh);
        }
    }
}
