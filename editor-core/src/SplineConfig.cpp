
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"

#include <algorithm>
#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    SplineConfig::Type typeFrom(std::string_view text, SplineConfig::Type fallback) {

        if (text == "centripetal") return SplineConfig::Type::Centripetal;
        if (text == "chordal") return SplineConfig::Type::Chordal;
        if (text == "catmullrom") return SplineConfig::Type::CatmullRom;
        return fallback;
    }

    const char* typeToken(SplineConfig::Type type) {

        switch (type) {
            case SplineConfig::Type::Centripetal: return "centripetal";
            case SplineConfig::Type::Chordal: return "chordal";
            case SplineConfig::Type::CatmullRom: return "catmullrom";
        }
        return "centripetal";
    }

    SplineConfig::MeshKind meshFrom(std::string_view text, SplineConfig::MeshKind fallback) {

        if (text == "none") return SplineConfig::MeshKind::None;
        if (text == "tube") return SplineConfig::MeshKind::Tube;
        return fallback;
    }

    const char* meshToken(SplineConfig::MeshKind kind) {

        switch (kind) {
            case SplineConfig::MeshKind::None: return "none";
            case SplineConfig::MeshKind::Tube: return "tube";
        }
        return "none";
    }

    CatmullRomCurve3::CurveType curveTypeOf(SplineConfig::Type type) {

        switch (type) {
            case SplineConfig::Type::Centripetal: return CatmullRomCurve3::centripetal;
            case SplineConfig::Type::Chordal: return CatmullRomCurve3::chordal;
            case SplineConfig::Type::CatmullRom: return CatmullRomCurve3::catmullrom;
        }
        return CatmullRomCurve3::centripetal;
    }

}// namespace


const char* SplineConfig::label(Type type) {

    switch (type) {
        case Type::Centripetal: return "Centripetal";
        case Type::Chordal: return "Chordal";
        case Type::CatmullRom: return "CatmullRom";
    }
    return "Centripetal";
}

const char* SplineConfig::label(MeshKind kind) {

    switch (kind) {
        case MeshKind::None: return "None";
        case MeshKind::Tube: return "Tube";
    }
    return "None";
}

std::string SplineConfig::encode() const {

    std::string out;
    out += "type=";
    out += typeToken(type);
    out += ";closed=";
    out += closed ? "1" : "0";
    out += ";tension=";
    out += number(tension);
    out += ";samples=";
    out += std::to_string(samples);
    out += ";mesh=";
    out += meshToken(mesh);
    out += ";radius=";
    out += number(radius);
    out += ";radialSegments=";
    out += std::to_string(radialSegments);
    return out;
}

std::optional<SplineConfig> SplineConfig::decode(const std::string& text) {

    SplineConfig config;

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "type") {
            config.type = typeFrom(value, config.type);
        } else if (key == "closed") {
            config.closed = value == "1" || value == "true";
        } else if (key == "tension") {
            config.tension = toFloat(value, config.tension);
        } else if (key == "samples") {
            config.samples = toInt(value, config.samples);
        } else if (key == "mesh") {
            config.mesh = meshFrom(value, config.mesh);
        } else if (key == "radius") {
            config.radius = toFloat(value, config.radius);
        } else if (key == "radialSegments") {
            config.radialSegments = toInt(value, config.radialSegments);
                    }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    // An empty string is still a spline — the key's presence is the definition,
    // and every parameter has a default.
    return config;
}

std::optional<SplineConfig> SplineConfig::read(const Object3D& object) {

    return readEntry<SplineConfig>(object, userDataKey);
}

void SplineConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void SplineConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

bool SplineConfig::isSpline(const Object3D& object) {

    return hasEntry(object, userDataKey);
}

Object3D* SplineConfig::splineOf(const Object3D& object) {

    auto* parent = object.parent;
    if (!parent || !isSpline(*parent)) return nullptr;
    // The generated mesh is a child of the spline without being a point of it.
    return isDerived(object) ? nullptr : parent;
}

bool SplineConfig::isDerived(const Object3D& object) {

    const auto it = object.userData.find(derivedKey);
    return it != object.userData.end();
}

void SplineConfig::markDerived(Object3D& object) {

    object.userData[derivedKey] = std::string("1");
}

Object3D* SplineConfig::derivedMesh(const Object3D& spline) {

    for (auto* child : spline.children) {
        if (isDerived(*child)) return child;
    }
    return nullptr;
}

std::vector<Object3D*> SplineConfig::controlPointNodes(const Object3D& spline) {

    std::vector<Object3D*> nodes;
    nodes.reserve(spline.children.size());
    for (auto* child : spline.children) {
        if (!isDerived(*child)) nodes.push_back(child);
    }
    return nodes;
}

std::vector<Vector3> SplineConfig::controlPoints(const Object3D& spline) {

    std::vector<Vector3> points;
    points.reserve(spline.children.size());
    for (const auto* child : spline.children) {
        if (!isDerived(*child)) points.push_back(child->position);
    }
    return points;
}

std::size_t SplineConfig::pointIndexOf(const Object3D& spline, const Object3D& child) {

    std::size_t index = 0;
    for (const auto* candidate : spline.children) {
        if (isDerived(*candidate)) continue;
        if (candidate == &child) return index;
        ++index;
    }
    return index;// == the point count, i.e. "not a control point"
}

std::size_t SplineConfig::childSlotForPointIndex(const Object3D& spline, std::size_t pointIndex) {

    std::size_t points = 0;
    // One past the last point seen, so appending goes before a trailing
    // generated mesh rather than after it.
    std::size_t afterLastPoint = 0;
    for (std::size_t i = 0; i < spline.children.size(); ++i) {
        if (isDerived(*spline.children[i])) continue;
        if (points == pointIndex) return i;
        ++points;
        afterLastPoint = i + 1;
    }
    return afterLastPoint;
}

std::shared_ptr<CatmullRomCurve3> SplineConfig::curve(const Object3D& spline) const {

    auto points = controlPoints(spline);
    if (points.size() < 2) return nullptr;

    return std::make_shared<CatmullRomCurve3>(
            std::move(points), closed, curveTypeOf(type), tension);
}

unsigned int SplineConfig::divisions(const Object3D& spline) const {

    // The count the user authored is per SEGMENT; getPoints() divides the whole
    // curve, so the two have to be multiplied out here.
    const auto points = static_cast<int>(controlPoints(spline).size());
    const int segments = points - (closed ? 0 : 1);
    return static_cast<unsigned int>(
            std::clamp(samples, 1, maxSamples) * std::max(segments, 1));
}
