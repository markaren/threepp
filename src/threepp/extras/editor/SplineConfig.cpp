
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"

#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Fixed 6 decimals with trailing zeros trimmed: stable across platforms
    // (unlike ostream defaults with a locale) and byte-identical for an
    // unchanged value, which keeps saved documents diff-clean.
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
    return out;
}

std::optional<SplineConfig> SplineConfig::decode(const std::string& text) {

    SplineConfig config;

    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find(';', start);
        const auto token = std::string_view(text).substr(
                start, (end == std::string::npos ? text.size() : end) - start);
        const auto eq = token.find('=');
        if (eq != std::string_view::npos) {
            const auto key = token.substr(0, eq);
            const auto value = token.substr(eq + 1);
            if (key == "type") {
                config.type = typeFrom(value, config.type);
            } else if (key == "closed") {
                config.closed = value == "1" || value == "true";
            } else if (key == "tension") {
                config.tension = toFloat(value, config.tension);
            } else if (key == "samples") {
                config.samples = toInt(value, config.samples);
            }
            // Unknown keys ignored on purpose: a document written by a newer
            // editor still loads here.
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }

    // An empty string is still a spline — the key's presence is the definition,
    // and every parameter has a default.
    return config;
}

std::optional<SplineConfig> SplineConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    return decode(std::any_cast<const std::string&>(it->second));
}

void SplineConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void SplineConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

bool SplineConfig::isSpline(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    return it != object.userData.end() && it->second.type() == typeid(std::string);
}

Object3D* SplineConfig::splineOf(const Object3D& object) {

    auto* parent = object.parent;
    return parent && isSpline(*parent) ? parent : nullptr;
}

std::vector<Vector3> SplineConfig::controlPoints(const Object3D& spline) {

    std::vector<Vector3> points;
    points.reserve(spline.children.size());
    for (const auto* child : spline.children) points.push_back(child->position);
    return points;
}

std::shared_ptr<CatmullRomCurve3> SplineConfig::curve(const Object3D& spline) const {

    auto points = controlPoints(spline);
    if (points.size() < 2) return nullptr;

    return std::make_shared<CatmullRomCurve3>(
            std::move(points), closed, curveTypeOf(type), tension);
}
