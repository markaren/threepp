
#include "threepp/extras/editor/TextConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/geometries/TextGeometry.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    // Same trimmed fixed-decimal encoding SplineConfig uses, for the same
    // reason: byte-identical output for an unchanged value keeps saved
    // documents diff-clean.
    // Parsed once from the embedded typeface JSON; every rebuild after the
    // first is just geometry.
    const Font& defaultFont() {

        static const Font font = FontLoader().defaultFont();
        return font;
    }

    TextConfig::Align alignFrom(std::string_view text, TextConfig::Align fallback) {

        if (text == "left") return TextConfig::Align::Left;
        if (text == "center") return TextConfig::Align::Center;
        if (text == "right") return TextConfig::Align::Right;
        return fallback;
    }

    const char* alignToken(TextConfig::Align align) {

        switch (align) {
            case TextConfig::Align::Left: return "left";
            case TextConfig::Align::Center: return "center";
            case TextConfig::Align::Right: return "right";
        }
        return "center";
    }

}// namespace


const char* TextConfig::label(Align align) {

    switch (align) {
        case Align::Left: return "Left";
        case Align::Center: return "Center";
        case Align::Right: return "Right";
    }
    return "Center";
}


std::string TextConfig::encodeParams() const {

    std::string out;
    out += "size=";
    out += number(size);
    out += ";depth=";
    out += number(depth);
    out += ";curveSegments=";
    out += std::to_string(curveSegments);
    out += ";align=";
    out += alignToken(align);
    return out;
}

std::optional<TextConfig> TextConfig::read(const Object3D& object) {

    const auto it = object.userData.find(textKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    TextConfig config;
    config.text = std::any_cast<const std::string&>(it->second);

    const auto params = object.userData.find(paramsKey);
    if (params != object.userData.end() && params->second.type() == typeid(std::string)) {
        const auto& encoded = std::any_cast<const std::string&>(params->second);
        codec::parsePairs(encoded, [&](std::string_view key, std::string_view value) {
            if (key == "size") {
                config.size = toFloat(value, config.size);
            } else if (key == "depth") {
                config.depth = toFloat(value, config.depth);
            } else if (key == "curveSegments") {
                config.curveSegments = toInt(value, config.curveSegments);
            } else if (key == "align") {
                config.align = alignFrom(value, config.align);
            }
            // Unknown keys ignored on purpose: a document written by a
            // newer editor still loads here.
        });
    }

    return config;
}

void TextConfig::write(Object3D& object) const {

    object.userData[textKey] = text;
    object.userData[paramsKey] = encodeParams();
}

bool TextConfig::isText(const Object3D& object) {

    return hasEntry(object, textKey);
}

std::shared_ptr<BufferGeometry> TextConfig::buildGeometry() const {

    const float glyphSize = std::max(size, 1e-3f);
    const auto segments = static_cast<unsigned int>(std::clamp(curveSegments, 1, maxCurveSegments));

    std::shared_ptr<BufferGeometry> geometry;
    if (depth > 0.f) {
        ExtrudeGeometry::Options options;
        options.curveSegments = segments;
        options.steps = 1;
        options.depth = depth;
        // The Options default is a bevel sized for unit-scale mechanical
        // parts; on type it swallows whole glyphs.
        options.bevelEnabled = false;
        geometry = ExtrudeGeometry::create(defaultFont().generateShapes(text, glyphSize), options);
    } else {
        geometry = TextGeometry::create(
                text, TextGeometry::Options(defaultFont(), glyphSize, segments));
    }

    // Anchor the block on the origin, so the gizmo grabs the text where
    // `align` says rather than the baseline's left end. Guarded: empty
    // content (or all spaces) has no vertices, and anchoring an empty
    // bounding box would translate by NaN.
    if (const auto* position = geometry->getAttribute<float>("position");
        position && position->count() > 0) {

        geometry->computeBoundingBox();
        const auto& box = *geometry->boundingBox;

        float offsetX = 0.f;
        switch (align) {
            case Align::Left: offsetX = -box.min().x; break;
            case Align::Center: offsetX = -(box.min().x + box.max().x) * 0.5f; break;
            case Align::Right: offsetX = -box.max().x; break;
        }
        geometry->translate(offsetX,
                            -(box.min().y + box.max().y) * 0.5f,
                            -(box.min().z + box.max().z) * 0.5f);
        // translate() refreshes the cached box; the sphere is what the
        // raycast gates on, so make sure it exists and agrees.
        geometry->computeBoundingSphere();
    }
    return geometry;
}

void TextConfig::apply(Object3D& object) const {

    write(object);

    if (auto* mesh = object.as<Mesh>()) {
        const auto old = mesh->geometry();
        mesh->setGeometry(buildGeometry());
        if (old) old->dispose();
    }
}
