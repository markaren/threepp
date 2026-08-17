
#include "threepp/extras/editor/GeneratorConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    // Same guarded read ScriptConfig uses: userData is std::any, and a document
    // written by another tool may carry a non-string under our key.
}// namespace

std::optional<std::string> GeneratorConfig::field(const std::string& name) const {

    return ScriptConfig::fieldIn(fields, name);
}

void GeneratorConfig::setField(const std::string& name, const std::string& value) {

    ScriptConfig::setFieldIn(fields, name, value);
}

void GeneratorConfig::eraseField(const std::string& name) {

    ScriptConfig::eraseFieldIn(fields, name);
}

void GeneratorConfig::retainFields(const std::vector<std::string>& names) {

    ScriptConfig::retainFieldsIn(fields, names);
}

std::optional<GeneratorConfig> GeneratorConfig::read(const Object3D& object) {

    GeneratorConfig config;
    config.source = readString(object, sourceKey);
    if (config.empty()) return std::nullopt;

    // Reusing ScriptConfig's encoding rather than repeating it: the format, the
    // delimiter sanitizing and the lenient parsing are all the same problem.
    config.fields = ScriptConfig::decodeFields(readString(object, fieldsKey));
    return config;
}

bool GeneratorConfig::isGenerator(const Object3D& object) {

    return !readString(object, sourceKey).empty();
}

void GeneratorConfig::write(Object3D& object) const {

    if (empty()) {
        erase(object);
        return;
    }

    object.userData[sourceKey] = source;

    // No fields means no entry, so a generator that exposes nothing leaves one
    // line in the document rather than two.
    GeneratorConfig copy = *this;
    ScriptConfig encoder;
    encoder.fields = copy.fields;
    const auto encoded = encoder.encodeFields();
    if (encoded.empty()) {
        object.userData.erase(fieldsKey);
    } else {
        object.userData[fieldsKey] = encoded;
    }
}

void GeneratorConfig::erase(Object3D& object) {

    object.userData.erase(sourceKey);
    object.userData.erase(fieldsKey);
}

Object3D* GeneratorConfig::generatedChild(const Object3D& generator) {

    for (auto* child : generator.children) {
        if (!child) continue;
        const auto it = child->userData.find(generatedKey);
        if (it != child->userData.end()) return child;
    }
    return nullptr;
}
