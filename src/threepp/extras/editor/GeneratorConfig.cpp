
#include "threepp/extras/editor/GeneratorConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Same guarded read ScriptConfig uses: userData is std::any, and a document
    // written by another tool may carry a non-string under our key.
    std::string readString(const Object3D& object, const char* key) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return {};
        if (it->second.type() != typeid(std::string)) return {};
        return std::any_cast<const std::string&>(it->second);
    }

}// namespace

std::optional<std::string> GeneratorConfig::field(const std::string& name) const {

    const auto it = std::find_if(fields.begin(), fields.end(),
                                 [&name](const ScriptConfig::Field& f) { return f.name == name; });
    if (it == fields.end()) return std::nullopt;
    return it->value;
}

void GeneratorConfig::setField(const std::string& name, const std::string& value) {

    const auto it = std::find_if(fields.begin(), fields.end(),
                                 [&name](const ScriptConfig::Field& f) { return f.name == name; });
    if (it != fields.end()) {
        it->value = value;
        return;
    }
    fields.push_back({name, value});
}

void GeneratorConfig::eraseField(const std::string& name) {

    fields.erase(std::remove_if(fields.begin(), fields.end(),
                                [&name](const ScriptConfig::Field& f) { return f.name == name; }),
                 fields.end());
}

void GeneratorConfig::retainFields(const std::vector<std::string>& names) {

    fields.erase(std::remove_if(fields.begin(), fields.end(),
                                [&names](const ScriptConfig::Field& f) {
                                    return std::find(names.begin(), names.end(), f.name) ==
                                           names.end();
                                }),
                 fields.end());
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
