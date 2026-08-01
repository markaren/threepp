
#include "threepp/extras/editor/ScriptConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

}// namespace


void ScriptConfig::setPath(std::string value) {

    path = std::move(value);
    if (!path.empty()) source.clear();
}

void ScriptConfig::setSource(std::string value) {

    source = std::move(value);
    if (!source.empty()) path.clear();
}

std::string ScriptConfig::toText(float value) {

    return codec::number(value);
}

std::string ScriptConfig::toText(int value) {

    return std::to_string(value);
}

std::string ScriptConfig::toText(bool value) {

    return value ? "1" : "0";
}

float ScriptConfig::toFloat(const std::string& text, float fallback) {

    return codec::toFloat(text, fallback);
}

int ScriptConfig::toInt(const std::string& text, int fallback) {

    return codec::toInt(text, fallback);
}

bool ScriptConfig::toBool(const std::string& text, bool fallback) {

    return codec::toBool(text, fallback);
}

std::string ScriptConfig::sanitized(std::string text) {

    text.erase(std::remove_if(text.begin(), text.end(),
                              [](char c) { return c == ';' || c == '='; }),
               text.end());
    return text;
}

std::optional<std::string> ScriptConfig::fieldIn(const std::vector<Field>& fields,
                                                 const std::string& name) {

    const auto it = std::find_if(fields.begin(), fields.end(),
                                 [&name](const Field& f) { return f.name == name; });
    if (it == fields.end()) return std::nullopt;
    return it->value;
}

void ScriptConfig::setFieldIn(std::vector<Field>& fields, const std::string& name,
                              const std::string& value) {

    const auto it = std::find_if(fields.begin(), fields.end(),
                                 [&name](const Field& f) { return f.name == name; });
    if (it != fields.end()) {
        it->value = value;
        return;
    }
    fields.push_back({name, value});
}

void ScriptConfig::eraseFieldIn(std::vector<Field>& fields, const std::string& name) {

    fields.erase(std::remove_if(fields.begin(), fields.end(),
                                [&name](const Field& f) { return f.name == name; }),
                 fields.end());
}

void ScriptConfig::retainFieldsIn(std::vector<Field>& fields, const std::vector<std::string>& names) {

    fields.erase(std::remove_if(fields.begin(), fields.end(),
                                [&names](const Field& f) {
                                    return std::find(names.begin(), names.end(), f.name) == names.end();
                                }),
                 fields.end());
}

std::optional<std::string> ScriptConfig::field(const std::string& name) const {

    return fieldIn(fields, name);
}

void ScriptConfig::setField(const std::string& name, const std::string& value) {

    setFieldIn(fields, name, value);
}

void ScriptConfig::eraseField(const std::string& name) {

    eraseFieldIn(fields, name);
}

void ScriptConfig::retainFields(const std::vector<std::string>& names) {

    retainFieldsIn(fields, names);
}

std::string ScriptConfig::encodeFields() const {

    std::string out;
    for (const auto& f : fields) {
        const auto name = sanitized(f.name);
        // A field with no name left after sanitizing cannot be read back, so it
        // is dropped rather than written as a stray "=value" token.
        if (name.empty()) continue;
        if (!out.empty()) out += ';';
        out += name;
        out += '=';
        out += sanitized(f.value);
    }
    return out;
}

std::vector<ScriptConfig::Field> ScriptConfig::decodeFields(const std::string& text) {

    std::vector<Field> out;
    if (text.empty()) return out;

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key.empty()) return;// `=value` with no key is malformed, not a field
        out.push_back({std::string(key), std::string(value)});
    });
    return out;
}

std::optional<ScriptConfig> ScriptConfig::read(const Object3D& object) {

    ScriptConfig config;
    // Inline first: a document that somehow carries both (hand-edited, or
    // merged) is read as inline and drops the path, so the next write restores
    // the one-form invariant instead of preserving the contradiction.
    config.source = readString(object, sourceKey);
    if (config.source.empty()) config.path = readString(object, pathKey);
    if (config.empty()) return std::nullopt;

    config.fields = decodeFields(readString(object, fieldsKey));
    return config;
}

void ScriptConfig::write(Object3D& object) const {

    if (empty()) {
        erase(object);
        return;
    }

    if (isInline()) {
        object.userData[sourceKey] = source;
        object.userData.erase(pathKey);
    } else {
        object.userData[pathKey] = path;
        object.userData.erase(sourceKey);
    }

    // No fields means no entry: an untouched script leaves one line in the
    // document, not two.
    const auto encoded = encodeFields();
    if (encoded.empty()) {
        object.userData.erase(fieldsKey);
    } else {
        object.userData[fieldsKey] = encoded;
    }
}

void ScriptConfig::erase(Object3D& object) {

    object.userData.erase(pathKey);
    object.userData.erase(sourceKey);
    object.userData.erase(fieldsKey);
}
