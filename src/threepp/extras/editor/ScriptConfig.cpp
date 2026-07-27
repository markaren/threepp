
#include "threepp/extras/editor/ScriptConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::string readString(const Object3D& object, const char* key) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return {};
        if (it->second.type() != typeid(std::string)) return {};
        return std::any_cast<const std::string&>(it->second);
    }

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

    char buffer[32];
    const int n = std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
    std::string out(buffer, buffer + (n > 0 ? n : 0));
    if (out.find('.') == std::string::npos) return out;
    while (!out.empty() && out.back() == '0') out.pop_back();
    if (!out.empty() && out.back() == '.') out.pop_back();
    return out.empty() ? "0" : out;
}

std::string ScriptConfig::toText(int value) {

    return std::to_string(value);
}

std::string ScriptConfig::toText(bool value) {

    return value ? "1" : "0";
}

float ScriptConfig::toFloat(const std::string& text, float fallback) {

    try {
        return std::stof(text);
    } catch (...) {
        return fallback;
    }
}

int ScriptConfig::toInt(const std::string& text, int fallback) {

    try {
        return std::stoi(text);
    } catch (...) {
        return fallback;
    }
}

bool ScriptConfig::toBool(const std::string& text, bool fallback) {

    if (text.empty()) return fallback;
    // Accepts what the editor writes ("1"/"0") plus what a hand-edited file is
    // likely to contain.
    if (text == "1" || text == "true" || text == "True") return true;
    if (text == "0" || text == "false" || text == "False") return false;
    return fallback;
}

std::string ScriptConfig::sanitized(std::string text) {

    text.erase(std::remove_if(text.begin(), text.end(),
                              [](char c) { return c == ';' || c == '='; }),
               text.end());
    return text;
}

std::optional<std::string> ScriptConfig::field(const std::string& name) const {

    const auto it = std::find_if(fields.begin(), fields.end(),
                                 [&name](const Field& f) { return f.name == name; });
    if (it == fields.end()) return std::nullopt;
    return it->value;
}

void ScriptConfig::setField(const std::string& name, const std::string& value) {

    const auto it = std::find_if(fields.begin(), fields.end(),
                                 [&name](const Field& f) { return f.name == name; });
    if (it != fields.end()) {
        it->value = value;
        return;
    }
    fields.push_back({name, value});
}

void ScriptConfig::eraseField(const std::string& name) {

    fields.erase(std::remove_if(fields.begin(), fields.end(),
                                [&name](const Field& f) { return f.name == name; }),
                 fields.end());
}

void ScriptConfig::retainFields(const std::vector<std::string>& names) {

    fields.erase(std::remove_if(fields.begin(), fields.end(),
                                [&names](const Field& f) {
                                    return std::find(names.begin(), names.end(), f.name) == names.end();
                                }),
                 fields.end());
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

    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find(';', start);
        const auto token = std::string_view(text).substr(
                start, (end == std::string::npos ? text.size() : end) - start);
        const auto eq = token.find('=');
        if (eq != std::string_view::npos && eq > 0) {
            out.push_back({std::string(token.substr(0, eq)), std::string(token.substr(eq + 1))});
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
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
