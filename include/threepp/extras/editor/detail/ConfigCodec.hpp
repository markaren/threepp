// Shared codec for the editor Config family's userData string format.
//
// Every Config serializes itself as a flat `key=value;…` string under one
// userData key. The four value codecs and the pair tokenizer below were
// copy-pasted into ~10 Config translation units (and had already begun to
// drift cosmetically); this header is the single source.
//
// Format stability matters: number() is fixed 6-decimals with trailing zeros
// trimmed — stable across platforms (no locale) and byte-identical for an
// unchanged value, which keeps saved documents diff-clean. Do not change the
// emitted text without considering every saved scene.

#ifndef THREEPP_EXTRAS_EDITOR_DETAIL_CONFIGCODEC_HPP
#define THREEPP_EXTRAS_EDITOR_DETAIL_CONFIGCODEC_HPP

#include "threepp/core/Object3D.hpp"

#include <any>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>

namespace threepp::editor::codec {

    // Fixed 6 decimals with trailing zeros trimmed (see header note).
    inline std::string number(float value) {

        char buffer[32];
        const int n = std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
        std::string out(buffer, buffer + (n > 0 ? n : 0));
        if (out.find('.') == std::string::npos) return out;
        while (!out.empty() && out.back() == '0') out.pop_back();
        if (!out.empty() && out.back() == '.') out.pop_back();
        return out.empty() ? "0" : out;
    }

    inline float toFloat(std::string_view text, float fallback) {

        try {
            return std::stof(std::string(text));
        } catch (...) {
            return fallback;
        }
    }

    inline int toInt(std::string_view text, int fallback) {

        try {
            return std::stoi(std::string(text));
        } catch (...) {
            return fallback;
        }
    }

    // Accepts what the editor writes ("1"/"0") plus what a hand-edited file is
    // likely to contain.
    inline bool toBool(std::string_view text, bool fallback) {

        if (text.empty()) return fallback;
        if (text == "1" || text == "true" || text == "True") return true;
        if (text == "0" || text == "false" || text == "False") return false;
        return fallback;
    }

    // Guarded userData string read: {} when the key is absent or not a string.
    inline std::string readString(const Object3D& object, const char* key) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return {};
        if (it->second.type() != typeid(std::string)) return {};
        return std::any_cast<const std::string&>(it->second);
    }

    // True when the object carries a string entry under `key`. This is the
    // whole of every isConveyor / isGranular / isParticleField: the entry's
    // PRESENCE is what makes a node one of those things, never its contents.
    inline bool hasEntry(const Object3D& object, const char* key) {

        const auto it = object.userData.find(key);
        return it != object.userData.end() && it->second.type() == typeid(std::string);
    }

    // Decode the entry under `key` through Config::decode.
    //
    // nullopt means "this object carries no such entry", which every read()
    // distinguishes from "carries one that decodes to the defaults" — the two
    // are different documents, and the erase-at-defaults write rule depends on
    // the difference. A present-but-unparsable entry still decodes (to the
    // defaults), because a newer editor's keys must not erase an older one's.
    template<class Config>
    std::optional<Config> readEntry(const Object3D& object, const char* key) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return std::nullopt;
        if (it->second.type() != typeid(std::string)) return std::nullopt;
        return Config::decode(std::any_cast<const std::string&>(it->second));
    }

    // Walk a flat `key=value;…` string, calling `apply(key, value)` per pair.
    // Unknown keys are the CALLER's business to ignore — every decode() does,
    // on purpose: a document written by a newer editor still loads.
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

}// namespace threepp::editor::codec

#endif//THREEPP_EXTRAS_EDITOR_DETAIL_CONFIGCODEC_HPP
