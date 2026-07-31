// "px,py,pz@tx,ty,tz" — a camera placement, in the one spelling the editor,
// the player and the shell all agree on.
//
// A document can say how it wants to be looked at (scene.userData["editorView"],
// see EditorApp::applyDocumentView), and the same text is what --shot takes on a
// command line. That is deliberate: a vantage can be copied out of a scene into
// a shell and back. Which is exactly why the parse lives here and not in one
// front end — a second parser is how the two spellings drift apart.
//
// Header-only, and free of everything but Vector3: the editor app, the player
// and any tool that wants to place a camera from a string can include it without
// taking on a translation unit or a link dependency.

#ifndef THREEPP_EDITOR_VIEWSPEC_HPP
#define THREEPP_EDITOR_VIEWSPEC_HPP

#include "threepp/math/Vector3.hpp"

#include <exception>
#include <sstream>
#include <string>

namespace threepp::editor {

    // Fills `position` and `target` from "px,py,pz@tx,ty,tz". Both outputs are
    // left untouched when the text does not parse, so a caller can seed them
    // with a default and ignore the result.
    inline bool parseViewSpec(const std::string& text, Vector3& position, Vector3& target) {

        const auto at = text.find('@');
        if (at == std::string::npos) return false;

        const auto triple = [](const std::string& part, Vector3& out) {
            std::istringstream stream(part);
            std::string field;
            float values[3];
            for (float& value : values) {
                if (!std::getline(stream, field, ',')) return false;
                try {
                    value = std::stof(field);
                } catch (const std::exception&) {
                    return false;
                }
            }
            // Everything after the third number is a typo, not a fourth axis.
            if (std::getline(stream, field, ',')) return false;
            out.set(values[0], values[1], values[2]);
            return true;
        };

        Vector3 wantPosition;
        Vector3 wantTarget;
        if (!triple(text.substr(0, at), wantPosition)) return false;
        if (!triple(text.substr(at + 1), wantTarget)) return false;

        position.copy(wantPosition);
        target.copy(wantTarget);
        return true;
    }

}// namespace threepp::editor

#endif//THREEPP_EDITOR_VIEWSPEC_HPP
