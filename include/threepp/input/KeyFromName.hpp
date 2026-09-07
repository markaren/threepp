// A friendly key NAME ('W', 'a', 'SPACE', 'UP', 'KP8') to threepp's Key enum.
//
// One mapping, one place. The names are a little contract that now crosses
// three surfaces — Canvas.is_key_down in the Python wheel, the editor's
// threepp.editor.is_key_down (answered from ImGui state, but documented as
// taking "the same names Canvas.is_key_down takes"), and the player's keyboard
// provider — so the spelling a script was written against must mean the same
// key everywhere. This header is where it means it.
//
// Unknown names map to Key::UNKNOWN, which no keyboard ever reports as held:
// a script polling a misspelled name reads False rather than raising, the same
// contract is_key_down keeps for a missing window.

#ifndef THREEPP_KEYFROMNAME_HPP
#define THREEPP_KEYFROMNAME_HPP

#include "threepp/input/KeyListener.hpp"

#include <cctype>
#include <string>

namespace threepp {

    inline Key keyFromName(std::string n) {

        for (auto& ch : n) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        // Letters/digits exploit the enum's contiguous A-Z and NUM_0-NUM_9 ranges.
        if (n.size() == 1 && n[0] >= 'A' && n[0] <= 'Z')
            return static_cast<Key>(static_cast<int>(Key::A) + (n[0] - 'A'));
        if (n.size() == 1 && n[0] >= '0' && n[0] <= '9')
            return static_cast<Key>(static_cast<int>(Key::NUM_0) + (n[0] - '0'));
        // Numpad keys: "KP8" / "NUM8" / "NUMPAD8" -> Key::KP_8 (distinct from the
        // top-row digit "8" -> Key::NUM_8).
        for (const std::string& pre : {std::string("KP"), std::string("NUMPAD"), std::string("NUM")}) {
            if (n.size() == pre.size() + 1 && n.compare(0, pre.size(), pre) == 0 &&
                n.back() >= '0' && n.back() <= '9')
                return static_cast<Key>(static_cast<int>(Key::KP_0) + (n.back() - '0'));
        }
        if (n == "SPACE") return Key::SPACE;
        if (n == "UP") return Key::UP;
        if (n == "DOWN") return Key::DOWN;
        if (n == "LEFT") return Key::LEFT;
        if (n == "RIGHT") return Key::RIGHT;
        if (n == "ESCAPE" || n == "ESC") return Key::ESCAPE;
        if (n == "ENTER") return Key::ENTER;
        if (n == "TAB") return Key::TAB;
        if (n == "SHIFT") return Key::LEFT_SHIFT;
        if (n == "CTRL" || n == "CONTROL") return Key::LEFT_CONTROL;
        return Key::UNKNOWN;
    }

}// namespace threepp

#endif//THREEPP_KEYFROMNAME_HPP
