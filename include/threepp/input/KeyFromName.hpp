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

    // The inverse of keyFromName: the name a key EVENT reports to scripts (the
    // Python wheel's Canvas.on_key_pressed & co). Contract: for every key
    // keyFromName knows, keyFromName(keyName(k)) == k, so the name a callback
    // receives is one is_key_down accepts. Those keys get keyFromName's canonical
    // spelling ('A', '7', 'KP7', 'SPACE', 'ESCAPE', 'SHIFT' for LEFT_SHIFT, 'CTRL'
    // for LEFT_CONTROL); every other key gets its enum identifier ('F1',
    // 'BACKSPACE', 'PAGE_UP', 'LEFT_ALT', 'UNKNOWN').
    inline std::string keyName(Key key) {

        const int k = static_cast<int>(key);
        if (k >= static_cast<int>(Key::A) && k <= static_cast<int>(Key::Z))
            return std::string(1, static_cast<char>('A' + (k - static_cast<int>(Key::A))));
        if (k >= static_cast<int>(Key::NUM_0) && k <= static_cast<int>(Key::NUM_9))
            return std::string(1, static_cast<char>('0' + (k - static_cast<int>(Key::NUM_0))));
        if (k >= static_cast<int>(Key::KP_0) && k <= static_cast<int>(Key::KP_9))
            return "KP" + std::to_string(k - static_cast<int>(Key::KP_0));
        if (k >= static_cast<int>(Key::F1) && k <= static_cast<int>(Key::F25))
            return "F" + std::to_string(k - static_cast<int>(Key::F1) + 1);

        switch (key) {
            case Key::SPACE: return "SPACE";
            case Key::APOSTROPHE: return "APOSTROPHE";
            case Key::COMMA: return "COMMA";
            case Key::MINUS: return "MINUS";
            case Key::PERIOD: return "PERIOD";
            case Key::SLASH: return "SLASH";
            case Key::SEMICOLON: return "SEMICOLON";
            case Key::EQUAL: return "EQUAL";
            case Key::LEFT_BRACKET: return "LEFT_BRACKET";
            case Key::BACKSLASH: return "BACKSLASH";
            case Key::RIGHT_BRACKET: return "RIGHT_BRACKET";
            case Key::GRAVE_ACCENT: return "GRAVE_ACCENT";
            case Key::ESCAPE: return "ESCAPE";
            case Key::ENTER: return "ENTER";
            case Key::TAB: return "TAB";
            case Key::BACKSPACE: return "BACKSPACE";
            case Key::INSERT: return "INSERT";
            case Key::DEL: return "DEL";
            case Key::RIGHT: return "RIGHT";
            case Key::LEFT: return "LEFT";
            case Key::DOWN: return "DOWN";
            case Key::UP: return "UP";
            case Key::PAGE_UP: return "PAGE_UP";
            case Key::PAGE_DOWN: return "PAGE_DOWN";
            case Key::HOME: return "HOME";
            case Key::END: return "END";
            case Key::CAPS_LOCK: return "CAPS_LOCK";
            case Key::SCROLL_LOCK: return "SCROLL_LOCK";
            case Key::NUM_LOCK: return "NUM_LOCK";
            case Key::PRINT_SCREEN: return "PRINT_SCREEN";
            case Key::PAUSE: return "PAUSE";
            case Key::LEFT_SHIFT: return "SHIFT";
            case Key::LEFT_CONTROL: return "CTRL";
            case Key::LEFT_ALT: return "LEFT_ALT";
            default: return "UNKNOWN";
        }
    }

}// namespace threepp

#endif//THREEPP_KEYFROMNAME_HPP
