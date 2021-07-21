
#ifndef THREEPP_KEYLISTENER_HPP
#define THREEPP_KEYLISTENER_HPP

#include "threepp/utils/uuid.hpp"

#include <functional>
#include <utility>

namespace threepp {

    struct KeyEvent {

        const int key;
        const int scancode;
        const int mods;

        KeyEvent(int key, int scancode, int mods)
            : key(key), scancode(scancode), mods(mods) {}
    };

    struct KeyListener {

        const std::string uuid = utils::generateUUID();

        virtual void onKeyPressed(KeyEvent evt) {}

        virtual void onKeyReleased(KeyEvent evt) {}

        virtual void onKeyRepeat(KeyEvent evt) {}

        virtual ~KeyListener() = default;
    };

    struct KeyAdapter: KeyListener {

        enum Mode {
            KEY_PRESSED,
            KEY_RELEASED,
            KEY_REPEAT
        };

        KeyAdapter(const Mode &mode, std::function<void(KeyEvent)> f)
            : mode_(mode), f_(std::move(f)) {}

        void onKeyPressed(KeyEvent evt) override {
            if (mode_ == KEY_PRESSED) f_(evt);
        }

        void onKeyReleased(KeyEvent evt) override {
            if (mode_ == KEY_RELEASED) f_(evt);
        }

        void onKeyRepeat(KeyEvent evt) override {
            if (mode_ == KEY_REPEAT) f_(evt);
        }

    private:
        const Mode mode_;
        std::function<void(KeyEvent)> f_;

    };

}// namespace threepp


#endif//THREEPP_KEYLISTENER_HPP
