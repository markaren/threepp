
#ifndef THREEPP_KEYLISTENER_HPP
#define THREEPP_KEYLISTENER_HPP

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

        virtual void onKeyPressed(KeyEvent evt) {}

        virtual void onKeyReleased(KeyEvent evt) {}

        virtual void onKeyRepeat(KeyEvent evt) {}

        virtual ~KeyListener() = default;
    };

    struct KeyAdapter: KeyListener {

        enum Mode {
            KEY_PRESSED = 1,
            KEY_RELEASED = 2,
            KEY_REPEAT = 4
        };

        KeyAdapter(const Mode& mode, std::function<void(KeyEvent)> f)
            : mode_(mode), f_(std::move(f)) {}

        void onKeyPressed(KeyEvent evt) override {
            if (mode_ == 1 || mode_ == 3 || mode_ == 5) f_(evt);
        }

        void onKeyReleased(KeyEvent evt) override {
            if (mode_ == 2 || mode_ == 3 || mode_ == 6) f_(evt);
        }

        void onKeyRepeat(KeyEvent evt) override {
            if (mode_ == 4 || mode_ == 5 || mode_ == 6) f_(evt);
        }

    private:
        Mode mode_;
        std::function<void(KeyEvent)> f_;
    };


    inline KeyAdapter::Mode operator|(KeyAdapter::Mode a, KeyAdapter::Mode b) {
        return static_cast<KeyAdapter::Mode>(static_cast<int>(a) | static_cast<int>(b));
    }


}// namespace threepp


#endif//THREEPP_KEYLISTENER_HPP
