
#ifndef THREEPP_KEYLISTENER_HPP
#define THREEPP_KEYLISTENER_HPP

#include <functional>
#include <utility>

namespace threepp {

    enum class Key;

    struct KeyEvent {

        const Key key;
        const int scancode;
        const int mods;

        KeyEvent(Key key, int scancode, int mods)
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

    enum class Key {

        UNKNOWN,
        SPACE,
        APOSTROPHE,
        COMMA,
        MINUS,
        PERIOD,
        SLASH,
        NUM_0,
        NUM_1,
        NUM_2,
        NUM_3,
        NUM_4,
        NUM_5,
        NUM_6,
        NUM_7,
        NUM_8,
        NUM_9,
        SEMICOLON,
        EQUAL,
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        N,
        O,
        P,
        Q,
        R,
        S,
        T,
        U,
        V,
        W,
        X,
        Y,
        Z,
        LEFT_BRACKET,  /* [ */
        BACKSLASH,     /* \ */
        RIGHT_BRACKET, /* ] */
        GRAVE_ACCENT,  /* ` */
        ESCAPE,
        ENTER,
        TAB,
        BACKSPACE,
        INSERT,
        DELETE,
        RIGHT,
        LEFT,
        DOWN,
        UP,
        PAGE_UP,
        PAGE_DOWN,
        HOME,
        END,
        CAPS_LOCK,
        SCROLL_LOCK,
        NUM_LOCK,
        PRINT_SCREEN,
        PAUSE,
        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,
        F13,
        F14,
        F15,
        F16,
        F17,
        F18,
        F19,
        F20,
        F21,
        F22,
        F23,
        F24,
        F25,
        KP_0,
        KP_1,
        KP_2,
        KP_3,
        KP_4,
        KP_5,
        KP_6,
        KP_7,
        KP_8,
        KP_9
    };

}// namespace threepp


#endif//THREEPP_KEYLISTENER_HPP
