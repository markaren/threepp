
#ifndef THREEPP_KEYLISTENER_HPP
#define THREEPP_KEYLISTENER_HPP

#include <functional>
#include <utility>

namespace threepp {

    struct MouseEvent {

        const int button;
        const int action;
        const int mods;

        MouseEvent(const int button, const int action, const int mods)
            : button(button), action(action), mods(mods) {}
    };

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
    };

    struct KeyPressAdapter : KeyListener {

        explicit KeyPressAdapter(std::function<void(KeyEvent)> f)
            : f(std::move(f)) {}

        void onKeyPressed(KeyEvent evt) override {
            f(evt);
        }

    private:
        std::function<void(KeyEvent)> f;
    };

}// namespace threepp


#endif//THREEPP_KEYLISTENER_HPP
