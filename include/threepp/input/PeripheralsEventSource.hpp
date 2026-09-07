
#ifndef THREEPP_PERIPHERALSEVENTSOURCE_HPP
#define THREEPP_PERIPHERALSEVENTSOURCE_HPP

#include "threepp/input/IOCapture.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/input/MouseListener.hpp"

#include "threepp/canvas/WindowSize.hpp"

#include <unordered_set>
#include <vector>

namespace threepp {

    class PeripheralsEventSource {

    public:
        [[nodiscard]] virtual WindowSize size() const = 0;

        void setIOCapture(IOCapture* capture);

        void addKeyListener(KeyListener& listener);

        bool removeKeyListener(const KeyListener& listener);

        // Owning callback registration — the event source holds the std::function, so
        // you can pass an inline lambda with no named listener object to keep alive
        // (mirrors three.js addEventListener). Returns an id for optional removal via
        // removeKeyListener(int); ignore it for fire-and-forget handlers that live
        // until the source is destroyed.
        int onKeyPressed(std::function<void(KeyEvent)> callback);
        int onKeyReleased(std::function<void(KeyEvent)> callback);
        int onKeyRepeat(std::function<void(KeyEvent)> callback);
        void removeKeyListener(int id);

        // Polling: is the key currently held? For continuous input (e.g. WASD), query
        // this per-frame instead of tracking press/release yourself. Reflects physical
        // key state even while IOCapture suppresses event dispatch (so it never sticks).
        [[nodiscard]] bool isKeyDown(Key key) const;

        void addMouseListener(MouseListener& listener);

        bool removeMouseListener(const MouseListener& listener);

        void onDrop(std::function<void(std::vector<std::string>)> paths);

        virtual ~PeripheralsEventSource() = default;

    protected:
        enum class KeyAction {
            PRESS,
            RELEASE,
            REPEAT
        };

        enum class MouseAction {
            PRESS,
            RELEASE
        };

        void onMousePressedEvent(int button, const Vector2& pos, MouseAction action);

        void onMouseMoveEvent(const Vector2& pos);

        void onMouseWheelEvent(const Vector2& eventData);

        void onKeyEvent(KeyEvent evt, KeyAction action);

        void onDropEvent(std::vector<std::string> paths);

    private:
        struct OwnedKeyListener {
            int id;
            KeyAction action;
            std::function<void(KeyEvent)> cb;
        };

        IOCapture* ioCapture_ = nullptr;
        std::vector<KeyListener*> keyListeners_;
        std::vector<OwnedKeyListener> ownedKeyListeners_;
        int nextKeyListenerId_ = 1;
        std::unordered_set<Key> keysDown_;// currently-held keys, for isKeyDown()
        std::vector<MouseListener*> mouseListeners_;
        std::function<void(std::vector<std::string>)> dropListener_;
    };

}// namespace threepp

#endif//THREEPP_PERIPHERALSEVENTSOURCE_HPP
