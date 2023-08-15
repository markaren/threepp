
#ifndef THREEPP_PERIPHERALSEVENTSOURCE_HPP
#define THREEPP_PERIPHERALSEVENTSOURCE_HPP

#include "threepp/input/KeyListener.hpp"
#include "threepp/input/MouseListener.hpp"

#include "threepp/canvas/WindowSize.hpp"

#include <vector>

namespace threepp {

    class PeripheralsEventSource {

    public:

        void addKeyListener(KeyListener* listener);

        bool removeKeyListener(const KeyListener* listener);

        void addMouseListener(MouseListener* listener);

        bool removeMouseListener(const MouseListener* listener);

        virtual ~PeripheralsEventSource() = default;

    protected:
        std::vector<KeyListener*> keyListeners;
        std::vector<MouseListener*> mouseListeners;
    };

}

#endif//THREEPP_PERIPHERALSEVENTSOURCE_HPP
