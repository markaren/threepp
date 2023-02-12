
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include <functional>
#include <memory>

#include "threepp/core/Clock.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/input/MouseListener.hpp"

namespace threepp {

    struct WindowSize {
        int width;
        int height;

        [[nodiscard]] float getAspect() const {
            return static_cast<float>(width) / static_cast<float>(height);
        }
    };

    class Canvas {

    public:
        struct Parameters;

        explicit Canvas(const Parameters &params = Parameters());

        [[nodiscard]] const WindowSize& getSize() const;

        [[nodiscard]] float getAspect() const;

        void setSize(WindowSize size);

        void onWindowResize(std::function<void(WindowSize)> f);

        void addKeyListener(KeyListener *listener);

        bool removeKeyListener(const KeyListener *listener);

        void addMouseListener(MouseListener* listener);

        bool removeMouseListener(const MouseListener* listener);

        void animate(const std::function<void()> &f);

        void animate(const std::function<void(float)> &f);

        void invokeLater(const std::function<void()>& f);

        [[nodiscard]] void* window_ptr() const;

        ~Canvas();

    private:
        class Impl;
        std::unique_ptr<Impl> pimpl_;

    public:
        struct Parameters {

            Parameters()
                : size_{640, 480},
                  antialiasing_{0},
                  title_{"threepp"} {}

            Parameters &title(std::string value);

            Parameters &size(WindowSize size);

            Parameters &size(int width, int height);

            Parameters &antialiasing(int antialiasing);

        private:
            WindowSize size_;
            int antialiasing_;
            std::string title_;

            friend class Canvas::Impl;
        };
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
