
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include "threepp/input/KeyListener.hpp"
#include "threepp/input/MouseListener.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>


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
        typedef std::variant<bool, int, WindowSize> ParameterValue;

        explicit Canvas(const Parameters& params = Parameters());

        explicit Canvas(const std::string& name);

        Canvas(const std::string& name, const std::unordered_map<std::string, ParameterValue>& values);

        [[nodiscard]] const WindowSize& getSize() const;

        [[nodiscard]] float getAspect() const;

        [[nodiscard]] int getFPS() const;

        void setSize(WindowSize size);

        void onWindowResize(std::function<void(WindowSize)> f);

        void addKeyListener(KeyListener* listener);

        bool removeKeyListener(const KeyListener* listener);

        void addMouseListener(MouseListener* listener);

        bool removeMouseListener(const MouseListener* listener);

        void animate(const std::function<void()>& f);

        void animate(const std::function<void(float)>& f);

        void animate(const std::function<void(float, float)>& f);

        void invokeLater(const std::function<void()>& f, float t = 0);

        [[nodiscard]] void* window_ptr() const;

        ~Canvas();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

    public:
        struct Parameters {

            Parameters();

            explicit Parameters(const std::unordered_map<std::string, ParameterValue>& values);

            Parameters& title(std::string value);

            Parameters& size(WindowSize size);

            Parameters& size(int width, int height);

            Parameters& antialiasing(int antialiasing);

            Parameters& vsync(bool flag);

        private:
            WindowSize size_{640, 480};
            int antialiasing_{0};
            std::string title_{"threpp"};
            bool vsync_{true};

            friend struct Canvas::Impl;
        };
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
