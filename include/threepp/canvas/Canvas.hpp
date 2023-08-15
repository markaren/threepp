
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include "threepp/input/PeripheralsEventSource.hpp"
#include "threepp/canvas/WindowSize.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace threepp {

    using MouseCaptureCallback = std::function<bool(void)>;
    using ScrollCaptureCallback = std::function<bool(void)>;
    using KeyboardCaptureCallback = std::function<bool(void)>;

    struct IOCapture {
        MouseCaptureCallback preventMouseEvent = [] { return false; };
        ScrollCaptureCallback preventScrollEvent = [] { return false; };
        KeyboardCaptureCallback preventKeyboardEvent = [] { return false; };
    };

    class Canvas: public PeripheralsEventSource {

    public:
        struct Parameters;
        typedef std::variant<bool, int, std::string, WindowSize> ParameterValue;

        explicit Canvas(const Parameters& params = Parameters());

        explicit Canvas(const std::string& name);

        Canvas(const std::string& name, const std::unordered_map<std::string, ParameterValue>& values);

        [[nodiscard]] const WindowSize& getSize() const;

        [[nodiscard]] float getAspect() const;

        void setSize(WindowSize size);

        void onWindowResize(std::function<void(WindowSize)> f);

        void setIOCapture(IOCapture* callback);

        void animate(const std::function<void()>& f);

        // returns false if application should quit, true otherwise
        bool animateOnce(const std::function<void()>& f);

        void invokeLater(const std::function<void()>& f, float t = 0);

        [[nodiscard]] void* windowPtr() const;

        ~Canvas() override;

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

            Parameters& resizable(bool flag);

            Parameters& favicon(const std::filesystem::path& path);

        private:
            WindowSize size_{640, 480};
            int antialiasing_{0};
            std::string title_{"threepp"};
            bool vsync_{true};
            bool resizable_{true};
            std::optional<std::filesystem::path> favicon_;

            friend struct Canvas::Impl;
        };
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
