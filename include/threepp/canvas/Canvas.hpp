
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include "threepp/canvas/WindowSize.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace threepp {

    class Canvas: public PeripheralsEventSource {

    public:
        struct Parameters;
        typedef std::variant<bool, int, std::string, WindowSize> ParameterValue;

        explicit Canvas(const Parameters& params = Parameters());

        explicit Canvas(const std::string& name);

        Canvas(const std::string& name, const std::unordered_map<std::string, ParameterValue>& values);

        //the current size of the Canvas window
        [[nodiscard]] WindowSize size() const override;

        [[nodiscard]] float aspect() const;

        void setSize(std::pair<int, int> size);

        void onWindowResize(std::function<void(WindowSize)> f);

        void animate(const std::function<void()>& f);

        // returns false if application should quit, true otherwise
        bool animateOnce(const std::function<void()>& f);

        [[nodiscard]] bool isOpen() const;

        void close();

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

            Parameters& exitOnKeyEscape(bool flag);

            Parameters& headless(bool flag);

        private:
            std::optional<WindowSize> size_;
            int antialiasing_{2};
            std::string title_{"threepp"};
            bool vsync_{true};
            bool resizable_{true};
            bool exitOnKeyEscape_{true};
            bool headless_{false};
            std::optional<std::filesystem::path> favicon_;

            friend struct Impl;
        };
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
