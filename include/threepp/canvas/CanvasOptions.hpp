
#ifndef THREEPP_CANVASOPTIONS_HPP
#define THREEPP_CANVASOPTIONS_HPP

#include <threepp/canvas/WindowSize.hpp>

#include <filesystem>
#include <optional>
#include <variant>
#include <unordered_map>

namespace threepp {

    namespace detail {
        typedef std::variant<bool, int, std::string, WindowSize> ParameterValue;
    }

    struct CanvasOptions {

        WindowSize size_{640, 480};
        int antialiasing_{0};
        std::string title_{"threepp"};
        bool vsync_{true};
        bool resizable_{true};
        std::optional<std::filesystem::path> favicon_;

        CanvasOptions() = default;

        explicit CanvasOptions(const std::unordered_map<std::string, detail::ParameterValue>& values);

        CanvasOptions& title(std::string value);

        CanvasOptions& size(WindowSize size);

        CanvasOptions& size(int width, int height);

        CanvasOptions& antialiasing(int antialiasing);

        CanvasOptions& vsync(bool flag);

        CanvasOptions& resizable(bool flag);

        CanvasOptions& favicon(const std::filesystem::path& path);
    };

}

#endif//THREEPP_CANVASOPTIONS_HPP
