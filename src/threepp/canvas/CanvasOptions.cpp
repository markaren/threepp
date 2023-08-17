
#include "threepp/canvas/CanvasOptions.hpp"

#include "threepp/utils/StringUtils.hpp"

#include <iostream>

using namespace threepp;


CanvasOptions::CanvasOptions(const std::unordered_map<std::string, detail::ParameterValue>& values) {
    
    std::vector<std::string> unused;
    for (const auto& [key, value] : values) {

        bool used = false;

        if (key == "antialiasing") {

            antialiasing(std::get<int>(value));
            used = true;

        } else if (key == "vsync") {

            vsync(std::get<bool>(value));
            used = true;

        } else if (key == "resizable") {

            resizable(std::get<bool>(value));
            used = true;

        } else if (key == "size") {

            auto _size = std::get<WindowSize>(value);
            size(_size);
            used = true;
        } else if (key == "favicon") {

            auto path = std::get<std::string>(value);
            favicon(path);
            used = true;
        }

        if (!used) {
            unused.emplace_back(key);
        }
    }

    if (!unused.empty()) {

        std::cerr << "Unused Canvas parameters: [" << utils::join(unused, ',') << "]" << std::endl;
    }
}


CanvasOptions& CanvasOptions::title(std::string value) {

    this->title_ = std::move(value);

    return *this;
}

CanvasOptions& CanvasOptions::size(WindowSize size) {

    this->size_ = size;

    return *this;
}

CanvasOptions& CanvasOptions::size(int width, int height) {

    return this->size({width, height});
}

CanvasOptions& CanvasOptions::antialiasing(int antialiasing) {

    this->antialiasing_ = antialiasing;

    return *this;
}

CanvasOptions& CanvasOptions::vsync(bool flag) {

    this->vsync_ = flag;

    return *this;
}

CanvasOptions& threepp::CanvasOptions::resizable(bool flag) {

    this->resizable_ = flag;

    return *this;
}

CanvasOptions& threepp::CanvasOptions::favicon(const std::filesystem::path& path) {

    if (std::filesystem::exists(path)) {
        favicon_ = path;
    } else {
        std::cerr << "Invalid favicon path: " << std::filesystem::absolute(path) << std::endl;
    }

    return *this;
}
