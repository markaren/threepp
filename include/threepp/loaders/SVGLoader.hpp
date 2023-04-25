
#ifndef THREEPP_SVGLOADER_HPP
#define THREEPP_SVGLOADER_HPP

#include <string>
#include <filesystem>
#include <memory>

#include "threepp/extras/core/ShapePath.hpp"

namespace threepp {

    class SVGLoader {

    public:
        float defaultDPI = 90;
        // Accepted units: 'mm', 'cm', 'in', 'pt', 'pc', 'px'
        std::string defaultUnit = "px";

//        SVGLoader();

        std::vector<ShapePath> load(const std::filesystem::path& path);

        std::vector<ShapePath> parse(std::string text);

//        ~SVGLoader();

    private:
//        struct Impl;
//        std::unique_ptr<Impl> pimpl_;

    };

}// namespace threepp

#endif//THREEPP_SVGLOADER_HPP
