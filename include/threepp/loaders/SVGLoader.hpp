
#ifndef THREEPP_SVGLOADER_HPP
#define THREEPP_SVGLOADER_HPP

#include <filesystem>
#include <memory>
#include <string>

#include "threepp/objects/Group.hpp"

namespace threepp {

    class SVGLoader {

    public:
        float defaultDPI = 90;
        // Accepted units: 'mm', 'cm', 'in', 'pt', 'pc', 'px'
        std::string defaultUnit = "px";

        std::shared_ptr<Group> load(const std::filesystem::path& path);

        std::shared_ptr<Group> parse(std::string text);
    };

}// namespace threepp

#endif//THREEPP_SVGLOADER_HPP
